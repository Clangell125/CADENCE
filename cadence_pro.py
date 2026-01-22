import sys, os, random, gc, json, multiprocessing
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QProgressBar, QFrame, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage, QFont

# RDkit and AI Imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, BRICS, Draw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras import backend as K
from tdc.multi_pred import DTI

# --- SCIENCE LOGIC ---
def get_metrics(mol):
    if not mol: return {"MW":0, "LOGP":0, "HBD":0, "HBA":0, "SA":10, "PENALTY": 4}
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        sa = round(2 + (Descriptors.HeavyAtomCount(mol) * 0.1) + (Lipinski.NumRotatableBonds(mol) * 0.1), 2)
        penalty = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        return {"MW": round(mw, 1), "LOGP": round(logp, 1), "HBD": hbd, 
                "HBA": hba, "SA": sa, "PENALTY": penalty}
    except: return {"MW":0, "LOGP":0, "HBD":0, "HBA":0, "SA":10, "PENALTY": 4}

class DiscoveryEngine(QThread):
    progress_update = pyqtSignal(int, str)
    leads_received = pyqtSignal(list)
    finished_signal = pyqtSignal()

    def __init__(self, sequence, gens):
        super().__init__()
        self.sequence, self.gens = sequence, gens
        self.final_leads = []

    def run(self):
        try:
            self.progress_update.emit(0, "Warming up Neural Network...")
            model = load_model('CADENCE.keras', compile=False)
            
            AA_MAP = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            t_ints = [AA_MAP.get(aa, 0) for aa in self.sequence.upper()[:1000]] + [0]*1000
            target_vec = np.array(t_ints[:1000], dtype='float32').reshape(1, 1000)

            df = DTI(name='Davis').get_data()
            current_pop = list(set(df['Drug'].tolist()))[:150] 

            for gen in range(self.gens):
                self.progress_update.emit(gen, f"Refining Leads: Generation {gen+1} of {self.gens}")
                unique_smiles = list(set(current_pop))
                mols = [Chem.MolFromSmiles(s) for s in unique_smiles if s]
                fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype='int8') for m in mols if m]
                
                preds = model.predict([np.array(fps, dtype='float32'), np.repeat(target_vec, len(fps), axis=0)], batch_size=64, verbose=0)
                pkds = (preds.flatten() * 5.79) + 5.0
                
                results = []
                for i, m in enumerate(mols):
                    met = get_metrics(m)
                    fit = pkds[i] - (met['SA'] * 0.35) - (met['PENALTY'] * 1.0)
                    res = {'SMILES': Chem.MolToSmiles(m), 'PKD': f"{pkds[i]:.2f}", 'FIT': fit}
                    res.update(met)
                    results.append(res)
                
                res_df = pd.DataFrame(results).sort_values(by='FIT', ascending=False).drop_duplicates(subset=['SMILES'])
                self.final_leads = res_df.head(6).to_dict(orient='records')
                self.leads_received.emit(self.final_leads)
                
                next_gen = res_df.head(20)['SMILES'].tolist()
                current_pop = next_gen * 5 + [random.choice(df['Drug'].tolist()) for _ in range(20)]
                K.clear_session(); gc.collect()

            self.progress_update.emit(self.gens, "Discovery Complete.")
            self.finished_signal.emit()
        except Exception as e: print(f"Error: {e}")

class LeadCard(QFrame):
    def __init__(self, data):
        super().__init__()
        self.setStyleSheet("background-color: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 8px;")
        self.setFixedSize(400, 350) # WIDER CARD
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)
        
        # 2D Structure
        img_lbl = QLabel()
        mol = Chem.MolFromSmiles(data.get('SMILES', ''))
        if mol:
            d = Draw.MolDraw2DCairo(360, 150) # Scaled for width
            d.drawOptions().useBWAtomColors = True 
            d.drawOptions().bondLineWidth = 2
            d.DrawMolecule(mol)
            d.FinishDrawing()
            img_lbl.setPixmap(QPixmap.fromImage(QImage.fromData(d.GetDrawingText())))
        layout.addWidget(img_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Header Info
        header_row = QHBoxLayout()
        pkd_lbl = QLabel(f"pKd: {data.get('PKD', '0.00')}")
        pkd_lbl.setStyleSheet("color: #007BFF; font-size: 20px; font-weight: bold; border: none;")
        header_row.addWidget(pkd_lbl)
        
        viol_lbl = QLabel(f"Violations: {data.get('PENALTY', 0)}")
        viol_lbl.setStyleSheet("color: #DC3545; font-size: 12px; font-weight: bold; border: none;")
        header_row.addWidget(viol_lbl, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_row)
        
        # SMILES (Expanded width allows for more text)
        smiles_txt = data.get('SMILES', '')
        smiles_lbl = QLabel(f"SMILES: {smiles_txt[:55]}..." if len(smiles_txt) > 55 else f"SMILES: {smiles_txt}")
        smiles_lbl.setStyleSheet("font-size: 10px; color: #777; border: none;")
        layout.addWidget(smiles_lbl)

        # Metric Grid (Now much cleaner in 3 columns)
        m_grid = QGridLayout()
        m_grid.setSpacing(10)
        
        metrics = [
            (f"MW: {data.get('MW', 0)}", 0, 0), (f"LogP: {data.get('LOGP', 0)}", 0, 1), (f"SA: {data.get('SA', 0)}", 0, 2),
            (f"HBD: {data.get('HBD', 0)}", 1, 0), (f"HBA: {data.get('HBA', 0)}", 1, 1)
        ]
        
        for text, r, c in metrics:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #333; font-size: 12px; font-weight: 600; border: none;")
            m_grid.addWidget(lbl, r, c)
            
        layout.addLayout(m_grid)

class CadenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CADENCE: Kinase Engine")
        self.setMinimumSize(1300, 900) # Increased window width
        self.setStyleSheet("background-color: #E9ECEF;") 
        widget = QWidget(); self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        title = QLabel("CADENCE: Kinase Engine")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #212529; padding: 10px;")
        layout.addWidget(title)

        input_row = QHBoxLayout()
        self.seq_in = QLineEdit(); self.seq_in.setPlaceholderText("Paste Protein Sequence...")
        self.seq_in.setStyleSheet("background: white; border: 1px solid #CCC; padding: 10px; color: black; border-radius: 4px;")
        input_row.addWidget(self.seq_in, 8)
        
        gen_lbl = QLabel("Gens:"); gen_lbl.setStyleSheet("color: #212529; font-weight: bold;")
        input_row.addWidget(gen_lbl)
        self.gen_in = QLineEdit("20"); self.gen_in.setFixedWidth(60)
        self.gen_in.setStyleSheet("background: white; border: 1px solid #CCC; color: black; text-align: center; font-weight: bold;")
        input_row.addWidget(self.gen_in)
        layout.addLayout(input_row)

        self.grid_container = QWidget(); self.grid = QGridLayout(self.grid_container)
        self.grid.setSpacing(20) # More space between cards
        layout.addWidget(self.grid_container)

        self.pbar = QProgressBar()
        self.pbar.setStyleSheet("QProgressBar { background: #DDD; height: 12px; border-radius: 6px; } QProgressBar::chunk { background: #007BFF; border-radius: 6px; }")
        layout.addWidget(self.pbar)
        
        self.status = QLabel("System Ready"); self.status.setStyleSheet("color: #666; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignCenter)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("EXECUTE OPTIMIZATION")
        self.run_btn.setFixedHeight(50); self.run_btn.setStyleSheet("background: #007BFF; color: white; font-weight: bold; border-radius: 4px;")
        self.run_btn.clicked.connect(self.run)
        
        self.export_btn = QPushButton("EXPORT DATA")
        self.export_btn.setFixedHeight(50); self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("background: #28A745; color: white; font-weight: bold; border-radius: 4px;")
        self.export_btn.clicked.connect(self.export_results)

        self.stop_btn = QPushButton("CANCEL")
        self.stop_btn.setFixedHeight(50); self.stop_btn.setStyleSheet("background: #DC3545; color: white; font-weight: bold; border-radius: 4px;")
        self.stop_btn.clicked.connect(self.stop)
        
        btns.addWidget(self.run_btn, 3); btns.addWidget(self.export_btn, 2); btns.addWidget(self.stop_btn, 1)
        layout.addLayout(btns)

    def run(self):
        seq = self.seq_in.text().strip(); gens = int(self.gen_in.text()) if self.gen_in.text().isdigit() else 20
        self.pbar.setMaximum(gens); self.run_btn.setEnabled(False); self.export_btn.setEnabled(False)
        self.engine = DiscoveryEngine(seq, gens)
        self.engine.progress_update.connect(lambda v, t: (self.pbar.setValue(v), self.status.setText(t)))
        self.engine.leads_received.connect(self.update_grid)
        self.engine.finished_signal.connect(self.on_finish); self.engine.start()

    def update_grid(self, leads):
        for i in reversed(range(self.grid.count())): self.grid.itemAt(i).widget().setParent(None)
        for idx, lead in enumerate(leads): self.grid.addWidget(LeadCard(lead), idx // 3, idx % 3)

    def on_finish(self):
        self.run_btn.setEnabled(True); self.export_btn.setEnabled(True)
        self.status.setText("Lead Optimization Successful.")

    def export_results(self):
        if not hasattr(self, 'engine'): return
        os.makedirs("CADENCE_RESULTS", exist_ok=True)
        pd.DataFrame(self.engine.final_leads).to_csv("CADENCE_RESULTS/discovery_summary.csv", index=False)
        for i, lead in enumerate(self.engine.final_leads):
            mol = Chem.MolFromSmiles(lead['SMILES'])
            if mol: Draw.MolToFile(mol, f"CADENCE_RESULTS/lead_{i+1}.png", size=(600, 400))
        self.status.setText("Export complete: /CADENCE_RESULTS")

    def stop(self):
        if hasattr(self, 'engine'): self.engine.terminate()
        self.status.setText("Simulation Stopped."); self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv); window = CadenceApp(); window.show(); sys.exit(app.exec())