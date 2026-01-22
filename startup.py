import sys, os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QComboBox, QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# --- THE SELECTION SCREEN TEMPLATE ---
class SelectionScreen(QWidget):
    def __init__(self, title, options, launch_callback, back_callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)

        # Title
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("font-size: 28px; font-weight: 700; color: #2C3E50; margin-bottom: 10px;")
        layout.addWidget(t_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        s_lbl = QLabel("Select target enzyme class to initialize environment")
        s_lbl.setStyleSheet("font-size: 14px; color: #7F8C8D; margin-bottom: 30px;")
        layout.addWidget(s_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        # Dropdown
        self.dropdown = QComboBox()
        self.dropdown.addItems(options)
        self.dropdown.setFixedSize(400, 45)
        self.dropdown.setStyleSheet("""
            QComboBox {
                border: 2px solid #DCDDE1; border-radius: 8px; padding: 5px 15px;
                font-size: 14px; background: white; color: #2F3640;
            }
            QComboBox::drop-down { border: none; }
        """)
        layout.addWidget(self.dropdown, alignment=Qt.AlignmentFlag.AlignCenter)

        # Launch Button
        self.launch_btn = QPushButton("INITIALIZE MODULE")
        self.launch_btn.setFixedSize(400, 50)
        self.launch_btn.setStyleSheet("""
            QPushButton {
                background-color: #2F3640; color: white; border-radius: 8px;
                font-weight: 800; font-size: 13px; margin-top: 20px;
            }
            QPushButton:hover { background-color: #007BFF; }
        """)
        self.launch_btn.clicked.connect(lambda: launch_callback(self.dropdown.currentText()))
        layout.addWidget(self.launch_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Back Button
        back_btn = QPushButton("Return to Suite Selection")
        back_btn.setFlat(True)
        back_btn.setStyleSheet("color: #95A5A6; font-size: 12px; margin-top: 15px;")
        back_btn.clicked.connect(back_callback)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignCenter)

# --- THE MAIN HUB ---
class CadenceHub(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CADENCE Enterprise Hub")
        self.setFixedSize(1000, 650)
        self.setStyleSheet("background-color: #FFFFFF;")
        
        self.active_instances = [] # Prevent garbage collection

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 0: Main Hub Screen
        self.init_main_hub()
        
        # 1: Generative Selection
        self.gen_select = SelectionScreen(
            "GENERATIVE MODELS", 
            ["Kinase (v3.5 Deployed)", "GPCR (In Training)", "Protease (Library Dev)"],
            self.boot_generative, 
            self.show_hub
        )
        self.stack.addWidget(self.gen_select)

        # 2: Predictive Selection
        self.pred_select = SelectionScreen(
            "PREDICTIVE TOOLS", 
            ["DTI Affinity Scoring", "Pharmacophore Mapping", "Toxicity Prediction"],
            self.boot_predictive, 
            self.show_hub
        )
        self.stack.addWidget(self.pred_select)

    def init_main_hub(self):
        hub_widget = QWidget()
        layout = QVBoxLayout(hub_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(60, 60, 60, 60)

        # Branding
        title = QLabel("CADENCE DISCOVERY SUITE")
        title.setStyleSheet("font-size: 42px; font-weight: 900; color: #1A1A1A; letter-spacing: 1px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        tagline = QLabel("Stealth-Stage BioTech Lead Optimization Platform")
        tagline.setStyleSheet("font-size: 15px; color: #888; margin-bottom: 40px;")
        layout.addWidget(tagline, alignment=Qt.AlignmentFlag.AlignCenter)

        # Grid of Cards
        grid = QHBoxLayout()
        grid.setSpacing(20)
        
        self.btn_gen = self.make_card("GENERATIVE", "Kinase Optimized", "#007BFF")
        self.btn_pre = self.make_card("PREDICTIVE", "Affinity Scoring", "#28A745")
        self.btn_val = self.make_card("VALIDATION", "Lipinski Checker", "#6C757D")

        self.btn_gen.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_pre.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        
        grid.addWidget(self.btn_gen)
        grid.addWidget(self.btn_pre)
        grid.addWidget(self.btn_val)
        layout.addLayout(grid)
        self.stack.addWidget(hub_widget)

    def make_card(self, title, sub, color):
        btn = QPushButton()
        btn.setFixedSize(260, 200)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: white; border: 2px solid #F0F0F0; border-radius: 12px;
            }}
            QPushButton:hover {{
                border: 2px solid {color};
            }}
        """)
        l = QVBoxLayout(btn)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        t = QLabel(title); t.setStyleSheet(f"font-weight: 900; font-size: 18px; color: #212529; border:none;")
        s = QLabel(sub); s.setStyleSheet(f"font-size: 12px; color: {color}; border:none;")
        
        l.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)
        l.addWidget(s, alignment=Qt.AlignmentFlag.AlignCenter)
        return btn

    def show_hub(self):
        self.stack.setCurrentIndex(0)

    def boot_generative(self, selection):
        if "Kinase" in selection:
            try:
                from cadence_pro import CadenceApp # Import your main engine
                new_app = CadenceApp()
                self.active_instances.append(new_app)
                new_app.show()
            except ImportError:
                print("Error: cadence_pro.py not found in directory.")
        else:
            print(f"{selection} model is currently in development.")

    def boot_predictive(self, selection):
        print(f"Launching Predictive Module: {selection}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set a clean global font
    app.setFont(QFont("Inter", 10)) 
    hub = CadenceHub()
    hub.show()
    sys.exit(app.exec())