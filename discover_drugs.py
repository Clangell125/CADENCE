import os
import random
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, DataStructs
from tdc.multi_pred import DTI
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. ROBUST WORKER ---
def safe_chaos_worker(smi, f_pool, m_rate):
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol: return None
        
        if m_rate > 0 and random.random() < m_rate:
            frags = list(BRICS.BRICSDecompose(mol))
            if frags:
                frags[random.randint(0, len(frags)-1)] = random.choice(f_pool)
                mol = next(BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in frags if f], ouch=False))
        
        if mol:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            final_smi = Chem.MolToSmiles(mol)
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype='int8')
            return (final_smi, fp)
    except:
        return None
    return None

if __name__ == "__main__":
    POP_SIZE = 1024 
    GENS = 15
    ELITE_SIZE = 15 # Increased slightly to allow more diverse parents
    SIM_THRESHOLD = 0.75 # Molecules more than 75% similar are considered "duplicates"
    MAX_SEQ_LEN = 1000
    Y_MIN, Y_MAX = 5.0, 10.79588

    print(f"🚀 Initializing Diversity-Aware Chaos Engine (Target: ERBB2)")
    model = load_model('CADENCE.keras', compile=False)
    
    ERBB2_SEQ = "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERLPQPPICTIDVYMIMVKCWMIDSECRPRFRELVSEFSRMARDPQRFVVIQNEDLGPASPLDSTFYRSLLEDDDMGDLVDAEEYLVPQQGFFCPDPAPGAGGMVHHRHRSSSTRSGGGDLTLGLEPSEEEAPRSPLAPSEGAGSDVFDGDLGMGAAKGLQSLPTHDPSPLQRYSEDPTVPLPSETDGYVAPLTCSPQPEYVNQPDVRPQPPSPREGPLPAARPAGATLERPKTLSPGKNGVVKDVFAFGGAVENPEYLTPQGGAAPQPHPPPAFSPAFDNLYYWDQDPPERGAPPSTFKGTPTAENPEYLGLDVPV"
    AA_MAP = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    t_ints = [AA_MAP.get(aa, 0) for aa in ERBB2_SEQ.upper()[:MAX_SEQ_LEN]]
    t_ints += [0] * (MAX_SEQ_LEN - len(t_ints))
    target_vec = np.array(t_ints, dtype='float32').reshape(1, MAX_SEQ_LEN)

    df = DTI(name='Davis').get_data()
    f_pool = []
    for s in df.sort_values(by='Y').head(100)['Drug']:
        m = Chem.MolFromSmiles(s)
        if m: f_pool.extend(BRICS.BRICSDecompose(m))
    f_pool = list(set(f_pool))

    current_pop = [random.choice(df['Drug'].tolist()) for _ in range(POP_SIZE)]

    for gen in range(GENS):
        next_gen_results = []
        m_rate = 0.65 if gen > 0 else 0.0 
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(safe_chaos_worker, s, f_pool, m_rate): s for s in current_pop}
            for fut in tqdm(as_completed(futures), total=POP_SIZE, desc=f"Gen {gen} CPU", leave=False):
                try:
                    res = fut.result(timeout=15)
                    if res: next_gen_results.append(res)
                except: continue

        if not next_gen_results:
            print("❌ Population collapse.")
            break

        # Prediction
        s_batch = [x[0] for x in next_gen_results]
        # Keep fingerprints as objects for similarity calculation
        fps_batch = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in s_batch]
        v_batch = np.array([np.array(f, dtype='int8') for f in fps_batch], dtype='float32')
        
        t_input = np.repeat(target_vec, len(v_batch), axis=0)
        preds = model.predict([v_batch, t_input], batch_size=512, verbose=0)
        pkds = (preds.flatten() * (Y_MAX - Y_MIN)) + Y_MIN
        
        # --- NEW DIVERSITY-BASED SELECTION ---
        res_df = pd.DataFrame({'SMILES': s_batch, 'pKd': pkds, 'fp': fps_batch})
        res_df = res_df.sort_values(by='pKd', ascending=False)

        elites = []
        elite_fps = []

        for _, row in res_df.iterrows():
            # Check similarity against already selected elites
            is_duplicate = False
            for e_fp in elite_fps:
                sim = DataStructs.TanimotoSimilarity(row['fp'], e_fp)
                if sim > SIM_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                elites.append(row['SMILES'])
                elite_fps.append(row['fp'])
            
            if len(elites) >= ELITE_SIZE:
                break

        print(f"Gen {gen} | Top pKd: {res_df.iloc[0]['pKd']:.4f} | Elites Found: {len(elites)}")

        # Re-populate
        mutated_parents = (elites * (POP_SIZE // len(elites) + 1))[:POP_SIZE // 2]
        
        fresh_blood = []
        gen_brics = BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in f_pool])
        while len(fresh_blood) < (POP_SIZE // 2):
            try: fresh_blood.append(Chem.MolToSmiles(next(gen_brics)))
            except: break
        
        current_pop = mutated_parents + fresh_blood
        
        del v_batch, t_input, next_gen_results
        K.clear_session()
        gc.collect()

    # Final Save
    res_df[['SMILES', 'pKd']].to_csv("generated_leads.csv", index=False)
    print("✅ Discovery Complete. Results saved with diversity filter.")