import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import load_model

# Encode the data
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: return np.zeros((n_bits,))
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except:
        return np.zeros((n_bits,))

def protein_to_vec(sequence, max_len=1000):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_int = {aa: i + 1 for i, aa in enumerate(amino_acids)}
    vec = np.zeros(max_len)
    for i, aa in enumerate(str(sequence)[:max_len]):
        vec[i] = aa_to_int.get(aa, 0)
    return vec


print("Loading Dataset...")
balanced_df = pd.read_csv("davis_clean.csv")

# Load dataset
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
test_df = test_df.reset_index(drop=True) 

print(f"Processing {len(test_df)} samples...")
X_d_ts = np.array([smiles_to_fp(s) for s in test_df['Drug']])
X_e_ts = np.array([protein_to_vec(s) for s in test_df['Target']])
y_ts = test_df['affinity_norm'].values

# Load the model
print("Loading CADENCE ...")
model = load_model('CADENCE.keras')
preds = model.predict([X_d_ts, X_e_ts], verbose=1).flatten()

# Denormalize
y_ts_pkd = (y_ts * 5.8) + 5.0
preds_pkd = (preds * 5.8) + 5.0



# A. Global Metrics
global_mae = mean_absolute_error(y_ts_pkd, preds_pkd)
global_r2 = r2_score(y_ts_pkd, preds_pkd)

# B. Nonzero Filtering
mask = y_ts > 0
active_y = y_ts_pkd[mask]
active_preds = preds_pkd[mask]

if len(active_y) > 0:
    active_mae = mean_absolute_error(active_y, active_preds)
    active_r2 = r2_score(active_y, active_preds)
else:
    active_mae, active_r2 = 0, 0



print("\n" + "="*50)
print(f"Global Dataset Performance (n={len(y_ts)})")
print(f"Global MAE: {global_mae:.4f} pKd")
print(f"Global R²:   {global_r2:.4f}")
print(f"Global MSE:  {mean_squared_error(y_ts_pkd, preds_pkd):.4f}")
print("-"*50)
print(f"Active Only Dataset Performance (n={len(active_y)})")
print(f"Active MAE: {active_mae:.4f} pKd")
print(f"Active R²:   {active_r2:.4f}")
print(f"Active MSE:  {mean_squared_error(active_y, active_preds):.4f}")
print("="*50)

