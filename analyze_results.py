import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# 1. Load Results
try:
    df = pd.read_csv('erbb2leads.csv')
    print("Successfully loaded dataset")
except FileNotFoundError:
    print("Error: Could not find dataset")
    exit()

def check_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return False, 0, 0, 0, 0
    
    mw = Descriptors.MolWt(mol)          
    logp = Descriptors.MolLogP(mol)      
    hbd = Descriptors.NumHDonors(mol)    
    hba = Descriptors.NumHAcceptors(mol) 
    
    # Lipinski's Rule of 5 
    conditions = [mw <= 500, logp <= 5, hbd <= 5, hba <= 10]
    is_lipinski = sum(conditions) == 4
    return is_lipinski, mw, logp, hbd, hba

# Process and Filter
results = df['SMILES'].apply(check_lipinski)
df[['is_druglike', 'MW', 'LogP', 'HBD', 'HBA']] = pd.DataFrame(results.tolist(), index=df.index)

# Remove duplicates 
df_unique = df.drop_duplicates(subset=['SMILES'])

# Sort by pKd
real_leads = df_unique[df_unique['is_druglike'] == True].sort_values(by='pKd', ascending=False)

# Pick Top 6 Leads
portfolio = real_leads.head(6)
portfolio.to_csv('ai_candidate_portfolio.csv', index=False)

# 4. Generate the Gallery
mols = [Chem.MolFromSmiles(s) for s in portfolio['SMILES']]
labels = [
    f"pKd: {row['pKd']:.2f}\nMW: {row['MW']:.1f}, LogP: {row['LogP']:.1f}" 
    for _, row in portfolio.iterrows()
]

img_data = Draw.MolsToGridImage(
    mols, 
    legends=labels, 
    subImgSize=(400, 400),
    molsPerRow=3,
    useSVG=True
)

with open('image_of_leads.svg', 'w') as f:
    f.write(img_data)

print(f"Analysis Complete.")
print(f"Found {len(real_leads)} unique druglike leads.")
print(f"Top Unique Lead pKd: {portfolio.iloc[0]['pKd']:.4f}")
