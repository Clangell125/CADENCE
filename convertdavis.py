import pandas as pd
import numpy as np
from tdc.multi_pred import DTI

# Get raw data
print("Fetching Davis dataset from TDC...")
data = DTI(name = 'DAVIS')

# convert to pandas
df = data.get_data()

# Turn y into Kd (in nM)
df = df.rename(columns={'Drug': 'Drug', 'Target': 'Target', 'Y': 'kd_nm'})

# Convert to pkd, add a small value to avoid log explosion
print("🧪onverting Kd to pKd...")
df['pkd'] = 9 - np.log10(df['kd_nm'] + 1e-9)

# Standardize data such that it is between 0 and 1 for training purposes
df['pkd'] = df['pkd'].clip(lower=5.0, upper=10.8)
DAVIS_MIN, DAVIS_MAX = 5.0, 10.8
df['affinity_norm'] = (df['pkd'] - DAVIS_MIN) / (DAVIS_MAX - DAVIS_MIN)

# 7. Save the cleaned file
df.to_csv('davis_clean.csv', index=False)

print(f"\nSuccess! Saved {len(df)} Davis interactions to 'davis_clean.csv'")