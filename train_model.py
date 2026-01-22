import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, Embedding, 
                                     Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, 
                                     concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


DAVIS_MIN, DAVIS_MAX = 5.0, 10.79588
DAVIS_RANGE = DAVIS_MAX - DAVIS_MIN

# Load clean Davis data
df = pd.read_csv('davis_clean.csv')
df = df.drop_duplicates(subset=['Drug', 'Target'])

AA_CODES = { "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, 
             "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20 }

def protein_to_int(seq):
    vec = [AA_CODES.get(aa, 0) for aa in str(seq).upper()[:1000]]
    return vec + [0] * (1000 - len(vec))

# --- 2. FEATURIZATION ---
print(f"Featurizing {len(df)} unique pairs...")
drug_feats, enzyme_feats = [], []
for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['Drug'])
    if mol:
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)).astype('float32')
        drug_feats.append(fp)
        enzyme_feats.append(protein_to_int(row['Target']))

# --- 3. MONITOR STRINGS ---
GLEEVEC = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
ABL1 = "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"

class DiscoveryMonitor(tf.keras.callbacks.Callback):
    def __init__(self, gleevec_smiles, aspirin_smiles, protein_seq):
        super().__init__()
        self.gleevec_fp = self._get_fp(gleevec_smiles)
        self.aspirin_fp = self._get_fp(aspirin_smiles)
        self.prot_vec = np.array([protein_to_int(protein_seq)])

    def _get_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)]).astype('float32')

    def on_epoch_end(self, epoch, logs=None):
        p_glee = self.model.predict([self.gleevec_fp, self.prot_vec], verbose=0)[0][0]
        p_aspi = self.model.predict([self.aspirin_fp, self.prot_vec], verbose=0)[0][0]
        val_glee = p_glee * DAVIS_RANGE + DAVIS_MIN
        val_aspi = p_aspi * DAVIS_RANGE + DAVIS_MIN
        print(f"[Epoch {epoch+1}] Monitor:")
        print(f"   - Gleevec pKd: {val_glee:.2f}")
        print(f"   - Aspirin pKd: {val_aspi:.2f}")
        print(f"   - Separation Gap: {val_glee - val_aspi:.2f}")

X_d = np.array(drug_feats)
X_e = np.array(enzyme_feats)
Y_n = df['affinity_norm'].values

X_d_tr, X_d_ts, X_e_tr, X_e_ts, y_tr, y_ts = train_test_split(X_d, X_e, Y_n, test_size=0.15, random_state=42)

# Drug Neurons
d_in = Input(shape=(2048,))
dx = Dense(1024, activation='relu')(d_in)
dx = BatchNormalization()(dx)
dx = Dropout(0.3)(dx)
dx = Dense(512, activation='relu')(dx)
drug_emb = Dense(256, activation='relu')(dx)

# Enzyme Neurons
e_in = Input(shape=(1000,))
e_embed = Embedding(21, 128)(e_in)

def c_block_improved(inputs, filters, k):
    # Stacked CNN
    c = Conv1D(filters, k, activation='relu', padding='same')(inputs)
    c = Conv1D(filters * 2, k, activation='relu', padding='same')(c)
    c = BatchNormalization()(c)
    # Dual pooling 
    g_max = GlobalMaxPooling1D()(c)
    g_avg = GlobalAveragePooling1D()(c)
    return concatenate([g_max, g_avg])

e_branch = concatenate([
    c_block_improved(e_embed, 64, 4), 
    c_block_improved(e_embed, 64, 8), 
    c_block_improved(e_embed, 64, 12)
])
e_emb = Dense(256, activation='relu')(e_branch)

# Fusion vector
fused = Concatenate()([drug_emb, e_emb])
f = Dense(1024, activation='relu')(fused)
f = BatchNormalization()(f) 
f = Dropout(0.4)(f)
f = Dense(512, activation='relu')(f)
f = Dense(256, activation='relu')(f)

# Sigmoid Activation Function
out = Dense(1, activation='sigmoid')(f) 

model = Model(inputs=[d_in, e_in], outputs=out)


model.compile(optimizer=Adam(0.0005), loss='mse', metrics=['mae']) 

# Callbacks
monitor = DiscoveryMonitor(GLEEVEC, ASPIRIN, ABL1)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)

cb = [
    EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True, verbose=1),
    lr_schedule,
    ModelCheckpoint('metal_CADENCE.keras', monitor='val_mae', save_best_only=True),
    monitor
]

# Training
model.fit(x=[X_d_tr, X_e_tr], y=y_tr, validation_data=([X_d_ts, X_e_ts], y_ts), 
          epochs=50, batch_size=64, callbacks=cb) # Smaller batch size often helps with generalization


preds = model.predict([X_d_ts, X_e_ts]).flatten()
y_act_pkd = y_ts * DAVIS_RANGE + DAVIS_MIN
y_pre_pkd = preds * DAVIS_RANGE + DAVIS_MIN
print(f"\nFinal Real-World MAE: {mean_absolute_error(y_act_pkd, y_pre_pkd):.4f}")
print(f"Final Real-World MSE: {mean_squared_error(y_act_pkd, y_pre_pkd):.4f}")