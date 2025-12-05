import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dotenv import load_dotenv # Import the tool to read .env files

# --- 1. CONFIGURATION ---
load_dotenv() # Load variables from .env (e.g., CLOUDINARY_URL, API_KEYS)

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. SYNTHETIC BIOCHEMISTRY DATASET ---
# In a real scenario, Reen would download a CSV from ChEMBL or PubChem.
# Here, we create a small dataset for demonstration.
data = {
    'smiles': [
        'CC(=O)OC1=CC=CC=C1C(=O)O', # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
        'C[C@]12C[C@@H](C)C3=C(C=CC(=O)C=C3)[C@@H]1CC[C@]2(O)C(=O)CO', # Dexamethasone
        'C1CCCCC1', # Cyclohexane
        'CCO', # Ethanol
        'C(C1C(C(C(C(O1)O)O)O)O)O', # Glucose
        'NCC(=O)O', # Glycine
        'C1=CC=C(C=C1)C(C2=CC=CC=C2)O', # Benzhydrol
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', # Ibuprofen
        'CN(C)C(=N)NC(=N)N' # Metformin
    ],
    'category': [
        'Inhibitor', 'Stimulant', 'Steroid', 'Solvent', 'Solvent', 
        'Carbohydrate', 'Amino Acid', 'Alcohol', 'Inhibitor', 'Hypoglycemic'
    ]
}

# --- 3. FEATURIZATION FUNCTION ---
def featurize_smiles(smiles):
    """Converts a chemical string into a 2048-bit numerical fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # Morgan Fingerprint (Radius 2 is standard for biochem)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    except:
        return None

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print("🧪 Processing Reen's Biochemistry Data...")
    
    # Optional: Verify keys are loaded (just to check)
    if os.getenv("GEMINI_API_KEY"):
        print("   -> Environment variables detected.")
    
    df = pd.DataFrame(data)
    
    # Convert SMILES to Features (Robust way handling potential errors)
    X_raw = [featurize_smiles(s) for s in df['smiles']]
    
    # Filter out any failures (None values) to prevent crashes
    valid_indices = [i for i, x in enumerate(X_raw) if x is not None]
    
    X = np.array([X_raw[i] for i in valid_indices])
    y = np.array([df['category'][i] for i in valid_indices])
    
    # Save processed data
    np.save(os.path.join(DATA_DIR, 'X_features.npy'), X)
    np.save(os.path.join(DATA_DIR, 'y_labels.npy'), y)
    
    print(f"✅ Data processed: {len(X)} compounds ready for training.")