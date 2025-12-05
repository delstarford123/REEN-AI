import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# --- CONFIGURATION ---
# This must match where train.py saved the file
MODEL_PATH = os.path.join('models', 'reen_biochem_model.pkl')

def get_fingerprint(smiles):
    """
    Helper to convert input SMILES to the exact format the model expects.
    Must match the logic in preprocess.py!
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Morgan Fingerprint (Radius 2, 2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return np.array(fp).reshape(1, -1)
    except:
        pass
    return None

def predict_molecule(smiles):
    """
    Main function called by the website.
    Returns: Dictionary with Classification and Confidence score.
    """
    # 1. Check if the "Brain" exists
    if not os.path.exists(MODEL_PATH):
        return {
            "error": "AI Model not found. Please run 'python train.py' first!"
        }

    try:
        # 2. Load the Brain
        model = joblib.load(MODEL_PATH)
        
        # 3. Convert Chemistry to Math
        features = get_fingerprint(smiles)

        if features is None:
            return {
                "error": "Invalid Chemical Structure. Please check the SMILES string."
            }

        # 4. Ask the Brain for a prediction
        prediction = model.predict(features)[0]
        
        # Get confidence (probability)
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)

        return {
            "classification": prediction,
            "confidence": round(confidence * 100, 2),
            "smiles": smiles
        }

    except Exception as e:
        return {"error": f"Prediction Error: {str(e)}"}