import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_DIR = 'data'
MODEL_DIR = 'models'

# Ensure the 'models' folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_reen_model():
    print("👩‍🔬 Training Reen's AI Model...")
    
    # 1. Load Data
    try:
        X = np.load(os.path.join(DATA_DIR, 'X_features.npy'))
        y = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
        print(f"   -> Loaded {len(X)} compounds.")
    except FileNotFoundError:
        print("❌ Error: Processed data not found. Please run 'preprocess.py' first!")
        return

    # 2. Split Data
    # test_size=0.1 means 10% of data is used for testing, 90% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Train Model 
    # n_estimators=100 creates 100 decision trees
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate (Check how smart it is)
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"📊 Model Accuracy: {acc*100:.2f}%")
    else:
        print("📊 Training complete (Dataset too small for split validation, but model is ready)")

    # 5. Save Model
    model_path = os.path.join(MODEL_DIR, 'reen_biochem_model.pkl')
    joblib.dump(clf, model_path)
    print(f"✅ Model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_reen_model()