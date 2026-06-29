import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_pipeline", "processed_data", "aligned_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml_engine", "saved_models")

def load_and_prepare_data(filepath):
    print("📥 Loading preprocessed dataset...")
    df = pd.read_csv(filepath)
    
    # 1. CREATE THE TARGET VARIABLE (Ground Truth)
    # If hard X-rays spike above 0, an event occurred. 
    df['is_flare'] = (df['hard_xray_total'] > 0).astype(int)
    
    # --- HACKATHON PROTOTYPE SAFEGUARD ---
    # If the user downloaded a quiet week with 0 flares, the ML model will fail to train.
    # We inject a synthetic anomaly so the pipeline can complete successfully.
    if df['is_flare'].sum() == 0 and len(df) > 100:
        print("⚠️ ALERT: Your data sample is 100% 'Quiet Sun' (0 flares).")
        print("💉 Injecting a synthetic flare signature so the AI has something to learn from...")
        
        # Create a fake build-up in soft X-rays (The Precursors)
        df.loc[40:70, 'soft_xray_derivative'] = np.linspace(0.1, 5.0, 31)
        df.loc[40:70, 'soft_to_hard_ratio'] = np.linspace(10, 100, 31)
        
        # Create the fake hard X-ray explosion (The Event)
        # Using specific list of indices to ensure exact length match
        flare_indices = [70, 71, 72, 73, 74]
        df.loc[flare_indices, 'hard_xray_total'] = [50, 200, 500, 100, 20]
        
        # Re-calculate the labels now that we have a flare
        df['is_flare'] = (df['hard_xray_total'] > 0).astype(int)
    # --------------------------------------
    
    # 2. THE PREDICTIVE SHIFT (Time-Series Forecasting)
    # We want to predict a flare 30 minutes BEFORE it happens.
    # So, we shift the 'is_flare' label backwards by 30 rows (30 mins).
    df['target_predict_30m'] = df['is_flare'].shift(-30)
    
    # Drop the last 30 rows since they don't have a future to predict
    df = df.dropna()
    
    print(f"✅ Data prepped. Finding precursors for {int(df['target_predict_30m'].sum())} flare-minutes.")
    return df

def train_model(df):
    print("\n🧠 Training Random Forest AI Model...")
    
    # Define our Features (X) and Target (y)
    features = ['soft_xray_counts', 'soft_xray_derivative', 'soft_to_hard_ratio']
    
    X = df[features]
    y = df['target_predict_30m']
    
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nDetailed Report:")
    
    # Dynamically handle cases where the test set might only have 1 class due to small data
    unique_labels = np.unique(y_test)
    full_target_names = {0: "Quiet Sun", 1: "Flare Imminent"}
    actual_names = [full_target_names[label] for label in unique_labels]
    
    print(classification_report(y_test, predictions, labels=unique_labels, target_names=actual_names))
    
    return model

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"❌ Could not find {DATA_PATH}. Run preprocess.py first!")
    else:
        # Train
        dataset = load_and_prepare_data(DATA_PATH)
        trained_model = train_model(dataset)
        
        # Save the model
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        model_filename = os.path.join(MODEL_DIR, "flare_predictor_v1.pkl")
        joblib.dump(trained_model, model_filename)
        print(f"\n💾 Model successfully saved to: {model_filename}")
        print("You can now use this file in your dashboard to make live predictions!")