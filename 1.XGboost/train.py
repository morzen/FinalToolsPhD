import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_training_data(pwd="../BaseData/"):
    """Load the encoded data directly"""
    print("Loading encoded data...")
    df = pd.read_csv(pwd + 'EncodedData.csv')
    print(f"Data: {len(df)} records, {df.shape[1]} columns")
    return df

def prepare_training_data(df):
    """Prepare data for efficient training - USE ALL 181 APTs"""
    print("\n🎯 PREPARING EFFICIENT TRAINING DATA (ALL APTs)")
    
    # Count records per APT (for info only)
    apt_counts = df['APTname'].value_counts()
    print(f"APT distribution:")
    print(f"  APTs with 1 record: {len(apt_counts[apt_counts == 1])}")
    print(f"  APTs with 2-10 records: {len(apt_counts[(apt_counts >= 2) & (apt_counts <= 10)])}")
    print(f"  APTs with 10+ records: {len(apt_counts[apt_counts > 10])}")
    
    # FIXED: Use ALL APTs - no filtering
    df_filtered = df.copy()
    print(f"Using ALL APTs: {df['APTname'].nunique()} APTs")
    print(f"Total records: {len(df_filtered)}")
    
    # Use ALL 7 features
    X = df_filtered[['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']].values
    y = df_filtered['APTname'].values
    
    print(f"Features: {X.shape}")
    print(f"Unique APTs: {len(np.unique(y))}")
    
    return X, y, df_filtered

class EfficientXGBoost:
    """Efficient XGBoost that learns APT patterns directly"""
    
    def __init__(self):
        self.model = None
        self.apt_features = {}
        self.apt_encoder = {}
    
    def train(self, X_train, y_train):
        """Train efficient XGBoost classifier"""
        print(f"\n🚀 TRAINING EFFICIENT XGBOOST")
        print(f"Training on {X_train.shape[0]} records")
        print(f"Features: {X_train.shape[1]}, APTs: {len(np.unique(y_train))}")
        
        # Create mapping from APT IDs to consecutive indices for XGBoost
        unique_apts = np.unique(y_train)
        self.apt_encoder = {apt: idx for idx, apt in enumerate(unique_apts)}
        
        # Convert y_train to consecutive indices
        y_train_encoded = np.array([self.apt_encoder[apt] for apt in y_train])
        
        # Train classifier to predict APT from features
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        print("Training classifier...")
        self.model.fit(X_train, y_train_encoded)
        print("✅ Training completed!")
        
        # Store feature patterns for each APT
        self._learn_apt_patterns(X_train, y_train)
    
    def _learn_apt_patterns(self, X, y):
        """Learn characteristic patterns for each APT"""
        print("Learning APT patterns...")
        for apt in np.unique(y):
            apt_mask = (y == apt)
            if np.sum(apt_mask) > 0:
                apt_data = X[apt_mask]
                # Store common features for this APT - ALL 7 features
                self.apt_features[apt] = {
                    'common_techniques': np.unique(apt_data[:, 0]),
                    'common_software': np.unique(apt_data[:, 1]),
                    'common_capec': np.unique(apt_data[:, 2]),
                    'common_cwe': np.unique(apt_data[:, 3]),
                    'common_cves': np.unique(apt_data[:, 4]),
                    'common_pulse': np.unique(apt_data[:, 5]),
                    'count': len(apt_data)
                }
    
    def predict_similarity(self, features1, features2):
        """Calculate similarity between two feature sets - PURE MODEL OUTPUT"""
        # Get probability distributions
        if np.all(features1 == -1):
            return 0.0
        if np.all(features2 == -1):
            return 0.0
            
        prob1 = self.model.predict_proba([features1])[0]
        prob2 = self.model.predict_proba([features2])[0]
        
        # Pure cosine similarity of XGBoost probability distributions
        cosine_sim = np.dot(prob1, prob2) / (np.linalg.norm(prob1) * np.linalg.norm(prob2))
        
        return np.clip(cosine_sim, 0.0, 1.0)
    
    def find_similar_apts(self, features, top_k=5):
        """Find most similar APTs to given features"""
        if np.all(features == -1):
            return []
            
        probabilities = self.model.predict_proba([features])[0]
        similar_apts = []
        
        # Convert back to original APT IDs
        reverse_encoder = {v: k for k, v in self.apt_encoder.items()}
        
        for apt_idx, prob in enumerate(probabilities):
            if prob > 0.01:  # Only consider APTs with meaningful probability
                original_apt = reverse_encoder[apt_idx]
                similar_apts.append((original_apt, prob))
        
        # Sort by probability
        similar_apts.sort(key=lambda x: x[1], reverse=True)
        return similar_apts[:top_k]

def main():
    """Efficient XGBoost training - ALL 181 APTs, no filtering"""
    print("🚀 EFFICIENT XGBOOST TRAINING (ALL 181 APTs)")
    print("=" * 50)
    print("🎯 SINGLE RECORDS → PATTERN LEARNING → SIMILARITY")
    print("✅ FIXED: Uses ALL APTs - no rare APT filtering")
    print("=" * 50)
    
    # Step 1: Load data
    df = load_training_data()
    
    # Step 2: Prepare training data (ALL APTs)
    X, y, df_filtered = prepare_training_data(df)
    
    # Step 3: Split for validation - EXACT SAME AS ORIGINAL
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training: {X_tr.shape[0]} records ({len(np.unique(y_tr))} APTs)")
    print(f"Validation: {X_val.shape[0]} records ({len(np.unique(y_val))} APTs)")
    
    # Step 4: Train efficient model
    model = EfficientXGBoost()
    model.train(X_tr, y_tr)
    
    # Step 5: Test similarity (pure model output)
    print(f"\n🧪 TESTING SIMILARITY (PURE MODEL)")
    
    test_sample = X_val[0]
    similar_apts = model.find_similar_apts(test_sample, top_k=3)
    
    print(f"Sample features: T={test_sample[0]}, SW={test_sample[1]}, CAPEC={test_sample[2]}, CWE={test_sample[3]}, CVE={test_sample[4]}, Pulse={test_sample[5]}")
    print(f"Most similar APTs:")
    for apt_id, similarity in similar_apts:
        print(f"  APT {apt_id}: {similarity:.4f}")
    
    if len(X_val) > 1:
        test_features2 = X_val[1]
        similarity = model.predict_similarity(test_sample, test_features2)
        print(f"Pure model cosine similarity: {similarity:.4f}")
    
    # Step 6: Save model with ALL APTs info
    save_data = {
        'model': model.model,
        'apt_features': model.apt_features,
        'apt_encoder': model.apt_encoder,
        'feature_columns': ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID'],
        'training_info': {
            'approach': 'efficient_xgboost_classifier_all_apts',
            'training_records': len(X_tr),
            'unique_apts': len(np.unique(y_tr)),
            'total_apts': len(np.unique(y)),  
            'apt_list': sorted(np.unique(y)),  
            'original_data_size': len(df)
        }
    }
    
    with open('efficient_apt_similarity.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n✅ Model saved as 'efficient_apt_similarity.pkl'")
    print(f"🎯 Efficient approach (ALL APTs):")
    print(f"   • No pair creation")
    print(f"   • ALL 181 APTs included")
    print(f"   • Direct classification training") 
    print(f"   • Pure XGBoost cosine similarity")
    print(f"   • Scalable to large datasets")
    
    # Show statistics
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Original records: {len(df)}")
    print(f"   Training records: {len(X_tr)}")
    print(f"   APTs in training: {len(np.unique(y_tr))}")
    print(f"   Total unique APTs: {len(np.unique(y))}")
    print(f"   Features per record: {X.shape[1]}")
    
    return save_data

if __name__ == "__main__":
    saved_data = main()
