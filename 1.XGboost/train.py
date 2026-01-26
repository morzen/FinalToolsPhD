"""
STANDARDIZED XGBOOST MODEL FOR APT SIMILARITY
=============================================
Author: [Your Name]
Date: [Current Date]
Description: Efficient XGBoost classifier for APT similarity detection
"""

# ============================
# 1. IMPORTS
# ============================
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================
# 2. CONFIGURATION
# ============================
class Config:
    """Configuration for XGBoost model"""

#variable shared by all models:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "efficient_apt_similarity.pkl"

#variables share by some models:
    MODEL_PARAMS = {
        'objective': 'multi:softprob',#XGboost only
        'n_estimators': 100,#XGboost only
        'max_depth': 6,#XGboost only
        'learning_rate': 0.1,#shared between XGboost GNN and siamese
        'random_state': 42 #shared between XGboost siames and RF
    }

#variables unique to this model:
    FEATURE_COLUMNS = ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']
    TARGET_COLUMN = 'APTname'
    
    TRAIN_TEST_SPLIT = 0.2

# ============================
# 3. DATA LOADING & PREPROCESSING
# ============================
class XGBoostDataLoader:
    """Handles data loading and preparation for XGBoost"""
    
    @staticmethod
    def load_data(data_path=Config.DATA_PATH):
        """Load encoded training data"""
        print("Loading encoded data...")
        df = pd.read_csv(data_path)
        print(f"Data: {len(df)} records, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def prepare_training_data(df):
        """Prepare data for XGBoost training"""
        print("\nPREPARING TRAINING DATA")
        print("=" * 50)
        
        # Count records per APT
        apt_counts = df['APTname'].value_counts()
        print(f"APT distribution:")
        print(f"  APTs with 1 record: {len(apt_counts[apt_counts == 1])}")
        print(f"  APTs with 2-10 records: {len(apt_counts[(apt_counts >= 2) & (apt_counts <= 10)])}")
        print(f"  APTs with 10+ records: {len(apt_counts[apt_counts > 10])}")
        
        # Use all data
        X = df[Config.FEATURE_COLUMNS].values
        y = df[Config.TARGET_COLUMN].values
        
        print(f"Features shape: {X.shape}")
        print(f"Unique APTs: {len(np.unique(y))}")
        
        return X, y, df

# ============================
# 4. MODEL DEFINITION
# ============================
class EfficientXGBoostModel:
    """Efficient XGBoost model for APT similarity detection"""
    
    def __init__(self):
        self.model = None
        self.apt_features = {}
        self.apt_encoder = {}
        self.reverse_encoder = {}
    
    def train(self, X_train, y_train):
        """Train the XGBoost classifier"""
        print(f"\nTRAINING XGBOOST MODEL")
        print("=" * 50)
        
        # Create APT encoding
        unique_apts = np.unique(y_train)
        self.apt_encoder = {apt: idx for idx, apt in enumerate(unique_apts)}
        self.reverse_encoder = {v: k for k, v in self.apt_encoder.items()}
        
        # Encode target variable
        y_train_encoded = np.array([self.apt_encoder[apt] for apt in y_train])
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**Config.MODEL_PARAMS)
        print("Training classifier...")
        self.model.fit(X_train, y_train_encoded)
        
        # Learn APT patterns
        self._learn_apt_patterns(X_train, y_train)
        print("Training completed!")
    
    def _learn_apt_patterns(self, X, y):
        """Learn characteristic patterns for each APT"""
        for apt in np.unique(y):
            apt_mask = (y == apt)
            if np.sum(apt_mask) > 0:
                apt_data = X[apt_mask]
                self.apt_features[apt] = {
                    'common_techniques': np.unique(apt_data[:, 0]),
                    'common_software': np.unique(apt_data[:, 1]),
                    'common_capec': np.unique(apt_data[:, 2]),
                    'common_cwe': np.unique(apt_data[:, 3]),
                    'common_cve': np.unique(apt_data[:, 4]),
                    'common_pulse': np.unique(apt_data[:, 5]),
                    'count': len(apt_data)
                }
    
    def predict_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        # Handle missing features
        if self._has_missing_features(features1) or self._has_missing_features(features2):
            return 0.0
        
        # Get probability distributions
        prob1 = self.model.predict_proba([features1])[0]
        prob2 = self.model.predict_proba([features2])[0]
        
        # Calculate cosine similarity
        cosine_sim = np.dot(prob1, prob2) / (np.linalg.norm(prob1) * np.linalg.norm(prob2))
        return np.clip(cosine_sim, 0.0, 1.0)
    
    def _has_missing_features(self, features):
        """Check if features are missing"""
        return all(f == -1 for f in features[:3])
    
    def find_similar_apts(self, features, top_k=5):
        """Find most similar APTs to given features"""
        if self._has_missing_features(features):
            return []
        
        probabilities = self.model.predict_proba([features])[0]
        similar_apts = []
        
        for apt_idx, prob in enumerate(probabilities):
            if prob > 0.01:  # Meaningful probability threshold
                original_apt = self.reverse_encoder[apt_idx]
                similar_apts.append((original_apt, prob))
        
        # Sort by probability
        similar_apts.sort(key=lambda x: x[1], reverse=True)
        return similar_apts[:top_k]

# ============================
# 5. TRAINING FUNCTION
# ============================
def train_xgboost_model():
    """Main training pipeline for XGBoost"""
    print("XGBOOST TRAINING PIPELINE")
    print("=" * 60)
    
    # Load and prepare data
    data_loader = XGBoostDataLoader()
    df = data_loader.load_data()
    X, y, df_filtered = data_loader.prepare_training_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TRAIN_TEST_SPLIT, random_state=42
    )
    print(f"\nData split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Train model
    model = EfficientXGBoostModel()
    model.train(X_train, y_train)
    
    return model, X_train, y_train, X_val, y_val

# ============================
# 6. MODEL SAVING
# ============================
def save_xgboost_model(model, X_train, y_train, filename=Config.MODEL_PATH):
    """Save trained XGBoost model with metadata"""
    save_data = {
        'model': model.model,
        'apt_features': model.apt_features,
        'apt_encoder': model.apt_encoder,
        'feature_columns': Config.FEATURE_COLUMNS,
        'training_info': {
            'approach': 'efficient_xgboost_classifier_all_apts',
            'training_records': len(X_train),
            'unique_apts': len(np.unique(y_train)),
            'apt_list': sorted(np.unique(y_train)),
            'model_params': Config.MODEL_PARAMS
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\nModel saved to '{filename}'")
    return save_data

# ============================
# 7. MAIN EXECUTION
# ============================
def main():
    """Execute the XGBoost training pipeline"""
    # Train model
    model, X_train, y_train, X_val, y_val = train_xgboost_model()
    
    # Test predictions
    print(f"\nTESTING MODEL PREDICTIONS")
    print("=" * 50)
    
    if len(X_val) > 0:
        test_sample = X_val[0]
        similar_apts = model.find_similar_apts(test_sample, top_k=3)
        
        print(f"Sample features: {test_sample}")
        print(f"Most similar APTs:")
        for apt_id, similarity in similar_apts:
            print(f"  APT {apt_id}: {similarity:.4f}")
        
        # Test similarity between two samples
        if len(X_val) > 1:
            test_features2 = X_val[1]
            similarity = model.predict_similarity(test_sample, test_features2)
            print(f"\nSimilarity between two samples: {similarity:.4f}")
    
    # Save model
    save_data = save_xgboost_model(model, X_train, y_train)
    
    print(f"\nTRAINING SUMMARY")
    print("=" * 50)
    info = save_data['training_info']
    print(f"• Approach: {info['approach']}")
    print(f"• Training records: {info['training_records']}")
    print(f"• Unique APTs: {info['unique_apts']}")
    print(f"• Model parameters: {info['model_params']}")
    
    return save_data

if __name__ == "__main__":
    saved_data = main()