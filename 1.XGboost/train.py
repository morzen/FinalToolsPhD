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
    MODEL_PATH = "6feat_efficient_apt_similarity.pkl"

#variables share by some models:
    MODEL_PARAMS = {
        'objective': 'multi:softprob',#XGboost only
        'n_estimators': 200,          # INCREASED: More trees for complex patterns
        'max_depth': 10,              # INCREASED: Deeper trees
        'learning_rate': 0.05,        # REDUCED: Slower learning
        'subsample': 0.8,             # ADDED: Prevent overfitting
        'colsample_bytree': 0.8,      # ADDED: Feature sampling
        'random_state': 42 #shared between XGboost siames and RF
    }

#variables unique to this model:
    FEATURE_COLUMNS = ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']
    TARGET_COLUMN = 'APTname'
    
    TRAIN_TEST_SPLIT = 0.2

# ============================
# 3. DATA LOADING & PREPROCESSING (WITH DIMENSIONALITY REDUCTION)
# ============================
from sklearn.decomposition import PCA

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
    def prepare_multi_hot_encoding(df, n_components=256):
        """Create multi-hot encoding with PCA for dimensionality reduction"""
        print("\nCREATING MULTI-HOT ENCODING WITH DIMENSIONALITY REDUCTION")
        print("=" * 50)
        
        # Get all unique features across ALL APTs
        all_techniques = sorted(df['techniques_str'].unique())
        all_software = sorted(df['software_str'].unique())
        all_capec = sorted(df['CAPEC_ID'].unique())
        all_cwe = sorted(df['CWE_ID'].unique())
        all_cve = sorted(df['CVE_ID'].unique())
        all_pulse = sorted(df['Pulse_ID'].unique())
        
        # Create APT-to-features mapping
        apt_features = {}
        unique_apts = df['APTname'].unique()
        
        for apt in unique_apts:
            apt_data = df[df['APTname'] == apt]
            
            # Get all unique features for this APT
            apt_techniques = set(apt_data['techniques_str'].unique())
            apt_software = set(apt_data['software_str'].unique())
            apt_capec = set(apt_data['CAPEC_ID'].unique())
            apt_cwe = set(apt_data['CWE_ID'].unique())
            apt_cve = set(apt_data['CVE_ID'].unique())
            apt_pulse = set(apt_data['Pulse_ID'].unique())
            
            # Create binary vectors for each feature type
            techniques_vector = [1 if t in apt_techniques else 0 for t in all_techniques]
            software_vector = [1 if s in apt_software else 0 for s in all_software]
            capec_vector = [1 if c in apt_capec else 0 for c in all_capec]
            cwe_vector = [1 if w in apt_cwe else 0 for w in all_cwe]
            cve_vector = [1 if v in apt_cve else 0 for v in all_cve]
            pulse_vector = [1 if p in apt_pulse else 0 for p in all_pulse]
            
            # Combine all vectors
            combined_vector = (techniques_vector + software_vector + 
                             capec_vector + cwe_vector + cve_vector + pulse_vector)
            
            apt_features[apt] = {
                'binary_vector': combined_vector,
                'techniques': apt_techniques,
                'software': apt_software,
                'capec': apt_capec,
                'cwe': apt_cwe,
                'cve': apt_cve,
                'pulse': apt_pulse,
                'count': len(apt_data)
            }
        
        # Create training data (X) and labels (y)
        X = []
        y = []
        
        for apt, features in apt_features.items():
            X.append(features['binary_vector'])
            y.append(apt)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"Original feature vector length: {X.shape[1]}")
        print(f"Number of APTs: {X.shape[0]}")
        
        # Apply PCA for dimensionality reduction
        print(f"\nApplying PCA dimensionality reduction...")
        pca = PCA(n_components=min(n_components, X.shape[0]))
        X_reduced = pca.fit_transform(X)
        
        print(f"Reduced feature vector length: {X_reduced.shape[1]}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Save feature metadata for later use
        feature_metadata = {
            'all_techniques': all_techniques,
            'all_software': all_software,
            'all_capec': all_capec,
            'all_cwe': all_cwe,
            'all_cve': all_cve,
            'all_pulse': all_pulse,
            'techniques_offset': 0,
            'software_offset': len(all_techniques),
            'capec_offset': len(all_techniques) + len(all_software),
            'cwe_offset': len(all_techniques) + len(all_software) + len(all_capec),
            'cve_offset': len(all_techniques) + len(all_software) + len(all_capec) + len(all_cwe),
            'pulse_offset': len(all_techniques) + len(all_software) + len(all_capec) + len(all_cwe) + len(all_cve),
            'pca': pca  # Save PCA for transforming query vectors
        }
        
        print(f"\nTotal features breakdown:")
        print(f"  Techniques: {len(all_techniques)}")
        print(f"  Software: {len(all_software)}")
        print(f"  CAPEC: {len(all_capec)}")
        print(f"  CWE: {len(all_cwe)}")
        print(f"  CVE: {len(all_cve)}")
        print(f"  Pulse: {len(all_pulse)}")
        
        return X_reduced, y, apt_features, feature_metadata

# ============================
# 4. MODEL DEFINITION (UPDATED)
# ============================
class EfficientXGBoostModel:
    """Efficient XGBoost model for APT similarity detection with multi-hot encoding"""
    
    def __init__(self):
        self.model = None
        self.apt_features = {}
        self.apt_encoder = {}
        self.reverse_encoder = {}
        self.feature_metadata = {}
        self.all_apt_vectors = {}  # Store binary vectors for all APTs
    
    def train(self, X, y, apt_features, feature_metadata):
        """Train the XGBoost classifier on multi-hot encoded data"""
        print(f"\nTRAINING XGBOOST MODEL WITH MULTI-HOT ENCODING")
        print("=" * 50)
        
        # Store metadata
        self.feature_metadata = feature_metadata
        self.apt_features = apt_features
        
        # Create APT encoding
        unique_apts = np.unique(y)
        self.apt_encoder = {apt: idx for idx, apt in enumerate(unique_apts)}
        self.reverse_encoder = {v: k for k, v in self.apt_encoder.items()}
        
        # Encode target variable
        y_encoded = np.array([self.apt_encoder[apt] for apt in y])
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**Config.MODEL_PARAMS)
        print(f"Training classifier on {X.shape[0]} samples...")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(unique_apts)}")
        
        # Train on all data
        self.model.fit(X, y_encoded)
        
        # Store binary vectors for all APTs
        for apt in unique_apts:
            apt_idx = self.apt_encoder[apt]
            self.all_apt_vectors[apt] = X[apt_idx]
        
        print("Training completed!")
        
        # Evaluate model
        self._evaluate_model(X, y_encoded)
    
    def create_query_vector(self, techniques, software, capec, cwe, cve, pulse):
        """Convert query evidence to binary vector and apply PCA transformation"""
        # Create full binary vector
        vector = np.zeros(len(self.feature_metadata['all_techniques']) +
                        len(self.feature_metadata['all_software']) +
                        len(self.feature_metadata['all_capec']) +
                        len(self.feature_metadata['all_cwe']) +
                        len(self.feature_metadata['all_cve']) +
                        len(self.feature_metadata['all_pulse']), dtype=np.float32)
        
        # Mark techniques
        for t in techniques:
            if t in self.feature_metadata['all_techniques']:
                idx = self.feature_metadata['all_techniques'].index(t)
                vector[idx] = 1.0
        
        for s in software:
            if s in self.feature_metadata['all_software']:
                idx = self.feature_metadata['all_software'].index(s) + self.feature_metadata['software_offset']
                vector[idx] = 1.0
        
        for c in capec:
            if c in self.feature_metadata['all_capec']:
                idx = self.feature_metadata['all_capec'].index(c) + self.feature_metadata['capec_offset']
                vector[idx] = 1.0
        
        for w in cwe:
            if w in self.feature_metadata['all_cwe']:
                idx = self.feature_metadata['all_cwe'].index(w) + self.feature_metadata['cwe_offset']
                vector[idx] = 1.0
        
        for v in cve:
            if v in self.feature_metadata['all_cve']:
                idx = self.feature_metadata['all_cve'].index(v) + self.feature_metadata['cve_offset']
                vector[idx] = 1.0
        
        for p in pulse:
            if p in self.feature_metadata['all_pulse']:
                idx = self.feature_metadata['all_pulse'].index(p) + self.feature_metadata['pulse_offset']
                vector[idx] = 1.0
        
        # Apply PCA transformation to match training data
        vector_pca = self.feature_metadata['pca'].transform([vector])[0]
        return vector_pca
    
    def predict_similarity(self, query_vector, apt_vector):
        """Calculate similarity between query vector and APT vector"""
        # Use cosine similarity
        dot_product = np.dot(query_vector, apt_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_apt = np.linalg.norm(apt_vector)
        
        if norm_query == 0 or norm_apt == 0:
            return 0.0
        
        similarity = dot_product / (norm_query * norm_apt)
        return np.clip(similarity, 0.0, 1.0)
    
    def find_similar_apts(self, query_vector, top_k=10):
        """Find most similar APTs to given query vector"""
        similarities = []
        
        for apt, apt_vector in self.all_apt_vectors.items():
            similarity = self.predict_similarity(query_vector, apt_vector)
            similarities.append((apt, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _evaluate_model(self, X, y_encoded):
        """Evaluate model quality"""
        print("\nMODEL QUALITY CHECK:")
        print("-" * 30)
        
        # Get predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_encoded)
        print(f"Training accuracy: {accuracy:.4f}")
        
        # Check prediction diversity
        avg_prob_std = np.mean(np.std(y_pred_proba, axis=1))
        print(f"Average prediction std: {avg_prob_std:.4f}")

# ============================
# 5. TRAINING FUNCTION (UPDATED)
# ============================
def train_xgboost_model():
    """Main training pipeline for XGBoost with multi-hot encoding"""
    print("XGBOOST TRAINING PIPELINE (MULTI-HOT ENCODING)")
    print("=" * 60)
    
    # Load data
    data_loader = XGBoostDataLoader()
    df = data_loader.load_data()
    
    # Create multi-hot encoding with PCA
    X, y, apt_features, feature_metadata = data_loader.prepare_multi_hot_encoding(df, n_components=100)
    
    print(f"\nTRAINING ON ALL APTs")
    print(f"Total APTs: {X.shape[0]}")
    print(f"Feature vector length: {X.shape[1]}")
    
    # Train model
    model = EfficientXGBoostModel()
    model.train(X, y, apt_features, feature_metadata)
    
    return model, X, y, apt_features, feature_metadata

# ============================
# 6. MODEL SAVING (UPDATED)
# ============================
def save_xgboost_model(model, X, y, apt_features, feature_metadata, filename=Config.MODEL_PATH):
    """Save trained XGBoost model with multi-hot encoding metadata"""
    save_data = {
        'model': model.model,
        'apt_features': apt_features,
        'apt_encoder': model.apt_encoder,
        'reverse_encoder': model.reverse_encoder,
        'feature_metadata': feature_metadata,
        'all_apt_vectors': model.all_apt_vectors,
        'training_info': {
            'approach': 'efficient_xgboost_multi_hot_encoding',
            'num_apts': len(y),
            'feature_vector_length': X.shape[1],
            'unique_apts': len(np.unique(y)),
            'apt_list': sorted(np.unique(y)),
            'model_params': Config.MODEL_PARAMS,
            'feature_counts': {
                'techniques': len(feature_metadata['all_techniques']),
                'software': len(feature_metadata['all_software']),
                'capec': len(feature_metadata['all_capec']),
                'cwe': len(feature_metadata['all_cwe']),
                'cve': len(feature_metadata['all_cve']),
                'pulse': len(feature_metadata['all_pulse'])
            }
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\nModel saved to '{filename}'")
    print(f"  • Feature vector length: {X.shape[1]}")
    print(f"  • Number of APTs: {len(y)}")
    
    return save_data
# ============================
# 7. MAIN EXECUTION (UPDATED)
# ============================
def main():
    """Execute the XGBoost training pipeline"""
    # Train model (now returns 5 values)
    model, X_train, y_train, apt_features, feature_metadata = train_xgboost_model()
    
    # Test predictions
    print(f"\nTESTING MODEL PREDICTIONS")
    print("=" * 50)
    
    # Test on some training samples
    if len(X_train) > 0:
        test_sample = X_train[0]
        similar_apts = model.find_similar_apts(test_sample, top_k=3)
        
        apt_name = y_train[0] if len(y_train) > 0 else "Unknown"
        print(f"Sample APT: {apt_name}")
        print(f"Most similar APTs:")
        for apt_id, similarity in similar_apts:
            print(f"  {apt_id}: {similarity:.4f}")
        
        # Test similarity between two samples
        if len(X_train) > 1:
            test_features2 = X_train[1]
            similarity = model.predict_similarity(test_sample, test_features2)
            apt_name2 = y_train[1] if len(y_train) > 1 else "Unknown"
            print(f"\nSimilarity between {apt_name} and {apt_name2}: {similarity:.4f}")
    
    # Save model (now requires all 5 parameters)
    save_data = save_xgboost_model(model, X_train, y_train, apt_features, feature_metadata)
    
    print(f"\nTRAINING SUMMARY")
    print("=" * 50)
    info = save_data['training_info']
    print(f"• Approach: {info['approach']}")
    print(f"• Number of APTs: {info['num_apts']}")
    print(f"• Feature vector length: {info['feature_vector_length']}")
    print(f"• Unique APTs: {info['unique_apts']}")
    print(f"• Feature counts:")
    print(f"  - Techniques: {info['feature_counts']['techniques']}")
    print(f"  - Software: {info['feature_counts']['software']}")
    print(f"  - CAPEC: {info['feature_counts']['capec']}")
    print(f"  - CWE: {info['feature_counts']['cwe']}")
    print(f"  - CVE: {info['feature_counts']['cve']}")
    print(f"  - Pulse: {info['feature_counts']['pulse']}")
    print(f"• Model parameters: {info['model_params']}")
    
    return save_data

if __name__ == "__main__":
    saved_data = main()

