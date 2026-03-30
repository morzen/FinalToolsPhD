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
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "3feat_efficient_apt_similarity_nopca.pkl"
    MODEL_PARAMS = {
        'objective': 'multi:softprob',
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
    }
    FEATURE_COLUMNS = ['techniques_str', 'software_str','CVE_ID']
    TARGET_COLUMN = 'APTname'
    TRAIN_TEST_SPLIT = 0.2

# ============================
# 3. DATA LOADING & PREPROCESSING (NO PCA)
# ============================
class XGBoostDataLoader:
    @staticmethod
    def load_data(data_path=Config.DATA_PATH):
        print("Loading encoded data...")
        df = pd.read_csv(data_path)
        print(f"Data: {len(df)} records, {df.shape[1]} columns")
        return df

    @staticmethod
    def prepare_multi_hot_encoding(df):
        print("\nCREATING MULTI-HOT ENCODING (NO PCA)")
        print("=" * 50)

        # Get all unique features (sorted for reproducibility)
        all_techniques = sorted(df['techniques_str'].unique())
        all_software = sorted(df['software_str'].unique())
        all_cve = sorted(df['CVE_ID'].unique())

        # Create feature-to-index mappings
        technique_to_idx = {t: i for i, t in enumerate(all_techniques)}
        software_to_idx = {s: i for i, s in enumerate(all_software)}
        cve_to_idx = {v: i for i, v in enumerate(all_cve)}

        # Build a dictionary keyed by APT (using the original unique values, but later we will sort)
        apt_features = {}
        unique_apts_unsorted = df['APTname'].unique()
        for apt in unique_apts_unsorted:
            apt_data = df[df['APTname'] == apt]
            apt_techniques = set(apt_data['techniques_str'].unique())
            apt_software = set(apt_data['software_str'].unique())
            apt_cve = set(apt_data['CVE_ID'].unique())

            techniques_vector = [1 if t in apt_techniques else 0 for t in all_techniques]
            software_vector = [1 if s in apt_software else 0 for s in all_software]
            cve_vector = [1 if v in apt_cve else 0 for v in all_cve]

            combined_vector = (techniques_vector + software_vector + cve_vector )

            apt_features[apt] = {
                'binary_vector': combined_vector,
                'techniques': apt_techniques,
                'software': apt_software,
                'cve': apt_cve,
                'count': len(apt_data)
            }

        # --- FIX: Use a consistent sorted order for all APTs ---
        unique_apts = sorted(apt_features.keys())   # sorted list of APT IDs
        X = []
        y = []
        for apt in unique_apts:
            X.append(apt_features[apt]['binary_vector'])
            y.append(apt)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        print(f"Feature vector length: {X.shape[1]}")
        print(f"Number of APTs: {X.shape[0]}")

        # Offsets (useful for reconstruction)
        techniques_offset = 0
        software_offset = len(all_techniques)
        cve_offset = software_offset + len(all_software)

        feature_metadata = {
            'all_techniques': all_techniques,
            'all_software': all_software,
            'all_cve': all_cve,
            'techniques_offset': techniques_offset,
            'software_offset': software_offset,
            'cve_offset': cve_offset,
            'feature_mapping': {
                'techniques': technique_to_idx,
                'software': software_to_idx,
                'cve': cve_to_idx
            }
        }

        print(f"\nTotal features breakdown:")
        print(f"  Techniques: {len(all_techniques)}")
        print(f"  Software: {len(all_software)}")
        print(f"  CVE: {len(all_cve)}")

        return X, y, apt_features, feature_metadata

# ============================
# 4. MODEL DEFINITION
# ============================
class EfficientXGBoostModel:
    def __init__(self):
        self.model = None
        self.apt_features = {}
        self.apt_encoder = {}
        self.reverse_encoder = {}
        self.feature_metadata = {}
        self.all_binary_vectors = {}   # Store binary vectors for all APTs

    def train(self, X, y, apt_features, feature_metadata):
        print(f"\nTRAINING XGBOOST CLASSIFIER ON BINARY VECTORS")
        print("=" * 50)

        self.feature_metadata = feature_metadata
        self.apt_features = apt_features

        unique_apts = np.unique(y)          # already sorted because y is sorted
        self.apt_encoder = {apt: idx for idx, apt in enumerate(unique_apts)}
        self.reverse_encoder = {v: k for k, v in self.apt_encoder.items()}
        y_encoded = np.array([self.apt_encoder[apt] for apt in y])

        # Train XGBoost on the high-dimensional binary vectors
        self.model = xgb.XGBClassifier(**Config.MODEL_PARAMS)
        print(f"Training classifier on {X.shape[0]} samples, {X.shape[1]} features...")
        self.model.fit(X, y_encoded)

        # Store binary vectors for all APTs – using the same index order
        for apt in unique_apts:
            apt_idx = self.apt_encoder[apt]
            self.all_binary_vectors[apt] = X[apt_idx]

        print("Training completed!")
        self._evaluate_model(X, y_encoded)

    def predict_similarity(self, vec1, vec2):
        """Cosine similarity on binary vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.clip(dot / (norm1 * norm2), 0.0, 1.0)

    def find_similar_apts(self, query_vector, top_k=10):
        similarities = []
        for apt, vec in self.all_binary_vectors.items():
            sim = self.predict_similarity(query_vector, vec)
            similarities.append((apt, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def create_query_vector(self, techniques, software, cve):
        """Build binary vector from evidence using feature mappings."""
        feature_mapping = self.feature_metadata.get('feature_mapping', {})
        total_len = self.feature_metadata['cve_offset'] + len(self.feature_metadata['all_cve'])
        vector = np.zeros(total_len, dtype=np.float64)

        if 'techniques' in feature_mapping:
            tech_map = feature_mapping['techniques']
            for t in techniques:
                if t in tech_map:
                    vector[tech_map[t]] = 1.0

        if 'software' in feature_mapping:
            soft_map = feature_mapping['software']
            for s in software:
                if s in soft_map:
                    idx = soft_map[s] + self.feature_metadata['software_offset']
                    vector[idx] = 1.0

        if 'cve' in feature_mapping:
            cve_map = feature_mapping['cve']
            for v in cve:
                if v in cve_map:
                    idx = cve_map[v] + self.feature_metadata['cve_offset']
                    vector[idx] = 1.0

        return vector

    def _evaluate_model(self, X, y_encoded):
        y_pred = self.model.predict(X)
        acc = np.mean(y_pred == y_encoded)
        print(f"Training accuracy: {acc:.4f}")

# ============================
# 5. TRAINING FUNCTION
# ============================
def train_xgboost_model():
    print("XGBOOST TRAINING PIPELINE (NO PCA)")
    print("=" * 60)
    data_loader = XGBoostDataLoader()
    df = data_loader.load_data()
    X, y, apt_features, feature_metadata = data_loader.prepare_multi_hot_encoding(df)

    print(f"\nTRAINING ON ALL APTs")
    print(f"Total APTs: {X.shape[0]}")
    print(f"Feature vector length: {X.shape[1]}")

    model = EfficientXGBoostModel()
    model.train(X, y, apt_features, feature_metadata)
    return model, X, y, apt_features, feature_metadata

# ============================
# 6. MODEL SAVING
# ============================
def save_xgboost_model(model, X, y, apt_features, feature_metadata, filename=Config.MODEL_PATH):
    save_data = {
        'model': model.model,
        'apt_features': apt_features,
        'apt_encoder': model.apt_encoder,
        'reverse_encoder': model.reverse_encoder,
        'feature_metadata': feature_metadata,
        'all_binary_vectors': model.all_binary_vectors,
        'training_info': {
            'approach': 'xgboost_no_pca',
            'num_apts': len(y),
            'feature_vector_length': X.shape[1],
            'unique_apts': len(np.unique(y)),
            'apt_list': sorted(np.unique(y)),
            'model_params': Config.MODEL_PARAMS,
            'feature_counts': {
                'techniques': len(feature_metadata['all_techniques']),
                'software': len(feature_metadata['all_software']),
                'cve': len(feature_metadata['all_cve'])
            }
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nModel saved to '{filename}'")
    return save_data

# ============================
# 7. MAIN
# ============================
if __name__ == "__main__":
    model, X, y, apt_features, feature_metadata = train_xgboost_model()
    save_xgboost_model(model, X, y, apt_features, feature_metadata)