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
    MODEL_PATH = "6feat_efficient_apt_similarity_nopca.pkl"
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
    FEATURE_COLUMNS = ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']
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
        all_capec = sorted(df['CAPEC_ID'].unique())
        all_cwe = sorted(df['CWE_ID'].unique())
        all_cve = sorted(df['CVE_ID'].unique())
        all_pulse = sorted(df['Pulse_ID'].unique())

        # Create feature-to-index mappings
        technique_to_idx = {t: i for i, t in enumerate(all_techniques)}
        software_to_idx = {s: i for i, s in enumerate(all_software)}
        capec_to_idx = {c: i for i, c in enumerate(all_capec)}
        cwe_to_idx = {w: i for i, w in enumerate(all_cwe)}
        cve_to_idx = {v: i for i, v in enumerate(all_cve)}
        pulse_to_idx = {p: i for i, p in enumerate(all_pulse)}

        # Build a dictionary keyed by APT (using the original unique values, but later we will sort)
        apt_features = {}
        unique_apts_unsorted = df['APTname'].unique()
        for apt in unique_apts_unsorted:
            apt_data = df[df['APTname'] == apt]
            apt_techniques = set(apt_data['techniques_str'].unique())
            apt_software = set(apt_data['software_str'].unique())
            apt_capec = set(apt_data['CAPEC_ID'].unique())
            apt_cwe = set(apt_data['CWE_ID'].unique())
            apt_cve = set(apt_data['CVE_ID'].unique())
            apt_pulse = set(apt_data['Pulse_ID'].unique())

            techniques_vector = [1 if t in apt_techniques else 0 for t in all_techniques]
            software_vector = [1 if s in apt_software else 0 for s in all_software]
            capec_vector = [1 if c in apt_capec else 0 for c in all_capec]
            cwe_vector = [1 if w in apt_cwe else 0 for w in all_cwe]
            cve_vector = [1 if v in apt_cve else 0 for v in all_cve]
            pulse_vector = [1 if p in apt_pulse else 0 for p in all_pulse]

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
        capec_offset = software_offset + len(all_software)
        cwe_offset = capec_offset + len(all_capec)
        cve_offset = cwe_offset + len(all_cwe)
        pulse_offset = cve_offset + len(all_cve)

        feature_metadata = {
            'all_techniques': all_techniques,
            'all_software': all_software,
            'all_capec': all_capec,
            'all_cwe': all_cwe,
            'all_cve': all_cve,
            'all_pulse': all_pulse,
            'techniques_offset': techniques_offset,
            'software_offset': software_offset,
            'capec_offset': capec_offset,
            'cwe_offset': cwe_offset,
            'cve_offset': cve_offset,
            'pulse_offset': pulse_offset,
            'feature_mapping': {
                'techniques': technique_to_idx,
                'software': software_to_idx,
                'capec': capec_to_idx,
                'cwe': cwe_to_idx,
                'cve': cve_to_idx,
                'pulse': pulse_to_idx
            }
        }

        print(f"\nTotal features breakdown:")
        print(f"  Techniques: {len(all_techniques)}")
        print(f"  Software: {len(all_software)}")
        print(f"  CAPEC: {len(all_capec)}")
        print(f"  CWE: {len(all_cwe)}")
        print(f"  CVE: {len(all_cve)}")
        print(f"  Pulse: {len(all_pulse)}")

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

    def create_query_vector(self, techniques, software, capec, cwe, cve, pulse):
        """Build binary vector from evidence using feature mappings."""
        # This method is provided for completeness (used by query scripts)
        feature_mapping = self.feature_metadata.get('feature_mapping', {})
        vector = np.zeros(self.feature_metadata['pulse_offset'] +
                          len(self.feature_metadata['all_pulse']), dtype=np.float64)

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
        if 'capec' in feature_mapping:
            capec_map = feature_mapping['capec']
            for c in capec:
                if c in capec_map:
                    idx = capec_map[c] + self.feature_metadata['capec_offset']
                    vector[idx] = 1.0
        if 'cwe' in feature_mapping:
            cwe_map = feature_mapping['cwe']
            for w in cwe:
                if w in cwe_map:
                    idx = cwe_map[w] + self.feature_metadata['cwe_offset']
                    vector[idx] = 1.0
        if 'cve' in feature_mapping:
            cve_map = feature_mapping['cve']
            for v in cve:
                if v in cve_map:
                    idx = cve_map[v] + self.feature_metadata['cve_offset']
                    vector[idx] = 1.0
        if 'pulse' in feature_mapping:
            pulse_map = feature_mapping['pulse']
            for p in pulse:
                if p in pulse_map:
                    idx = pulse_map[p] + self.feature_metadata['pulse_offset']
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
    return save_data

# ============================
# 7. MAIN
# ============================
if __name__ == "__main__":
    model, X, y, apt_features, feature_metadata = train_xgboost_model()
    save_xgboost_model(model, X, y, apt_features, feature_metadata)