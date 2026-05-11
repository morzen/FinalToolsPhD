# ============================
# 1. IMPORTS
# ============================
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


# ============================
# 2. CONFIGURATION
# ============================
class Config:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "3feat_efficient_apt_similarity_nopca.pkl"
    MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'random_state': 42,
        'tree_method': 'hist',
    }
    # Augmentation settings
    N_AUGMENTED_SAMPLES_PER_APT = 200   # how many partial profiles per APT
    DROP_PROB = 0.3                     # probability of dropping each active feature


# ============================
# 3. DATA LOADING & PROFILE BUILDING
# ============================
class XGBoostDataLoader:
    @staticmethod
    def load_data(data_path=Config.DATA_PATH):
        print("Loading encoded data...")
        df = pd.read_csv(data_path)
        print(f"Data: {len(df)} records, {df.shape[1]} columns")
        return df

    @staticmethod
    def build_apt_profiles(df):
        print("\nBUILDING APT PROFILES (FULL VECTORS)")
        print("=" * 50)

        all_techniques = sorted(df['techniques_str'].dropna().unique())
        all_software   = sorted(df['software_str'].dropna().unique())
        all_cve        = sorted(df['CVE_ID'].dropna().unique())

        # Index mappings
        technique_to_idx = {t: i for i, t in enumerate(all_techniques)}
        offset_soft = len(all_techniques)
        software_to_idx = {s: i + offset_soft for i, s in enumerate(all_software)}

        offset_cve = offset_soft + len(all_software)       # start index for CVE
        cve_to_idx = {v: i + offset_cve for i, v in enumerate(all_cve)}

        total_features = offset_cve + len(all_cve)         # final size
        print(f"Feature vector length: {total_features}")

        # Build one full vector per APT
        apt_profiles = {}
        for apt, group in df.groupby('APTname'):
            techniques = set(group['techniques_str'].dropna().unique())
            software   = set(group['software_str'].dropna().unique())
            cve        = set(group['CVE_ID'].dropna().unique())

            vec = np.zeros(total_features, dtype=np.float64)
            for t in techniques:
                if t in technique_to_idx:
                    vec[technique_to_idx[t]] = 1.0
            for s in software:
                if s in software_to_idx:
                    vec[software_to_idx[s]] = 1.0
            for v in cve:
                if v in cve_to_idx:
                    vec[cve_to_idx[v]] = 1.0

            apt_profiles[apt] = {
                'vector': vec,
                'count': len(group),
                'techniques': techniques,
                'software': software,
                'cve': cve,
            }

        unique_apts = sorted(apt_profiles.keys())
        X_full = np.array([apt_profiles[apt]['vector'] for apt in unique_apts])
        y_full = np.array(unique_apts)

        print(f"Number of APTs: {len(unique_apts)}")
        counts = [apt_profiles[a]['count'] for a in unique_apts]
        print(f"Rows per APT – min: {min(counts)}, avg: {np.mean(counts):.0f}, max: {max(counts)}")

        feature_metadata = {
            'all_techniques': all_techniques,
            'all_software': all_software,
            'all_cve': all_cve,

            'techniques_offset': 0,
            'software_offset': offset_soft,
            'cve_offset': offset_cve,           # now correctly the start index

            'feature_mapping': {
                'techniques': technique_to_idx,
                'software': software_to_idx,
                'cve': cve_to_idx
            }
        }

        return X_full, y_full, apt_profiles, feature_metadata


# ============================
# 4. DATA AUGMENTATION
# ============================
def augment_from_profiles(X_full, y_full, n_samples_per_apt, drop_prob):
    """
    For each APT, create many partial copies by randomly dropping
    a fraction of the active features.
    """
    X_aug, y_aug = [], []
    for i, apt in enumerate(y_full):
        full_vec = X_full[i]
        for _ in range(n_samples_per_apt):
            masked = full_vec.copy()
            active_indices = np.where(masked == 1)[0]
            if len(active_indices) > 0:
                drop_mask = np.random.rand(len(active_indices)) < drop_prob
                # Keep at least one active feature
                if np.all(drop_mask):
                    keep = np.random.randint(len(active_indices))
                    drop_mask[keep] = False
                masked[active_indices[drop_mask]] = 0
            X_aug.append(masked)
            y_aug.append(apt)
    return np.array(X_aug, dtype=np.float64), np.array(y_aug)


# ============================
# 5. PROGRESS CALLBACK
# ============================
class TqdmProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds):
        super().__init__()
        self.bar = tqdm(total=total_rounds, desc="Training XGBoost", unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        self.bar.update(1)
        return False

    def after_training(self, model):
        self.bar.close()
        return model


# ============================
# 6. WRAPPER (picklable)
# ============================
class PicklableXGBWrapper:
    def __init__(self, booster, num_classes, seed):
        self._Booster = booster
        self.n_classes_ = num_classes
        self.objective = 'multi:softprob'
        self.seed = seed
        self.classes_ = np.arange(num_classes)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X, missing=np.nan)
        return self._Booster.predict(dtest)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# ============================
# 7. MODEL CLASS
# ============================
class EfficientXGBoostModel:
    def __init__(self):
        self.model = None
        self.apt_features = {}
        self.apt_encoder = {}
        self.reverse_encoder = {}
        self.feature_metadata = {}
        self.all_binary_vectors = {}

    def train(self, X_full, y_full, apt_profiles, feature_metadata):
        print(f"\nTRAINING XGBOOST ON AUGMENTED APT PROFILES")
        print("=" * 50)

        self.feature_metadata = feature_metadata

        # Build rich apt_features dict (same structure as before)
        self.apt_features = {
            apt: {
                'binary_vector': apt_profiles[apt]['vector'].tolist(),
                'techniques': apt_profiles[apt]['techniques'],
                'software': apt_profiles[apt]['software'],
                'cve': apt_profiles[apt]['cve'],
                'count': apt_profiles[apt]['count'],
            }
            for apt in y_full
        }

        # Encode labels
        unique_apts = np.unique(y_full)
        self.apt_encoder = {apt: idx for idx, apt in enumerate(unique_apts)}
        self.reverse_encoder = {v: k for k, v in self.apt_encoder.items()}

        # Store full binary vectors for all APTs (needed by query for reporting)
        for apt in unique_apts:
            self.all_binary_vectors[apt] = X_full[self.apt_encoder[apt]]

        # Augment training data
        print(f"Generating augmented samples ({Config.N_AUGMENTED_SAMPLES_PER_APT} per APT)...")
        X_train, y_train = augment_from_profiles(
            X_full, y_full,
            n_samples_per_apt=Config.N_AUGMENTED_SAMPLES_PER_APT,
            drop_prob=Config.DROP_PROB
        )
        y_encoded = np.array([self.apt_encoder[apt] for apt in y_train])

        # DMatrix (dense is fine now)
        dtrain = xgb.DMatrix(X_train, label=y_encoded, missing=np.nan)

        num_class = len(self.apt_encoder)
        params = {
            'objective': 'multi:softprob',
            'max_depth': Config.MODEL_PARAMS['max_depth'],
            'learning_rate': Config.MODEL_PARAMS['learning_rate'],
            'subsample': Config.MODEL_PARAMS['subsample'],
            'colsample_bytree': Config.MODEL_PARAMS['colsample_bytree'],
            'reg_lambda': Config.MODEL_PARAMS['reg_lambda'],
            'reg_alpha': Config.MODEL_PARAMS['reg_alpha'],
            'num_class': num_class,
            'tree_method': Config.MODEL_PARAMS['tree_method'],
            'seed': Config.MODEL_PARAMS['random_state'],
            'verbosity': 0,
        }
        total_trees = Config.MODEL_PARAMS['n_estimators']

        callback = TqdmProgressCallback(total_trees)
        start_time = time.time()
        bst = xgb.train(params, dtrain, num_boost_round=total_trees, callbacks=[callback])
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed/60:.1f} minutes.")

        self.model = PicklableXGBWrapper(bst, num_class, Config.MODEL_PARAMS['random_state'])

        # Accuracy on full profiles (they should be recognised perfectly)
        y_full_enc = np.array([self.apt_encoder[a] for a in y_full])
        preds_full = self.model.predict(X_full)
        acc_full = np.mean(preds_full == y_full_enc)
        print(f"Accuracy on full APT profiles: {acc_full:.4f}")

        # Accuracy on augmented training set (informative)
        preds_aug = self.model.predict(X_train)
        acc_aug = np.mean(preds_aug == y_encoded)
        print(f"Accuracy on augmented training set: {acc_aug:.4f}")


# ============================
# 8. TRAINING FUNCTION
# ============================
def train_xgboost_model():
    print("XGBOOST TRAINING PIPELINE – AUGMENTED APT PROFILES")
    print("=" * 60)
    loader = XGBoostDataLoader()
    df = loader.load_data()
    X_full, y_full, apt_profiles, feature_metadata = loader.build_apt_profiles(df)

    model = EfficientXGBoostModel()
    model.train(X_full, y_full, apt_profiles, feature_metadata)
    return model, X_full, y_full, model.apt_features, feature_metadata


# ============================
# 9. SAVE MODEL
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
            'approach': 'xgboost_augmented_profiles',
            'num_apts': len(model.apt_encoder),
            'feature_vector_length': X.shape[1],
            'apt_list': sorted(list(model.apt_encoder.keys())),
            'model_params': Config.MODEL_PARAMS,
            'augmentation': {
                'n_samples_per_apt': Config.N_AUGMENTED_SAMPLES_PER_APT,
                'drop_prob': Config.DROP_PROB,
            },
            'feature_counts': {
                'techniques': len(feature_metadata['all_techniques']),
                'software': len(feature_metadata['all_software']),
                'cve': len(feature_metadata['all_cve']),
            }
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nModel saved to '{filename}'")


# ============================
# 10. MAIN
# ============================
if __name__ == "__main__":
    model, X, y, apt_features, feature_metadata = train_xgboost_model()
    save_xgboost_model(model, X, y, apt_features, feature_metadata)