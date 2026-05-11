import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import xgboost as xgb
from itertools import islice
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# 🔽  WRAPPER CLASS (must match train.py) so pickle can load the model
# ============================================================================
class PicklableXGBWrapper:
    """Wrapper to mimic XGBClassifier for predict_proba."""
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
# ============================================================================


#####################################################################
# EFFICIENT MODEL PREDICTOR (updated to use model probabilities)
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model using binary vectors."""
    def __init__(self, model, apt_features, apt_encoder, feature_metadata):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
        self.feature_metadata = feature_metadata
        self.last_binary_vector = None

    def predict_proba_vector(self, vector):
        """
        Return class probabilities from the model for a single vector.
        """
        return self.model.predict_proba(vector.reshape(1, -1))[0]

    def predict_vector_similarity(self, vector1, vector2):
        """Cosine similarity (kept for quick checks if needed)."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.clip(dot_product / (norm1 * norm2), 0.0, 1.0)

    def create_vector_from_evidence(self, evidence):
        """Build binary vector from evidence (same as in query.py)."""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        cve = evidence.get('cve', [])

        total_length = (self.feature_metadata['cve_offset'] +
                len(self.feature_metadata['all_cve']))
        
        vector = np.zeros(total_length, dtype=np.float64)

        fm = self.feature_metadata['feature_mapping']

        # Techniques
        if 'techniques' in fm:
            tech_map = fm['techniques']
            for t in techniques:
                if t in tech_map:
                    vector[tech_map[t]] = 1.0
        # Software
        if 'software' in fm:
            soft_map = fm['software']
            for s in software:
                if s in soft_map:
                    idx = soft_map[s]
                    vector[idx] = 1.0
        # CVE
        if 'cve' in fm:
            cve_map = fm['cve']
            for v in cve:
                if v in cve_map:
                    idx = cve_map[v]
                    vector[idx] = 1.0

        self.last_binary_vector = vector.copy()
        return vector


def load_efficient_xgboost_model():
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('3feat_efficient_apt_similarity_nopca.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print(" Loaded model (no PCA).")
        return saved_data
    except FileNotFoundError:
        print(" Model file not found. Please run train.py first.")
        return None


def load_apt_profiles_from_data():
    """Load APT profiles from the encoded data file."""
    print(" LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].unique()),
            'software': set(apt_data['software_str'].unique()),
            'cve': set(apt_data['CVE_ID'].unique()),
            'total_features': len(apt_data)
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles


#####################################################################
# FEATURE ORDERING VERIFICATION
#####################################################################

def verify_feature_ordering(saved_data):
    print("\n" + "=" * 80)
    print("FEATURE ORDERING VERIFICATION")
    print("=" * 80)
    fm = saved_data['feature_metadata']
    print(f"\nFeature counts:")
    print(f"  Techniques: {len(fm['all_techniques'])}")
    print(f"  Software: {len(fm['all_software'])}")
    print(f"  CVE: {len(fm['all_cve'])}")
    print(f"\nOffsets:")
    print(f"  techniques_offset: {fm['techniques_offset']}")
    print(f"  software_offset: {fm['software_offset']}")
    print(f"  cve_offset: {fm['cve_offset']}")
    if 'feature_mapping' in fm:
        print("\n✅ Feature mappings found.")
    else:
        print("\n⚠️  No feature mappings found.")
    if 'all_binary_vectors' in saved_data:
        print(f"\n✅ Binary vectors saved – count: {len(saved_data['all_binary_vectors'])}")
    else:
        print("\n⚠️  Binary vectors not saved.")
    return True


#####################################################################
# VECTOR CONSISTENCY CHECK
#####################################################################

def test_vector_consistency(saved_data):
    print("\n" + "=" * 80)
    print("VECTOR CONSISTENCY CHECK")
    print("=" * 80)
    profiles = load_apt_profiles_from_data()
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin']
                        for _, row in df_mapping.iterrows()}

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    all_binary = saved_data.get('all_binary_vectors', {})
    print(f"\nTesting first 10 APTs...")
    total_tested = 0
    good, partial, poor = 0, 0, 0
    for i, (apt_id, stored_vec) in enumerate(list(all_binary.items())[:10]):
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile = profiles.get(apt_id, {})
        if not profile:
            continue
        evidence = {
            'techniques': list(profile.get('techniques', [])),
            'software': list(profile.get('software', [])),
            'cve': list(profile.get('cve', []))
        }
        v1 = predictor.create_vector_from_evidence(evidence)
        v2 = predictor.create_vector_from_evidence(evidence)
        sim_stored = predictor.predict_vector_similarity(v1, stored_vec)
        sim_rep = predictor.predict_vector_similarity(v1, v2)
        print(f"\n{apt_name} (ID: {apt_id}):")
        print(f"  Stored vs New similarity: {sim_stored:.6f}")
        print(f"  Reproducibility (v1 vs v2): {sim_rep:.6f}")
        total_tested += 1
        if sim_stored > 0.9:
            good += 1
        elif sim_stored > 0.5:
            partial += 1
        else:
            poor += 1
    print("\n" + "-" * 60)
    print(f"Summary: good >0.9: {good}, partial 0.5‑0.9: {partial}, poor <0.5: {poor}")


#####################################################################
# SIMILARITY MATRIX USING MODEL PROBABILITIES
#####################################################################

def create_similarity_matrix_efficient(saved_data):
    """
    Build a matrix of model probabilities: row = query APT (full profile),
    column = target APT, value = probability assigned by the model.
    """
    print("\nCreating similarity matrix from model probabilities...")
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin']
                        for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )

    n = len(apt_names)
    prob_matrix = np.zeros((n, n))
    print(f"Computing probability matrix for {n} APTs...")
    for i, query_apt_id in enumerate(apt_ids):
        if i % 20 == 0:
            print(f"  row {i}/{n}")

        # Build query vector from the full profile of this APT
        prof = profiles[query_apt_id]
        evidence = {
            'techniques': list(prof['techniques']),
            'software': list(prof['software']),
            'cve': list(prof['cve'])
        }
        qvec = predictor.create_vector_from_evidence(evidence)
        probs = predictor.predict_proba_vector(qvec)   # shape (num_classes,)

        # Map probabilities back to APT ids using the encoder
        for j, target_apt_id in enumerate(apt_ids):
            if target_apt_id in predictor.apt_encoder:
                idx = predictor.apt_encoder[target_apt_id]
                prob_matrix[i, j] = probs[idx]
            else:
                prob_matrix[i, j] = 0.0

    prob_df = pd.DataFrame(prob_matrix, index=apt_names, columns=apt_names)
    return prob_df, apt_names


def evaluate_similarity_matrix_efficient(prob_df, apt_names):
    """Evaluate self‑recognition using the probability matrix."""
    print("\n" + "=" * 80)
    print("EVALUATION 1: MODEL PROBABILITY MATRIX SELF‑RECOGNITION")
    print("=" * 80)
    for top in [1, 5, 10, 20]:
        success = 0
        for i, name in enumerate(apt_names):
            # check if the APT is within the top‑K highest probabilities
            row = prob_df.iloc[i]
            top_apts = row.nlargest(top).index
            if name in top_apts:
                success += 1
        total = len(apt_names)
        print(f"  Top {top}: {success}/{total} ({success/total*100:.2f}%)")


#####################################################################
# FULL DATA QUERY (using model probabilities)
#####################################################################

def evaluate_full_data_query(saved_data):
    print("\n" + "=" * 80)
    print("EVALUATION 2: FULL DATA QUERY (using model probabilities)")
    print("=" * 80)
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin']
                        for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )

    results = {1: 0, 5: 0, 10: 0, 20: 0}
    rank_sum = 0
    rank_count = 0

    for i, apt_id in enumerate(apt_ids):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_ids)}...")
        prof = profiles[apt_id]
        evidence = {
            'techniques': list(prof['techniques']),
            'software': list(prof['software']),
            'cve': list(prof['cve'])
        }
        qvec = predictor.create_vector_from_evidence(evidence)
        probs = predictor.predict_proba_vector(qvec)

        # Pair each apt_name with its probability
        prob_list = []
        for j, target_id in enumerate(apt_ids):
            if target_id in predictor.apt_encoder:
                idx = predictor.apt_encoder[target_id]
                prob_list.append((apt_names[j], probs[idx]))
            else:
                prob_list.append((apt_names[j], 0.0))
        prob_list.sort(key=lambda x: x[1], reverse=True)

        # Determine rank of the true APT
        target_name = apt_names[i]
        rank = next((k+1 for k, (name, _) in enumerate(prob_list) if name == target_name), len(apt_ids)+1)
        rank_sum += rank
        rank_count += 1
        for top in results:
            if rank <= top:
                results[top] += 1

    total = len(apt_ids)
    avg_rank = rank_sum / rank_count if rank_count else 0
    print("\nResults:")
    for top in sorted(results):
        print(f"  Top {top}: {results[top]}/{total} ({results[top]/total*100:.2f}%)")
    print(f"  Average self‑rank: {avg_rank:.2f}")


#####################################################################
# DIVIDED DATA QUERY (partial evidence) using model probabilities
#####################################################################

def evaluate_profile_division_efficient(saved_data, divider=2, top_x_values=[1,5,10,20]):
    print("\n" + "=" * 80)
    print(f"EVALUATION 3: DIVIDED DATA (1/{divider} of features) – model probabilities")
    print("=" * 80)
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin']
                        for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )

    results_dict = {}
    for i, apt_id in enumerate(apt_ids):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_ids)}...")
        prof = profiles[apt_id]

        # Build divided evidence (random subset)
        def random_subset(lst):
            if not lst:
                return []
            lst = list(lst)
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            return lst[:sz]

        evidence = {
            'techniques': random_subset(prof['techniques']),
            'software': random_subset(prof['software']),
            'cve': random_subset(prof['cve'])
        }

        qvec = predictor.create_vector_from_evidence(evidence)
        probs = predictor.predict_proba_vector(qvec)

        # Build (name, prob) list
        prob_list = []
        for j, target_id in enumerate(apt_ids):
            if target_id in predictor.apt_encoder:
                idx = predictor.apt_encoder[target_id]
                prob_list.append((apt_names[j], probs[idx]))
            else:
                prob_list.append((apt_names[j], 0.0))
        prob_list.sort(key=lambda x: x[1], reverse=True)
        results_dict[apt_names[i]] = {name: prob for name, prob in prob_list}

    # Top‑K accuracy
    for top in top_x_values:
        success = 0
        for target_name, ranked_list in results_dict.items():
            top_apts = set(list(ranked_list.keys())[:top])
            if target_name in top_apts:
                success += 1
        total = len(results_dict)
        print(f"  Top {top}, divider {divider}: {success}/{total} ({success/total*100:.2f}%)")


#####################################################################
# HEATMAP PLOTTING
#####################################################################

def plot_similarity_heatmap(sim_df):
    print("\n" + "=" * 80)
    print("PLOTTING HEATMAP (probability matrix)")
    print("=" * 80)
    plt.figure(figsize=(16, 14))
    # Custom colormap: red (low) → yellow → green (high)
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    sns.heatmap(sim_df, cmap=cmap, center=0.5, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Probability'})
    plt.title("APT Probability Matrix (Model Output)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_similarity_matrix_nopca.png', dpi=300, bbox_inches='tight')
    plt.show()


#####################################################################
# MAIN
#####################################################################

def main():
    print(" EFFICIENT XGBOOST APT EVALUATION (USING MODEL PROBABILITIES)")
    print("=" * 60)
    saved_data = load_efficient_xgboost_model()
    if saved_data is None:
        return

    ti = saved_data['training_info']
    print(f"\nModel info:")
    print(f"  APTs: {ti['num_apts']}")               # fixed key
    print(f"  Feature vector length: {ti['feature_vector_length']}")
    print(f"  Feature counts: {ti['feature_counts']}")

    verify_feature_ordering(saved_data)
    test_vector_consistency(saved_data)

    prob_df, apt_names = create_similarity_matrix_efficient(saved_data)
    if prob_df is not None:
        print("\nSample 5x5 of probability matrix:")
        print(prob_df.iloc[:5, :5])
        print(f"\nMatrix shape: {prob_df.shape}")

        evaluate_similarity_matrix_efficient(prob_df, apt_names)
        evaluate_full_data_query(saved_data)
        evaluate_profile_division_efficient(saved_data, divider=2)

        plot_similarity_heatmap(prob_df)
    else:
        print("Failed to create probability matrix.")

if __name__ == "__main__":
    main()