import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from itertools import islice

#####################################################################
# EFFICIENT MODEL PREDICTOR (NO PCA – USES BINARY VECTORS)
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model using binary vectors (no PCA)"""
    def __init__(self, model, apt_features, apt_encoder, feature_metadata):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
        self.feature_metadata = feature_metadata
        self.last_binary_vector = None  # For debugging

    def predict_vector_similarity(self, vector1, vector2):
        """Cosine similarity on binary vectors."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, 0.0, 1.0)

    def create_vector_from_evidence(self, evidence):
        """Build binary vector from evidence using feature mappings (no PCA)."""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        capec = evidence.get('capec', [])
        cwe = evidence.get('cwe', [])
        cve = evidence.get('cve', [])
        pulse = evidence.get('pulse', [])

        dtype = np.float64
        total_length = (self.feature_metadata['pulse_offset'] +
                        len(self.feature_metadata['all_pulse']))
        vector = np.zeros(total_length, dtype=dtype)

        feature_mapping = self.feature_metadata.get('feature_mapping', {})

        if feature_mapping:
            # Techniques
            if 'techniques' in feature_mapping:
                tech_map = feature_mapping['techniques']
                for t in techniques:
                    if t in tech_map:
                        vector[tech_map[t]] = 1.0
            # Software
            if 'software' in feature_mapping:
                soft_map = feature_mapping['software']
                for s in software:
                    if s in soft_map:
                        idx = soft_map[s] + self.feature_metadata['software_offset']
                        vector[idx] = 1.0
            # CAPEC
            if 'capec' in feature_mapping:
                capec_map = feature_mapping['capec']
                for c in capec:
                    if c in capec_map:
                        idx = capec_map[c] + self.feature_metadata['capec_offset']
                        vector[idx] = 1.0
            # CWE
            if 'cwe' in feature_mapping:
                cwe_map = feature_mapping['cwe']
                for w in cwe:
                    if w in cwe_map:
                        idx = cwe_map[w] + self.feature_metadata['cwe_offset']
                        vector[idx] = 1.0
            # CVE
            if 'cve' in feature_mapping:
                cve_map = feature_mapping['cve']
                for v in cve:
                    if v in cve_map:
                        idx = cve_map[v] + self.feature_metadata['cve_offset']
                        vector[idx] = 1.0
            # Pulse
            if 'pulse' in feature_mapping:
                pulse_map = feature_mapping['pulse']
                for p in pulse:
                    if p in pulse_map:
                        idx = pulse_map[p] + self.feature_metadata['pulse_offset']
                        vector[idx] = 1.0
        else:
            # Fallback to index lookup (less reliable)
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

        self.last_binary_vector = vector.copy()
        return vector  # No PCA transformation!

def load_efficient_xgboost_model():
    """Load the model (expects 'all_binary_vectors' in saved data)."""
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('6feat_efficient_apt_similarity_nopca.pkl', 'rb') as f:
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
            'capec': set(apt_data['CAPEC_ID'].unique()),
            'cwe': set(apt_data['CWE_ID'].unique()),
            'cve': set(apt_data['CVE_ID'].unique()),
            'pulse': set(apt_data['Pulse_ID'].unique()),
            'total_features': len(apt_data)
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles

#####################################################################
# FEATURE ORDERING VERIFICATION
#####################################################################

def verify_feature_ordering(saved_data):
    """Check that feature metadata is present."""
    print("\n" + "="*80)
    print("FEATURE ORDERING VERIFICATION")
    print("="*80)
    fm = saved_data['feature_metadata']
    print(f"\nFeature counts:")
    print(f"  Techniques: {len(fm['all_techniques'])}")
    print(f"  Software: {len(fm['all_software'])}")
    print(f"  CAPEC: {len(fm['all_capec'])}")
    print(f"  CWE: {len(fm['all_cwe'])}")
    print(f"  CVE: {len(fm['all_cve'])}")
    print(f"  Pulse: {len(fm['all_pulse'])}")
    print(f"\nOffsets:")
    print(f"  techniques_offset: {fm['techniques_offset']}")
    print(f"  software_offset: {fm['software_offset']}")
    print(f"  capec_offset: {fm['capec_offset']}")
    print(f"  cwe_offset: {fm['cwe_offset']}")
    print(f"  cve_offset: {fm['cve_offset']}")
    print(f"  pulse_offset: {fm['pulse_offset']}")
    if 'feature_mapping' in fm:
        print("\n✅ Feature mappings found – will use for consistent indexing.")
    else:
        print("\n⚠️  No feature mappings found – may be inconsistent.")
    if 'all_binary_vectors' in saved_data:
        print(f"\n✅ Binary vectors saved – number: {len(saved_data['all_binary_vectors'])}")
    else:
        print("\n⚠️  Binary vectors not saved – cannot proceed.")
    return True

#####################################################################
# VECTOR CONSISTENCY CHECK (BASIC)
#####################################################################

def test_vector_consistency(saved_data):
    """Test if newly created binary vectors match stored ones."""
    print("\n" + "="*80)
    print("VECTOR CONSISTENCY CHECK")
    print("="*80)
    profiles = load_apt_profiles_from_data()
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    all_binary_vectors = saved_data.get('all_binary_vectors', {})
    print(f"\nTesting first 10 APTs...")
    print("-" * 60)
    total_tested = 0
    good = 0
    partial = 0
    poor = 0
    for i, (apt_id, stored_vec) in enumerate(list(all_binary_vectors.items())[:10]):
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile = profiles.get(apt_id, {})
        if not profile:
            continue
        evidence = {
            'techniques': list(profile.get('techniques', [])),
            'software': list(profile.get('software', [])),
            'capec': list(profile.get('capec', [])),
            'cwe': list(profile.get('cwe', [])),
            'cve': list(profile.get('cve', [])),
            'pulse': list(profile.get('pulse', []))
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
# SIMILARITY MATRIX USING STORED BINARY VECTORS
#####################################################################

def create_similarity_matrix_efficient(saved_data):
    """Build similarity matrix from stored binary vectors."""
    print("\nCreating similarity matrix from stored binary vectors...")
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    all_binary = saved_data.get('all_binary_vectors', {})
    if not all_binary:
        print("⚠️  No binary vectors found.")
        return None, apt_names

    # Build list in same order
    vecs = []
    for aid in apt_ids:
        if aid in all_binary:
            vecs.append(all_binary[aid])
        else:
            # Fallback – reconstruct from profile (should not happen)
            profile = profiles[aid]
            ev = {
                'techniques': list(profile['techniques']),
                'software': list(profile['software']),
                'capec': list(profile['capec']),
                'cwe': list(profile['cwe']),
                'cve': list(profile['cve']),
                'pulse': list(profile['pulse'])
            }
            vecs.append(predictor.create_vector_from_evidence(ev))

    n = len(apt_names)
    sim_mat = np.zeros((n, n))
    print(f"Computing pairwise cosine similarities for {n} APTs...")
    for i in range(n):
        if i % 20 == 0:
            print(f"  row {i}/{n}")
        for j in range(n):
            sim_mat[i, j] = predictor.predict_vector_similarity(vecs[i], vecs[j])
    sim_df = pd.DataFrame(sim_mat, index=apt_names, columns=apt_names)
    return sim_df, apt_names

def evaluate_similarity_matrix_efficient(sim_df, apt_names):
    """Evaluate stored‑vector self‑similarity."""
    print("\n" + "="*80)
    print("EVALUATION 1: STORED BINARY VECTORS SELF‑SIMILARITY")
    print("="*80)
    for top_x in [1,5,10,20]:
        success = 0
        for i, name in enumerate(apt_names):
            top = sim_df.iloc[i].nlargest(top_x)
            if name in top.index:
                success += 1
        total = len(apt_names)
        print(f"Top {top_x}: {success}/{total} ({success/total*100:.2f}%)")

#####################################################################
# FULL DATA QUERY (simulate real query with full profile)
#####################################################################

def evaluate_full_data_query(saved_data):
    print("\n" + "="*80)
    print("EVALUATION 2: FULL DATA QUERY (complete profile)")
    print("="*80)
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    all_binary = saved_data.get('all_binary_vectors', {})
    if not all_binary:
        print("No binary vectors.")
        return

    # Build target database
    target_vecs = []
    for aid in apt_ids:
        if aid in all_binary:
            target_vecs.append(all_binary[aid])
        else:
            # reconstruct
            p = profiles[aid]
            ev = {
                'techniques': list(p['techniques']),
                'software': list(p['software']),
                'capec': list(p['capec']),
                'cwe': list(p['cwe']),
                'cve': list(p['cve']),
                'pulse': list(p['pulse'])
            }
            target_vecs.append(predictor.create_vector_from_evidence(ev))

    results_by_top = {1:0,5:0,10:0,20:0}
    rank_sum = 0
    rank_count = 0

    for i, target_name in enumerate(apt_names):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_names)}...")
        aid = apt_ids[i]
        prof = profiles[aid]
        evidence = {
            'techniques': list(prof['techniques']),
            'software': list(prof['software']),
            'capec': list(prof['capec']),
            'cwe': list(prof['cwe']),
            'cve': list(prof['cve']),
            'pulse': list(prof['pulse'])
        }
        qvec = predictor.create_vector_from_evidence(evidence)
        sims = []
        for j, tvec in enumerate(target_vecs):
            sim = predictor.predict_vector_similarity(qvec, tvec)
            sims.append((apt_names[j], sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, _) in enumerate(sims, 1):
            if name == target_name:
                rank_sum += rank
                rank_count += 1
                for top in results_by_top:
                    if rank <= top:
                        results_by_top[top] += 1
                break

    total = len(apt_names)
    avg_rank = rank_sum / rank_count if rank_count else 0
    print("\nResults:")
    for top in sorted(results_by_top):
        print(f"  Top {top}: {results_by_top[top]}/{total} ({results_by_top[top]/total*100:.2f}%)")
    print(f"  Average self‑rank: {avg_rank:.2f}")

#####################################################################
# DIVIDED DATA QUERY (partial evidence)
#####################################################################

def evaluate_profile_division_efficient(saved_data, divider=2, top_x_values=[1,5,10,20]):
    print("\n" + "="*80)
    print(f"EVALUATION 3: DIVIDED DATA (1/{divider} of features)")
    print("="*80)
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]

    predictor = EfficientXGBoostPredictor(
        saved_data['model'],
        saved_data['apt_features'],
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    all_binary = saved_data.get('all_binary_vectors', {})
    if not all_binary:
        print("No binary vectors.")
        return

    # Build target database
    target_vecs = []
    for aid in apt_ids:
        if aid in all_binary:
            target_vecs.append(all_binary[aid])
        else:
            p = profiles[aid]
            ev = {
                'techniques': list(p['techniques']),
                'software': list(p['software']),
                'capec': list(p['capec']),
                'cwe': list(p['cwe']),
                'cve': list(p['cve']),
                'pulse': list(p['pulse'])
            }
            target_vecs.append(predictor.create_vector_from_evidence(ev))

    resultdict = {}
    for i, target_name in enumerate(apt_names):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_names)}...")
        aid = apt_ids[i]
        prof = profiles[aid]
        # create divided evidence
        div_ev = {
            'techniques': [],
            'software': [],
            'capec': [],
            'cwe': [],
            'cve': [],
            'pulse': []
        }
        # techniques
        if prof['techniques']:
            lst = list(prof['techniques'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['techniques'] = lst[:sz]
        # software
        if prof['software']:
            lst = list(prof['software'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['software'] = lst[:sz]
        # capec
        if prof['capec']:
            lst = list(prof['capec'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['capec'] = lst[:sz]
        # cwe
        if prof['cwe']:
            lst = list(prof['cwe'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['cwe'] = lst[:sz]
        # cve
        if prof['cve']:
            lst = list(prof['cve'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['cve'] = lst[:sz]
        # pulse
        if prof['pulse']:
            lst = list(prof['pulse'])
            random.shuffle(lst)
            sz = max(1, len(lst) // divider)
            div_ev['pulse'] = lst[:sz]

        qvec = predictor.create_vector_from_evidence(div_ev)
        sims = []
        for j, tvec in enumerate(target_vecs):
            sim = predictor.predict_vector_similarity(qvec, tvec)
            sims.append((apt_names[j], sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        resultdict[target_name] = {name: s for name, s in sims}

    for topX in top_x_values:
        success = 0
        for target, ranklist in resultdict.items():
            top_items = dict(list(ranklist.items())[:topX])
            if target in top_items:
                success += 1
        total = len(resultdict)
        print(f"Top {topX}, divider {divider}: {success}/{total} ({success/total*100:.2f}%)")

#####################################################################
# HEATMAP PLOTTING
#####################################################################

def plot_similarity_heatmap(sim_df):
    print("\n" + "="*80)
    print("PLOTTING HEATMAP")
    print("="*80)
    plt.figure(figsize=(16,14))
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    sns.heatmap(sim_df, cmap=cmap, center=0.5, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink':0.8, 'label':'Similarity'})
    plt.title("APT Similarity Matrix (Binary Vectors, No PCA)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_similarity_matrix_nopca.png', dpi=300, bbox_inches='tight')
    plt.show()

#####################################################################
# MAIN
#####################################################################

def main():
    print(" EFFICIENT XGBOOST APT SIMILARITY EVALUATION (NO PCA)")
    print("=" * 60)
    saved_data = load_efficient_xgboost_model()
    if saved_data is None:
        return
    print(f"\nModel info:")
    ti = saved_data['training_info']
    print(f"  APTs: {ti['unique_apts']}")
    print(f"  Feature vector length: {ti['feature_vector_length']}")
    print(f"  Feature counts: {ti['feature_counts']}")

    verify_feature_ordering(saved_data)
    test_vector_consistency(saved_data)

    sim_df, apt_names = create_similarity_matrix_efficient(saved_data)
    if sim_df is not None:
        print("\nSample 5x5:")
        print(sim_df.iloc[:5, :5])
        print(f"\nMatrix shape: {sim_df.shape}")

        evaluate_similarity_matrix_efficient(sim_df, apt_names)
        evaluate_full_data_query(saved_data)
        evaluate_profile_division_efficient(saved_data, divider=2)

        plot_similarity_heatmap(sim_df)
    else:
        print("Failed to create similarity matrix.")

if __name__ == "__main__":
    main()