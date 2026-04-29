import pandas as pd
import pickle
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import random
import math
import xgboost as xgb

# ============================================================================
# 🔽  WRAPPER CLASS (same as train.py) so pickle can load the model
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
        return self._Booster.predict(dtest)   # shape (n_samples, n_classes)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
# ============================================================================


#####################################################################
# EFFICIENT MODEL PREDICTOR FOR XGBOOST BINARY VECTORS (NO PCA)
#####################################################################

class EfficientXGBoostPredictorBinary:
    """Predictor for the XGBoost model using binary vectors."""
    def __init__(self, model, apt_features, apt_encoder, feature_metadata):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
        self.feature_metadata = feature_metadata

    def predict_proba_vector(self, vector):
        """Return class probabilities from the model."""
        return self.model.predict_proba(vector.reshape(1, -1))[0]

    def create_vector_from_evidence(self, evidence):
        """Build binary vector from evidence (matching train.py)."""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        capec = evidence.get('capec', [])
        cwe = evidence.get('cwe', [])
        cve = evidence.get('cve', [])
        pulse = evidence.get('pulse', [])

        total_length = (self.feature_metadata['pulse_offset'] +
                        len(self.feature_metadata['all_pulse']))
        vector = np.zeros(total_length, dtype=np.float64)

        fm = self.feature_metadata['feature_mapping']
        # Techniques
        if 'techniques' in fm:
            for t in techniques:
                if t in fm['techniques']:
                    vector[fm['techniques'][t]] = 1.0
        # Software
        if 'software' in fm:
            for s in software:
                if s in fm['software']:
                    vector[fm['software'][s]] = 1.0
        # CAPEC
        if 'capec' in fm:
            for c in capec:
                if c in fm['capec']:
                    vector[fm['capec'][c]] = 1.0
        # CWE
        if 'cwe' in fm:
            for w in cwe:
                if w in fm['cwe']:
                    vector[fm['cwe'][w]] = 1.0
        # CVE
        if 'cve' in fm:
            for v in cve:
                if v in fm['cve']:
                    vector[fm['cve'][v]] = 1.0
        # Pulse
        if 'pulse' in fm:
            for p in pulse:
                if p in fm['pulse']:
                    vector[fm['pulse'][p]] = 1.0

        return vector


def load_efficient_xgboost_model_nopca():
    print(" LOADING EFFICIENT XGBOOST NO-PCA MODEL...")
    try:
        with open('6feat_efficient_apt_similarity_nopca.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print(" Loaded model.")
        return saved_data
    except FileNotFoundError:
        print(" Model file not found. Run train.py first.")
        return None


def load_apt_profiles_from_data():
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


def load_mappings():
    """Load mappings and APT name mapping."""
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    technique_mapping = {}
    software_mapping = {}
    capec_mapping = {}
    cwe_mapping = {}
    cve_mapping = {}
    pulse_mapping = {}

    for _, row in df_mapping.iterrows():
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            software_mapping[row['software_Origin']] = row['software_Encoded']
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']

    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        if row['APTnameEncoded'] not in apt_name_mapping:
            apt_name_mapping[row['APTnameEncoded']] = row['APTnameOrigin']

    return technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping


# ============================================================
# UNIFIED PER-FEATURE 30/70 MIX (FULL)
# ============================================================
def sample_mixed_features(attacker_profile, mimicked_profile, rng=None,
                          attacker_fraction=0.3, max_count=10):
    if rng is None:
        rng = random.Random()
    mixed = {}
    categories = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
    for cat in categories:
        atk_list = list(attacker_profile.get(cat, []))
        mim_list = list(mimicked_profile.get(cat, []))
        if not atk_list and not mim_list:
            mixed[cat] = []
            continue
        result = []
        atk_copy = atk_list[:]
        mim_copy = mim_list[:]
        total_needed = min(max_count, len(atk_copy) + len(mim_copy))
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_copy:
                chosen = rng.choice(atk_copy)
                result.append(chosen)
                atk_copy.remove(chosen)
            elif mim_copy:
                chosen = rng.choice(mim_copy)
                result.append(chosen)
                mim_copy.remove(chosen)
            else:
                break
        mixed[cat] = result
    return mixed


def sample_mixed_features_partial(attacker_profile, mimicked_profile, rng=None,
                                  attacker_fraction=0.3, max_count=10,
                                  volume_fraction=0.5):
    full_mixed = sample_mixed_features(attacker_profile, mimicked_profile, rng,
                                       attacker_fraction, max_count)
    partial = {}
    for cat, feats in full_mixed.items():
        feats = list(feats)
        rng.shuffle(feats)
        keep = max(1, int(len(feats) * volume_fraction)) if feats else 0
        partial[cat] = feats[:keep]
    return partial


# ============================================================
# WORKERS (UPDATED TO USE MODEL PROBABILITIES)
# ============================================================
def process_single_false_flag_binary(args):
    (attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, software_mapping,
     capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping,
     profiles, apt_ids, seed) = args

    rng = random.Random(seed)
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")

    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]

    # 30/70 mixed evidence (full volume)
    mixed_evidence = sample_mixed_features(attacker_profile, mimicked_profile, rng,
                                           attacker_fraction=0.3, max_count=10)

    predictor = EfficientXGBoostPredictorBinary(
        saved_data['model'], saved_data['apt_features'],
        saved_data['apt_encoder'], saved_data['feature_metadata']
    )

    query_vector = predictor.create_vector_from_evidence(mixed_evidence)
    probs = predictor.predict_proba_vector(query_vector)   # shape (num_classes,)

    # Map probabilities to APT IDs
    prob_list = []
    for apt_id in apt_ids:
        if apt_id in predictor.apt_encoder:
            idx = predictor.apt_encoder[apt_id]
            prob_list.append((apt_id, probs[idx]))
        else:
            prob_list.append((apt_id, 0.0))
    prob_list.sort(key=lambda x: x[1], reverse=True)

    attacker_rank = next((i+1 for i, (aid, _) in enumerate(prob_list) if aid == attacker_apt_id), None)
    mimicked_rank = next((i+1 for i, (aid, _) in enumerate(prob_list) if aid == mimicked_apt_id), None)
    attacker_prob = probs[predictor.apt_encoder[attacker_apt_id]] if attacker_apt_id in predictor.apt_encoder else 0.0
    mimicked_prob = probs[predictor.apt_encoder[mimicked_apt_id]] if mimicked_apt_id in predictor.apt_encoder else 0.0

    return {
        'key': f"{attacker_name}→{mimicked_name}",
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_probability': attacker_prob,
        'mimicked_probability': mimicked_prob
    }


def process_single_false_flag_partial_binary(args):
    (attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, software_mapping,
     capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping,
     profiles, apt_ids, seed) = args

    rng = random.Random(seed)
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")

    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]

    # Partial evidence: full mix then volume reduction
    partial_evidence = sample_mixed_features_partial(attacker_profile, mimicked_profile, rng,
                                                     attacker_fraction=0.3, max_count=10,
                                                     volume_fraction=0.5)

    predictor = EfficientXGBoostPredictorBinary(
        saved_data['model'], saved_data['apt_features'],
        saved_data['apt_encoder'], saved_data['feature_metadata']
    )

    query_vector = predictor.create_vector_from_evidence(partial_evidence)
    probs = predictor.predict_proba_vector(query_vector)

    prob_list = []
    for apt_id in apt_ids:
        if apt_id in predictor.apt_encoder:
            idx = predictor.apt_encoder[apt_id]
            prob_list.append((apt_id, probs[idx]))
        else:
            prob_list.append((apt_id, 0.0))
    prob_list.sort(key=lambda x: x[1], reverse=True)

    attacker_rank = next((i+1 for i, (aid, _) in enumerate(prob_list) if aid == attacker_apt_id), None)
    mimicked_rank = next((i+1 for i, (aid, _) in enumerate(prob_list) if aid == mimicked_apt_id), None)
    attacker_prob = probs[predictor.apt_encoder[attacker_apt_id]] if attacker_apt_id in predictor.apt_encoder else 0.0
    mimicked_prob = probs[predictor.apt_encoder[mimicked_apt_id]] if mimicked_apt_id in predictor.apt_encoder else 0.0

    return {
        'key': f"{attacker_name}→{mimicked_name}",
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_probability': attacker_prob,
        'mimicked_probability': mimicked_prob
    }


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_false_flag_attacks_binary_parallel(saved_data, top_x_values=[1, 5, 10, 20]):
    print(" EVALUATING FALSE-FLAG ATTACKS (XGBOOST MODEL PROBABILITIES)")
    print("=" * 80)
    print(" Mixed evidence: 30% attacker + 70% mimicked features (per category, up to 10)")
    print(" Ranking is by model probability, not cosine similarity.")
    print("=" * 80)

    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    _, _, _, _, _, _, apt_name_mapping = load_mappings()

    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total = len(all_combinations)
    print(f"Testing {total} false-flag combinations...")

    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data, None, None, None, None, None, None,
                apt_name_mapping, profiles, apt_ids, i)
        args_list.append(args)

    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_comb = {executor.submit(process_single_false_flag_binary, args): args for args in args_list}
        completed = 0
        for future in as_completed(future_to_comb):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank'],
                    'attacker_probability': result['attacker_probability'],
                    'mimicked_probability': result['mimicked_probability']
                }
                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total
                    remaining = est_total - elapsed
                    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                print(f"Combination failed: {e}")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")

    print("\n" + "="*80)
    print("FALSE-FLAG RESULTS (MODEL PROBABILITIES, FULL EVIDENCE)")
    print("="*80)
    for top_x in top_x_values:
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        for res in all_results.values():
            ar = res['attacker_rank']
            mr = res['mimicked_rank']
            a_ok = ar is not None and ar <= top_x
            m_ok = mr is not None and mr <= top_x
            if a_ok: attacker_detected += 1
            if m_ok: mimicked_deceived += 1
            if a_ok and m_ok: both_correct += 1
            if not a_ok and not m_ok: neither_correct += 1
        t = len(all_results)
        print(f"\nTop-{top_x}:")
        print(f"  Attacker detected: {attacker_detected}/{t} ({attacker_detected/t*100:.1f}%)")
        print(f"  Mimicked APT:     {mimicked_deceived}/{t} ({mimicked_deceived/t*100:.1f}%)")
        print(f"  Both: {both_correct}/{t} ({both_correct/t*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{t} ({neither_correct/t*100:.1f}%)")
    return all_results, apt_name_mapping, profiles, apt_ids


def evaluate_false_flag_attacks_partial_binary(saved_data, top_x_values=[1, 5, 10, 20]):
    print("\n" + "="*80)
    print("FALSE-FLAG: PARTIAL EVIDENCE (volume reduced ~50%)")
    print("="*80)
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    _, _, _, _, _, _, apt_name_mapping = load_mappings()

    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total = len(all_combinations)
    print(f"Testing {total} combinations...")

    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data, None, None, None, None, None, None,
                apt_name_mapping, profiles, apt_ids, i)
        args_list.append(args)

    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_comb = {executor.submit(process_single_false_flag_partial_binary, args): args for args in args_list}
        completed = 0
        for future in as_completed(future_to_comb):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank'],
                    'attacker_probability': result['attacker_probability'],
                    'mimicked_probability': result['mimicked_probability']
                }
                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total
                    remaining = est_total - elapsed
                    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                print(f"Combination failed: {e}")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")

    print("\nFALSE-FLAG RESULTS (PARTIAL EVIDENCE)")
    for top_x in top_x_values:
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        for res in all_results.values():
            ar = res['attacker_rank']
            mr = res['mimicked_rank']
            a_ok = ar is not None and ar <= top_x
            m_ok = mr is not None and mr <= top_x
            if a_ok: attacker_detected += 1
            if m_ok: mimicked_deceived += 1
            if a_ok and m_ok: both_correct += 1
            if not a_ok and not m_ok: neither_correct += 1
        t = len(all_results)
        print(f"\nTop-{top_x}:")
        print(f"  Attacker detected: {attacker_detected}/{t} ({attacker_detected/t*100:.1f}%)")
        print(f"  Mimicked APT:     {mimicked_deceived}/{t} ({mimicked_deceived/t*100:.1f}%)")
        print(f"  Both: {both_correct}/{t} ({both_correct/t*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{t} ({neither_correct/t*100:.1f}%)")
    return all_results


def plot_false_flag_analysis_binary(results, apt_name_mapping, profiles, apt_ids):
    """Deception heatmap using rank difference (attacker - mimicked)."""
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    deception_matrix = np.zeros((n, n))
    id_to_idx = {aid: i for i, aid in enumerate(apt_ids)}

    for key, res in results.items():
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
            attacker_id = next((aid for aid, name in apt_name_mapping.items() if name == attacker_name), None)
            mimicked_id = next((aid for aid, name in apt_name_mapping.items() if name == mimicked_name), None)
        except:
            continue
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        a_r = res['attacker_rank'] if res['attacker_rank'] is not None else n
        m_r = res['mimicked_rank'] if res['mimicked_rank'] is not None else n
        deception_matrix[i, j] = a_r - m_r

    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("All deception scores zero; nothing to plot.")
        return

    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=-max_abs, vmax=max_abs, square=True,
                cbar_kws={'label': 'Deception Score (Att Rank - Mim Rank)\n>0: Deception, <0: Detection'})
    plt.title('XGBoost False-Flag Deception (Model Probabilities, Full Evidence)')
    plt.xlabel('Mimicked APT (70% features)')
    plt.ylabel('Actual Attacker (30% features)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('false_flag_deception_matrix_xgboost_probabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" Heatmap saved.")


def main():
    print(" FALSE-FLAG EVALUATION: XGBoost Model Probabilities")
    print("=" * 70)

    saved_data = load_efficient_xgboost_model_nopca()
    if saved_data is None:
        return

    # Fixed key
    print(f"Model loaded: {saved_data['training_info']['num_apts']} APTs")
    print(f"Feature vector length: {saved_data['training_info']['feature_vector_length']}")

    results_full, apt_name_mapping, profiles, apt_ids = evaluate_false_flag_attacks_binary_parallel(
        saved_data, top_x_values=[1, 5, 10, 20]
    )
    results_partial = evaluate_false_flag_attacks_partial_binary(
        saved_data, top_x_values=[1, 5, 10, 20]
    )
    plot_false_flag_analysis_binary(results_full, apt_name_mapping, profiles, apt_ids)
    print("\n Done.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()