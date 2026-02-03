import os
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "apt_similarity_model.joblib"
TOP_X_VALUES = [1, 5, 10, 20]
MAX_WORKERS = min(mp.cpu_count(), 8)


# ============================================================
# LOADING RF MODEL + DATA (YOUR JOBLIB)
# ============================================================

def load_efficient_rf_model(path=MODEL_PATH):
    """
    Load your RF similarity joblib model.
    Expects keys from your training script:
      - 'rf_model'
      - 'binarizers'
      - 'all_features'
      - 'apt_profiles' (APT name -> {encoded_id, techniques, software, cve})
      - 'apt_names'
      - 'apt_name_dict'
      - 'X'
      - 'feature_types' (['techniques','software','cve'])
      - 'df_mapping'
    """
    print(" LOADING EFFICIENT RF MODEL...")
    if not os.path.exists(path):
        print(f" {path} not found! Please run your RF training script first.")
        return None
    
    model_data = joblib.load(path)
    
    print(f" Loaded {path}")
    print(f"   APTs: {len(model_data['apt_names'])}")
    print(f"   Feature types: {model_data['feature_types']}")
    print(f"   Feature matrix shape: {model_data['X'].shape}")
    print(f"   Available CPU cores: {mp.cpu_count()}")
    
    return model_data


# ============================================================
# RF PREDICTOR USING PREDICT_PROBA + COSINE
# ============================================================

class EfficientRFPredictor:
    """
    Predict similarity between a false-flag evidence vector and all APTs,
    using cosine similarity of RF predict_proba outputs.
    """
    def __init__(self, model_data):
        self.model_data = model_data
        self.model = model_data['rf_model']
        self.X = model_data['X']
        self.apt_names = model_data['apt_names']
    
    def predict_similarity(self, false_flag_features):
        prob_false = self.model.predict_proba([false_flag_features])[0]
        
        similarities = []
        for apt_features in self.X:
            prob_apt = self.model.predict_proba([apt_features])[0]
            num = np.dot(prob_false, prob_apt)
            den = np.linalg.norm(prob_false) * np.linalg.norm(prob_apt) + 1e-8
            cosine_sim = num / den
            similarities.append(float(np.clip(cosine_sim, 0.0, 1.0)))
        
        return np.array(similarities, dtype=float)


# ============================================================
# 30/70 FALSE-FLAG EVIDENCE (FULL + PARTIAL) – GNN-STYLE
# ============================================================

def create_false_flag_features_full(attacker_profile, mimicked_profile, model_data,
                                    rng=None, attacker_fraction=0.3, max_count=10):
    """
    FULL evidence:
      For each feature type, sample up to max_count features with 30% attacker / 70% mimicked ratio,
      then binarize them with the corresponding MultiLabelBinarizer.
    Mirrors GNN's sample_mixed_evidence.
    """
    if rng is None:
        rng = random
    
    binarizers = model_data['binarizers']
    feature_types = model_data['feature_types']  # ['techniques','software','cve']
    
    mixed_features = {}
    
    def sample_mixed_features(atk_feats, mim_feats, max_count):
        atk_list = list(atk_feats)
        mim_list = list(mim_feats)
        if not atk_list and not mim_list:
            return []
        
        total_needed = min(max_count, len(atk_list) + len(mim_list))
        result = []
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_list:
                choice = rng.choice(atk_list)
                result.append(choice)
                atk_list.remove(choice)
            elif mim_list:
                choice = rng.choice(mim_list)
                result.append(choice)
                mim_list.remove(choice)
            else:
                break
        return result[:max_count]
    
    for ft in feature_types:
        atk_feats = attacker_profile.get(ft, [])
        mim_feats = mimicked_profile.get(ft, [])
        mixed_features[ft] = sample_mixed_features(atk_feats, mim_feats, max_count)
    
    feature_vector = []
    for ft in feature_types:
        feats = mixed_features[ft]
        binary_vec = binarizers[ft].transform([feats]).flatten()
        feature_vector.extend(binary_vec)
    
    return np.array(feature_vector, dtype=float)


def create_false_flag_features_partial(attacker_profile, mimicked_profile, model_data,
                                       rng=None, attacker_fraction=0.3, max_count=10,
                                       volume_fraction=0.5):
    """
    PARTIAL evidence:
      1) Build FULL 30/70 mixed evidence per feature type (as above, list form).
      2) Shuffle and keep ~50% per feature type.
      3) Binarize with the same binarizers.
    Matches GNN's create_partial_mixed_evidence semantics.
    """
    if rng is None:
        rng = random
    
    binarizers = model_data['binarizers']
    feature_types = model_data['feature_types']
    
    # Step 1: full 30/70 lists
    full_mixed = {}
    
    def sample_mixed_features(atk_feats, mim_feats, max_count):
        atk_list = list(atk_feats)
        mim_list = list(mim_feats)
        if not atk_list and not mim_list:
            return []
        total_needed = min(max_count, len(atk_list) + len(mim_list))
        result = []
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_list:
                choice = rng.choice(atk_list)
                result.append(choice)
                atk_list.remove(choice)
            elif mim_list:
                choice = rng.choice(mim_list)
                result.append(choice)
                mim_list.remove(choice)
            else:
                break
        return result[:max_count]
    
    for ft in feature_types:
        atk_feats = attacker_profile.get(ft, [])
        mim_feats = mimicked_profile.get(ft, [])
        full_mixed[ft] = sample_mixed_features(atk_feats, mim_feats, max_count)
    
    # Step 2: volume reduction per feature type
    partial_mixed = {}
    for ft in feature_types:
        feats = list(full_mixed[ft])
        if not feats:
            partial_mixed[ft] = []
            continue
        rng.shuffle(feats)
        keep_count = max(1, int(len(feats) * volume_fraction))
        partial_mixed[ft] = feats[:keep_count]
    
    # Step 3: binarize and concat
    feature_vector = []
    for ft in feature_types:
        feats = partial_mixed[ft]
        binary_vec = binarizers[ft].transform([feats]).flatten()
        feature_vector.extend(binary_vec)
    
    return np.array(feature_vector, dtype=float)


# ============================================================
# SINGLE FALSE-FLAG EVALUATION (FULL / PARTIAL)
# ============================================================

def process_single_false_flag_rf(args):
    attacker_apt_id, mimicked_apt_id, model_data, apt_ids, partial, seed = args
    rng = random.Random(seed)
    
    # Map encoded IDs back to APT names using apt_profiles
    attacker_name = next(
        (name for name in model_data['apt_names']
         if model_data['apt_profiles'][name]['encoded_id'] == attacker_apt_id),
        f"APT{attacker_apt_id}"
    )
    mimicked_name = next(
        (name for name in model_data['apt_names']
         if model_data['apt_profiles'][name]['encoded_id'] == mimicked_apt_id),
        f"APT{mimicked_apt_id}"
    )
    
    attacker_profile = model_data['apt_profiles'][attacker_name]
    mimicked_profile = model_data['apt_profiles'][mimicked_name]
    
    # 30/70 mixed evidence
    if partial:
        false_flag_features = create_false_flag_features_partial(
            attacker_profile, mimicked_profile, model_data,
            rng=rng, attacker_fraction=0.3, max_count=10, volume_fraction=0.5
        )
    else:
        false_flag_features = create_false_flag_features_full(
            attacker_profile, mimicked_profile, model_data,
            rng=rng, attacker_fraction=0.3, max_count=10
        )
    
    predictor = EfficientRFPredictor(model_data)
    similarities = predictor.predict_similarity(false_flag_features)
    
    # Rank all APTs
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_results = {model_data['apt_names'][idx]: float(similarities[idx])
                      for idx in sorted_indices}
    ranked_names = list(ranked_results.keys())
    
    attacker_rank = ranked_names.index(attacker_name) + 1 if attacker_name in ranked_names else None
    mimicked_rank = ranked_names.index(mimicked_name) + 1 if mimicked_name in ranked_names else None
    
    key = f"{attacker_name}→{mimicked_name}"
    if partial:
        key += "_partial"
    
    return {
        'key': key,
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_similarity': ranked_results.get(attacker_name),
        'mimicked_similarity': ranked_results.get(mimicked_name)
    }


# ============================================================
# PARALLEL EVALUATION WITH ETA 
# ============================================================

def evaluate_false_flag_attacks_rf_parallel(model_data, top_x_values=TOP_X_VALUES, partial=False):
    scenario = "PARTIAL (50% volume from 30/70 mix)" if partial else "FULL (30/70 mixed evidence)"
    print("=" * 80)
    print(" FALSE-FLAG: 30% Attacker + 70% Mimicked features across ALL categories")
    print(f"(RF {scenario})")
    print("=" * 80)
    
    apt_ids = [model_data['apt_profiles'][name]['encoded_id'] for name in model_data['apt_names']]
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} false-flag combinations (30/70 mix)...")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    args_list = []
    for i, (attacker_id, mimicked_id) in enumerate(all_combinations):
        args_list.append((attacker_id, mimicked_id, model_data, apt_ids, partial, i))
    
    all_results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_false_flag_rf, args): i
                   for i, args in enumerate(args_list)}
        
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank'],
                    'attacker_similarity': result['attacker_similarity'],
                    'mimicked_similarity': result['mimicked_similarity']
                }
                completed += 1
                
                if completed % 100 == 0 or completed == total_combinations:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total_combinations if completed > 0 else 0
                    remaining = max(0, est_total - elapsed)
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"Worker error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n Completed {len(all_results)} evaluations in {total_time/60:.1f} minutes")
    
    # Top-k analysis (same as GNN/MIL)
    print("\n" + "=" * 80)
    print(f"FALSE-FLAG RESULTS - RF 30/70 MIX ({'PARTIAL' if partial else 'FULL'})")
    print("=" * 80)
    
    for top_x in top_x_values:
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        
        for res in all_results.values():
            ar = res['attacker_rank']
            mr = res['mimicked_rank']
            attacker_in_top = ar is not None and ar <= top_x
            mimicked_in_top = mr is not None and mr <= top_x
            
            if attacker_in_top: attacker_detected += 1
            if mimicked_in_top: mimicked_deceived += 1
            if attacker_in_top and mimicked_in_top: both_correct += 1
            if not attacker_in_top and not mimicked_in_top: neither_correct += 1
        
        total_cases = len(all_results)
        print(f"\n Top-{top_x} Analysis:")
        print(f"  Actual attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
    
    return all_results, model_data['apt_name_dict'], model_data['apt_profiles'], apt_ids


# ============================================================
# CONTINUOUS DECEPTION HEATMAP – MATCHING GNN/MIL
# ============================================================

def plot_false_flag_deception_rf(results, apt_name_mapping, apt_ids,
                                 output_path="false_flag_deception_matrix_rf_30_70_continuous.png"):
    """
    RF deception heatmap:
      deception_score = attacker_rank - mimicked_rank
      >0: mimicked ranked better (deception success)
      <0: attacker ranked better (detection success)
    """
    print("\n CREATING FALSE-FLAG DECEPTION HEATMAP (RF 30/70, CONTINUOUS)...")
    
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    id_to_idx = {aid: i for i, aid in enumerate(apt_ids)}
    
    deception_matrix = np.zeros((n, n), dtype=float)
    
    for key, res in results.items():
        if "_partial" in key.lower():
            continue
        
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
        except Exception:
            continue
        
        attacker_id = next((aid for aid, nm in apt_name_mapping.items() if nm == attacker_name), None)
        mimicked_id = next((aid for aid, nm in apt_name_mapping.items() if nm == mimicked_name), None)
        if attacker_id is None or mimicked_id is None:
            continue
        
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        
        ar = res["attacker_rank"]
        mr = res["mimicked_rank"]
        a_r = ar if ar is not None else n
        m_r = mr if mr is not None else n
        
        deception_matrix[i, j] = a_r - m_r  # >0 => mimicked better
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ All deception scores are zero; nothing to plot.")
        return
    
    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    vmin, vmax = -max_abs, max_abs
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        deception_matrix,
        xticklabels=apt_names,
        yticklabels=apt_names,
        cmap='RdBu_r',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={
            'label': 'Deception Score (Attacker Rank - Mimicked Rank)\n'
                     '>0: Deception success, <0: Detection success'
        }
    )
    
    plt.title(
        'RF False-Flag Deception (30/70 Mixed Evidence, Continuous Score)\n'
        'Rows=Actual Attacker | Columns=Mimicked APT',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {output_path}")


# ============================================================
# MAIN – MATCHING GNN/MIL ENTRYPOINT
# ============================================================

def main():
    print(" RF FALSE-FLAG EVALUATION: 30/70 MIXED EVIDENCE")
    print("=" * 70)
    print("Random Forest + cosine similarity on predict_proba")
    print("30% attacker + 70% mimicked features ACROSS ALL CATEGORIES")
    print("=" * 70)
    
    model_data = load_efficient_rf_model(MODEL_PATH)
    if model_data is None:
        return
    
    # FULL 30/70
    full_results, apt_name_mapping, profiles, apt_ids = evaluate_false_flag_attacks_rf_parallel(
        model_data, top_x_values=TOP_X_VALUES, partial=False
    )
    plot_false_flag_deception_rf(full_results, apt_name_mapping, apt_ids)
    
    # PARTIAL 30/70 + 50% volume reduction
    partial_results, _, _, _ = evaluate_false_flag_attacks_rf_parallel(
        model_data, top_x_values=TOP_X_VALUES, partial=True
    )
    
    print("\n RF 30/70 FALSE-FLAG EVALUATION COMPLETE!")
    print("Saved: false_flag_deception_matrix_rf_30_70_continuous.png")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
