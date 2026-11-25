import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD DATA AND PROFILES (same as your original TVD code)
# ============================================================

def create_apt_profiles_with_names(df_encoded, name_mapping):
    profiles = {}
    for _, row in df_encoded.iterrows():
        apt_encoded = row['APTname']
        apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
        if apt_name not in profiles:
            profiles[apt_name] = {
                'encoded_id': apt_encoded,
                'techniques': set(),
                'software': set(),
                'cve': set()
            }
        if row['techniques_str'] != -1:
            profiles[apt_name]['techniques'].add(row['techniques_str'])
        if row['software_str'] != -1:
            profiles[apt_name]['software'].add(row['software_str'])
        if row['CVE_ID'] != -1:
            profiles[apt_name]['cve'].add(row['CVE_ID'])
    return profiles

def load_apt_profiles_and_mappings():
    pwd = "../BaseData/"
    df_encoded = pd.read_csv(pwd + 'EncodedData.csv')
    df_mapping = pd.read_csv(pwd + 'comparison_Encoded_df.csv')
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)
    return apt_profiles, apt_name_dict

# ============================================================
# WEIGHTED TVD SIMILARITY (your existing function, slightly adapted)
# ============================================================

def calculate_tvd_similarity(encoded_evidence, apt_profiles, feature_weights):
    """Calculate similarity using Total Variation Distance with feature weighting"""
    similarities = []
    
    # First get all possible features to create probability distributions
    all_features = {
        'techniques': set(),
        'software': set(),
        'cve': set()
    }
    
    # Collect all unique features across all APTs
    for apt_name in apt_profiles:
        for feature_type in all_features:
            all_features[feature_type].update(apt_profiles[apt_name][feature_type])
    
    # Convert to lists for indexing
    feature_indices = {ft: list(features) for ft, features in all_features.items()}
    
    for apt_name in apt_profiles:
        apt_data = apt_profiles[apt_name]
        total_similarity = 0
        total_weight = 0
        
        for feature_type, weight in feature_weights.items():
            # Skip if evidence doesn't have this feature type
            if feature_type not in encoded_evidence:
                continue
                
            # Create probability distributions
            evidence_features = set(encoded_evidence[feature_type])
            profile_features = set(apt_data[feature_type])
            all_possible = feature_indices[feature_type]
            
            # Create binary vectors (1 if feature present, 0 otherwise)
            evidence_vec = np.array([1 if f in evidence_features else 0 for f in all_possible])
            profile_vec = np.array([1 if f in profile_features else 0 for f in all_possible])
            
            # Normalize to probability distributions
            evidence_dist = evidence_vec / evidence_vec.sum() if evidence_vec.sum() > 0 else evidence_vec
            profile_dist = profile_vec / profile_vec.sum() if profile_vec.sum() > 0 else profile_vec
            
            # Calculate TVD (half the L1 distance between distributions)
            tvd = 0.5 * np.sum(np.abs(evidence_dist - profile_dist))
            
            # Convert to similarity (1 - TVD)
            feature_similarity = (1 - tvd) * weight
            total_similarity += feature_similarity
            total_weight += weight
        
        # Normalize by total weight
        normalized_similarity = total_similarity / total_weight if total_weight > 0 else 0
        similarities.append(normalized_similarity)
    
    return similarities

# ============================================================
# UNIFIED PER‑FEATURE 30/70 MIXING (same as GNN/MIL/RF)
# ============================================================

MAX_FEATURES_PER_CAT = 10

def sample_mixed_features_unified(attacker_profile, mimicked_profile, rng=None,
                                  attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT):
    """
    For each category, sample up to `max_count` features,
    each chosen with probability attacker_fraction from attacker's set,
    otherwise from mimicked's set. Sampling without replacement.
    Returns dict with lists of encoded feature IDs.
    """
    if rng is None:
        rng = random.Random()
    mixed = {}
    categories = ['techniques', 'software', 'cve']
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

def sample_mixed_features_partial_unified(attacker_profile, mimicked_profile, rng=None,
                                          attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT,
                                          volume_fraction=0.5):
    """
    Partial evidence:
      1) Build full 30/70 mixed evidence (per-feature, up to max_count).
      2) Reduce volume per category: shuffle and keep volume_fraction (default 0.5).
    """
    full_mixed = sample_mixed_features_unified(attacker_profile, mimicked_profile, rng,
                                               attacker_fraction, max_count)
    partial_mixed = {}
    for cat, feats in full_mixed.items():
        feats = list(feats)
        rng.shuffle(feats)
        keep = max(1, int(len(feats) * volume_fraction)) if feats else 0
        partial_mixed[cat] = feats[:keep]
    return partial_mixed

# ============================================================
# WORKER FOR ONE FALSE‑FLAG PAIR
# ============================================================

def process_single_pair(args):
    attacker_name, mimicked_name, apt_profiles, feature_weights, partial, seed = args
    rng = random.Random(seed)
    attacker_profile = apt_profiles[attacker_name]
    mimicked_profile = apt_profiles[mimicked_name]
    
    if partial:
        mixed_evidence = sample_mixed_features_partial_unified(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT, volume_fraction=0.5
        )
    else:
        mixed_evidence = sample_mixed_features_unified(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT
        )
    
    # mixed_evidence is a dict of encoded lists; compute TVD similarity to all APTs
    all_apts = list(apt_profiles.keys())
    similarities = calculate_tvd_similarity(mixed_evidence, apt_profiles, feature_weights)
    
    # Rank
    ranked = sorted(zip(all_apts, similarities), key=lambda x: x[1], reverse=True)
    ranked_names = [apt for apt, _ in ranked]
    
    attacker_rank = ranked_names.index(attacker_name) + 1 if attacker_name in ranked_names else None
    mimicked_rank = ranked_names.index(mimicked_name) + 1 if mimicked_name in ranked_names else None
    
    key = f"{attacker_name}→{mimicked_name}"
    if partial:
        key += "_partial"
    
    return {
        'key': key,
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
    }

# ============================================================
# PARALLEL EVALUATION (FULL / PARTIAL)
# ============================================================

def evaluate_false_flag_tvd(apt_profiles, apt_name_dict, partial=False,
                            top_x_values=[1,5,10,20]):
    scenario = "PARTIAL (30/70 + 50% volume reduction)" if partial else "FULL (30/70 per‑feature mix)"
    print("\n" + "="*80)
    print(f" TVD FALSE-FLAG: {scenario}")
    print("="*80)
    
    apt_names = list(apt_profiles.keys())
    all_combinations = [(a, m) for a in apt_names for m in apt_names if a != m]
    total_combinations = len(all_combinations)
    print(f"Testing {total_combinations:,} false‑flag combinations...")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    feature_weights = {
        'techniques': 6/6, 'software': 5/6, 'cve': 2/6
    }
    
    args_list = []
    for idx, (attacker, mimicked) in enumerate(all_combinations):
        args_list.append((attacker, mimicked, apt_profiles, feature_weights, partial, idx))
    
    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pair, args): i for i, args in enumerate(args_list)}
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank']
                }
                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total_combinations
                    remaining = max(0, est_total - elapsed)
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                print(f"Worker error: {e}")
    
    total_time = time.time() - start_time
    print(f"\nCompleted {len(all_results):,} evaluations in {total_time/60:.1f} minutes")
    
    # Top‑k analysis
    print("\n" + "="*80)
    print(f"TVD RESULTS - {scenario}")
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
        total_cases = len(all_results)
        print(f"\n Top-{top_x}:")
        print(f"  Attacker detected:   {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked deceived:   {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both:                {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither:             {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results

# ============================================================
# DECEPTION HEATMAP (CONTINUOUS)
# ============================================================

def plot_deception_heatmap(results, apt_names, title_suffix="Full"):
    print("\n CREATING TVD DECEPTION HEATMAP...")
    n = len(apt_names)
    name_to_idx = {name: i for i, name in enumerate(apt_names)}
    deception_matrix = np.zeros((n, n), dtype=float)
    
    for key, res in results.items():
        if "_partial" in key:
            continue  # only plot full results (or we could pass a flag)
        try:
            attacker, mimicked = key.split("→")
            attacker = attacker.strip()
            mimicked = mimicked.strip()
        except:
            continue
        i = name_to_idx.get(attacker)
        j = name_to_idx.get(mimicked)
        if i is None or j is None:
            continue
        ar = res['attacker_rank'] if res['attacker_rank'] is not None else n
        mr = res['mimicked_rank'] if res['mimicked_rank'] is not None else n
        deception_matrix[i, j] = ar - mr   # positive = deception (mimicked ranks better)
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ All deception scores zero; nothing to plot.")
        return
    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    vmin, vmax = -max_abs, max_abs
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, square=True,
                cbar_kws={'label': 'Deception Score (Attacker Rank - Mimicked Rank)\n>0: Deception (RED), <0: Detection (BLUE)'})
    plt.title(f'TVD False‑Flag Deception – {title_suffix}\n30% Attacker + 70% Mimicked features per category',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_name = f'false_flag_deception_matrix_tvd_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_name}")

# ============================================================
# MAIN
# ============================================================

def main():
    print(" TVD FALSE-FLAG EVALUATION (UNIFIED 30/70 PER‑FEATURE MIX)")
    print("=" * 70)
    print("Uses weighted Total Variation Distance (1 - TVD) as similarity")
    print("30% attacker + 70% mimicked features per category (up to 10 features each)")
    print("Partial version additionally reduces volume by ~50%")
    print("=" * 70)
    
    apt_profiles, apt_name_dict = load_apt_profiles_and_mappings()
    apt_names = list(apt_profiles.keys())
    print(f"Loaded {len(apt_names)} APT groups")
    
    # FULL (no volume reduction)
    full_results = evaluate_false_flag_tvd(apt_profiles, apt_name_dict, partial=False)
    
    # PARTIAL (volume reduction)
    partial_results = evaluate_false_flag_tvd(apt_profiles, apt_name_dict, partial=True)
    
    # Heatmap for full results (you can also plot partial if desired)
    plot_deception_heatmap(full_results, apt_names, title_suffix="Full")
    
    print("\n TVD FALSE‑FLAG COMPLETE!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()