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

#####################################################################
# EFFICIENT MODEL PREDICTOR FOR XGBOOST - PURE MODEL OUTPUT (6D)
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model - uses ONLY model predict_proba"""
    def __init__(self, model, apt_features, apt_encoder):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
    
    def predict_similarity(self, features1, features2):
        """Predict similarity using ONLY XGBoost model probabilities - NO custom heuristics"""
        # keep same guard: empty main triple => 0
        if features1[0] == -1 and features1[1] == -1 and features1[2] == -1:
            return 0.0
        if features2[0] == -1 and features2[1] == -1 and features2[2] == -1:
            return 0.0
            
        # Get probability distributions DIRECTLY from XGBoost model
        prob1 = self.model.predict_proba([features1])[0]
        prob2 = self.model.predict_proba([features2])[0]
        
        # Pure cosine similarity of XGBoost probability distributions
        denom = (np.linalg.norm(prob1) * np.linalg.norm(prob2))
        if denom == 0:
            return 0.0
        cosine_sim = np.dot(prob1, prob2) / denom
        
        return np.clip(cosine_sim, 0.0, 1.0)


def load_efficient_xgboost_model():
    """Load the efficient XGBoost model and data"""
    print("🔍 LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('efficient_apt_similarity.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print(" Loaded efficient_apt_similarity.pkl")
        return saved_data
    except FileNotFoundError:
        print(" efficient_apt_similarity.pkl not found! Please run train.py first.")
        return None


def load_apt_profiles_from_data():
    """Load APT profiles from the encoded data file (6D sets)"""
    print("📊 LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software':   set(apt_data['software_str'].dropna().unique()),
            'cve':        set(apt_data['CVE_ID'].dropna().unique()),
            'capec':      set(apt_data['CAPEC_ID'].dropna().unique()),
            'cwe':        set(apt_data['CWE_ID'].dropna().unique()),
            'pulse':      set(apt_data['Pulse_ID'].dropna().unique()),
            'total_features': len(apt_data)
        }
    
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles


def load_mappings():
    """Load mappings once to avoid repeated file reading (6D)"""
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    
    technique_mapping = {}
    software_mapping  = {} 
    cve_mapping       = {}
    capec_mapping     = {}
    cwe_mapping       = {}
    pulse_mapping     = {}
    
    for _, row in df_mapping.iterrows():
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            if row['techniques_Origin'] not in technique_mapping:
                technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            if row['software_Origin'] not in software_mapping:
                software_mapping[row['software_Origin']] = row['software_Encoded']
        
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            if row['CVE_ID_Origin'] not in cve_mapping:
                cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
        
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            if row['CAPEC_ID_Origin'] not in capec_mapping:
                capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']
        
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            if row['CWE_ID_Origin'] not in cwe_mapping:
                cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']
        
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            if row['Pulse_ID_Origin'] not in pulse_mapping:
                pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']
    
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        if encoded_id not in apt_name_mapping:
            apt_name_mapping[encoded_id] = original_name
    
    return (technique_mapping, software_mapping, cve_mapping,
            capec_mapping, cwe_mapping, pulse_mapping, apt_name_mapping)


def create_feature_matrix(profiles, apt_ids):
    """Create feature matrix for all APTs using representative 6D encoded features"""
    df = pd.read_csv("../BaseData/EncodedData.csv")
    X_apts = []
    for apt_id in apt_ids:
        apt_data = df[df['APTname'] == apt_id]
        # representative max per field
        tech = apt_data['techniques_str'].max() if 'techniques_str' in apt_data.columns else -1
        soft = apt_data['software_str'].max()   if 'software_str'   in apt_data.columns else -1
        cve  = apt_data['CVE_ID'].max()         if 'CVE_ID'         in apt_data.columns else -1
        cap  = apt_data['CAPEC_ID'].max()       if 'CAPEC_ID'       in apt_data.columns else -1
        cwe  = apt_data['CWE_ID'].max()         if 'CWE_ID'         in apt_data.columns else -1
        pul  = apt_data['Pulse_ID'].max()       if 'Pulse_ID'       in apt_data.columns else -1
        
        apt_features = np.array([tech, soft, cve, cap, cwe, pul])
        X_apts.append(apt_features)
    
    return np.array(X_apts)


def sample_mixed_features(attacker_profile, mimicked_profile, rng=None, attacker_fraction=0.3):
    """
    Sample 30% attacker / 70% mimicked feature *triple* for false-flag testing
    ORIGINAL 3D VERSION (tech/software/cve) – kept for compatibility
    """
    if rng is None:
        rng = random

    atk_tech = list(attacker_profile['techniques'])
    atk_soft = list(attacker_profile['software'])
    atk_cve  = list(attacker_profile['cve'])

    mim_tech = list(mimicked_profile['techniques'])
    mim_soft = list(mimicked_profile['software'])
    mim_cve  = list(mimicked_profile['cve'])

    def pick_feature(atk_list, mim_list):
        if not atk_list and not mim_list:
            return -1

        use_attacker = (rng.random() < attacker_fraction)
        if use_attacker and atk_list:
            return rng.choice(atk_list)
        elif mim_list:
            return rng.choice(mim_list)
        elif atk_list:
            return rng.choice(atk_list)
        else:
            return -1

    technique = pick_feature(atk_tech, mim_tech)
    software  = pick_feature(atk_soft, mim_soft)
    cve       = pick_feature(atk_cve,  mim_cve)

    return np.array([technique, software, cve])


def sample_mixed_features_6d(attacker_profile, mimicked_profile, rng=None, attacker_fraction=0.3):
    """
    NEW 6D VERSION: 30% attacker / 70% mimicked for
    [techniques, software, cve, capec, cwe, pulse]
    """
    if rng is None:
        rng = random
    
    def pick(atk_set, mim_set):
        atk_list = list(atk_set)
        mim_list = list(mim_set)
        if not atk_list and not mim_list:
            return -1
        use_attacker = (rng.random() < attacker_fraction)
        if use_attacker and atk_list:
            return rng.choice(atk_list)
        elif mim_list:
            return rng.choice(mim_list)
        elif atk_list:
            return rng.choice(atk_list)
        else:
            return -1
    
    technique = pick(attacker_profile['techniques'], mimicked_profile['techniques'])
    software  = pick(attacker_profile['software'],   mimicked_profile['software'])
    cve       = pick(attacker_profile['cve'],        mimicked_profile['cve'])
    capec     = pick(attacker_profile['capec'],      mimicked_profile['capec'])
    cwe       = pick(attacker_profile['cwe'],        mimicked_profile['cwe'])
    pulse     = pick(attacker_profile['pulse'],      mimicked_profile['pulse'])
    
    return np.array([technique, software, cve, capec, cwe, pulse])


def process_single_false_flag_efficient(args):
    """Process single false-flag with 30/70 mixed evidence (FULL, 6D)"""
    (attacker_apt_id, mimicked_apt_id, saved_data,
     technique_mapping, software_mapping, cve_mapping,
     capec_mapping, cwe_mapping, pulse_mapping,
     apt_name_mapping, profiles, X_apts, apt_ids, seed) = args
    
    rng = random.Random(seed)  # Reproducible randomness
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # 30% ATTACKER / 70% MIMICKED 6D mixed evidence (FULL volume)
    false_flag_features = sample_mixed_features_6d(
        attacker_profile, mimicked_profile, rng=rng, attacker_fraction=0.3
    )
    
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder']
    )
    
    similarities = []
    for apt_features in X_apts:
        similarity = predictor.predict_similarity(false_flag_features, apt_features)
        similarities.append(similarity)
    
    ranked_results = {}
    sorted_indices = np.argsort(similarities)[::-1]
    
    for idx in sorted_indices:
        apt_id = apt_ids[idx]
        ranked_results[apt_id] = similarities[idx]
    
    ranked_apt_ids = list(ranked_results.keys())
    attacker_rank = ranked_apt_ids.index(attacker_apt_id) + 1 if attacker_apt_id in ranked_apt_ids else None
    mimicked_rank = ranked_apt_ids.index(mimicked_apt_id) + 1 if mimicked_apt_id in ranked_apt_ids else None
    
    return {
        'key': f"{attacker_name}→{mimicked_name}",
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_similarity': ranked_results.get(attacker_apt_id),
        'mimicked_similarity': ranked_results.get(mimicked_apt_id),
        'false_flag_features': false_flag_features
    }


def process_single_false_flag_partial_evidence(args):
    """Process single false-flag with 30/70 mixed evidence (PARTIAL volume, 6D)"""
    (attacker_apt_id, mimicked_apt_id, saved_data,
     technique_mapping, software_mapping, cve_mapping,
     capec_mapping, cwe_mapping, pulse_mapping,
     apt_name_mapping, profiles, X_apts, apt_ids, seed) = args
    
    rng = random.Random(seed)
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # Start with 30/70 6D mix
    full_mixed = sample_mixed_features_6d(
        attacker_profile, mimicked_profile, rng=rng, attacker_fraction=0.3
    )
    
    # PARTIAL: randomly drop ~33% evidence (2 of 6 slots)
    mask = [0, 1, 2, 3, 4, 5]
    rng.shuffle(mask)
    to_drop = mask[:2]  # 2 of 6 ≈ 33%
    
    partial = full_mixed.copy()
    for idx in to_drop:
        partial[idx] = -1
    
    false_flag_features = partial
    
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder']
    )
    
    similarities = []
    for apt_features in X_apts:
        similarity = predictor.predict_similarity(false_flag_features, apt_features)
        similarities.append(similarity)
    
    ranked_results = {}
    sorted_indices = np.argsort(similarities)[::-1]
    
    for idx in sorted_indices:
        apt_id = apt_ids[idx]
        ranked_results[apt_id] = similarities[idx]
    
    ranked_apt_ids = list(ranked_results.keys())
    attacker_rank = ranked_apt_ids.index(attacker_apt_id) + 1 if attacker_apt_id in ranked_apt_ids else None
    mimicked_rank = ranked_apt_ids.index(mimicked_apt_id) + 1 if mimicked_apt_id in ranked_apt_ids else None
    
    return {
        'key': f"{attacker_name}→{mimicked_name}",
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_similarity': ranked_results.get(attacker_apt_id),
        'mimicked_similarity': ranked_results.get(mimicked_apt_id),
        'false_flag_features': false_flag_features
    }


def evaluate_false_flag_attacks_efficient_parallel(saved_data, top_x_values=[1, 3, 5, 10, 20]):
    """FULL EVIDENCE: 30/70 mixed features (6D)"""
    print(" EVALUATING FALSE-FLAG ATTACKS (EFFICIENT XGBOOST - 30/70 MIX FULL, 6D)")
    print("=" * 80)
    print(" FALSE-FLAG: 30% Attacker + 70% Mimicked features per slot (T/S/CVE/CAPEC/CWE/Pulse)")
    print("=" * 80)
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    X_apts = create_feature_matrix(profiles, apt_ids)
    
    (technique_mapping, software_mapping, cve_mapping,
     capec_mapping, cwe_mapping, pulse_mapping, apt_name_mapping) = load_mappings()
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} false-flag combinations (30/70 FULL, 6D)...")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data,
                technique_mapping, software_mapping, cve_mapping,
                capec_mapping, cwe_mapping, pulse_mapping,
                apt_name_mapping, profiles, X_apts, apt_ids, i)
        args_list.append(args)
    
    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(process_single_false_flag_efficient, args): args 
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(future_to_combination):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank'],
                    'attacker_similarity': result['attacker_similarity'],
                    'mimicked_similarity': result['mimicked_similarity']
                }
                completed += 1
                if completed % 100 == 0:
                    elapsed_time = time.time() - start_time
                    estimated_total = (elapsed_time / completed) * total_combinations
                    remaining = estimated_total - elapsed_time
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"Combination failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\nCompleted {total_combinations} evaluations in {total_time/60:.1f} minutes")
    
    print("\n" + "="*80)
    print("FALSE-FLAG RESULTS - XGBOOST 30/70 FULL EVIDENCE (6D)")
    print("="*80)
    
    for top_x in top_x_values:
        print(f"\n📊 Top-{top_x} Analysis:")
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        
        for result in all_results.values():
            ar = result['attacker_rank']
            mr = result['mimicked_rank']
            attacker_in_top = ar is not None and ar <= top_x
            mimicked_in_top = mr is not None and mr <= top_x
            
            if attacker_in_top: attacker_detected += 1
            if mimicked_in_top: mimicked_deceived += 1
            if attacker_in_top and mimicked_in_top: both_correct += 1
            if not attacker_in_top and not mimicked_in_top: neither_correct += 1
        
        total_cases = len(all_results)
        print(f"  Actual attacker Top-{top_x}: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT Top-{top_x}: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both Top-{top_x}: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither Top-{top_x}: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"  🎭 Deception: {mimicked_deceived/total_cases*100:.1f}% | 🔍 Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖️ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results, apt_name_mapping, profiles, apt_ids


def evaluate_false_flag_attacks_partial_evidence(saved_data, top_x_values=[1, 3, 5, 10, 20]):
    """PARTIAL EVIDENCE: 30/70 mix + volume reduction (6D)"""
    print("\n" + "="*80)
    print("FALSE-FLAG: 30/70 MIX + PARTIAL VOLUME (~33% reduction, 6D)")
    print("=" * 80)
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    X_apts = create_feature_matrix(profiles, apt_ids)
    
    (technique_mapping, software_mapping, cve_mapping,
     capec_mapping, cwe_mapping, pulse_mapping, apt_name_mapping) = load_mappings()
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} partial evidence combinations (6D)...")
    
    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data,
                technique_mapping, software_mapping, cve_mapping,
                capec_mapping, cwe_mapping, pulse_mapping,
                apt_name_mapping, profiles, X_apts, apt_ids, i)
        args_list.append(args)
    
    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(process_single_false_flag_partial_evidence, args): args 
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(future_to_combination):
            try:
                result = future.result()
                all_results[result['key']] = {
                    'attacker_rank': result['attacker_rank'],
                    'mimicked_rank': result['mimicked_rank'],
                    'attacker_similarity': result['attacker_similarity'],
                    'mimicked_similarity': result['mimicked_similarity']
                }
                completed += 1
                if completed % 100 == 0:
                    elapsed_time = time.time() - start_time
                    estimated_total = (elapsed_time / completed) * total_combinations
                    remaining = estimated_total - elapsed_time
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"Combination failed: {e}")
    
    print("\n" + "="*80)
    print("FALSE-FLAG RESULTS - XGBOOST 30/70 PARTIAL EVIDENCE (6D)")
    print("="*80)
    
    for top_x in top_x_values:
        print(f"\n📊 Top-{top_x} Analysis:")
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        
        for result in all_results.values():
            ar = result['attacker_rank']
            mr = result['mimicked_rank']
            attacker_in_top = ar is not None and ar <= top_x
            mimicked_in_top = mr is not None and mr <= top_x
            
            if attacker_in_top: attacker_detected += 1
            if mimicked_in_top: mimicked_deceived += 1
            if attacker_in_top and mimicked_in_top: both_correct += 1
            if not attacker_in_top and not mimicked_in_top: neither_correct += 1
        
        total_cases = len(all_results)
        print(f"  Actual attacker Top-{top_x}: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT Top-{top_x}: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both Top-{top_x}: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither Top-{top_x}: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"  🎭 Deception: {mimicked_deceived/total_cases*100:.1f}% | 🔍 Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖️ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results


def plot_false_flag_analysis_efficient(results, apt_name_mapping, profiles, apt_ids):
    """
    XGBoost continuous deception heatmap:
    deception_score = attacker_rank - mimicked_rank
    
    > 0: mimicked ranked better (smaller rank) -> successful deception (RED)
    < 0: attacker ranked better                -> successful detection (BLUE)
    """
    print("\n📈 CREATING XGBOOST FALSE-FLAG DECEPTION HEATMAP (30/70, CONTINUOUS, 6D)...")
    
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    
    deception_matrix = np.zeros((n, n), dtype=float)
    id_to_idx = {apt_id: i for i, apt_id in enumerate(apt_ids)}
    
    for key, result in results.items():
        attacker_rank = result['attacker_rank']
        mimicked_rank = result['mimicked_rank']
        
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
        
        a_r = attacker_rank if attacker_rank is not None else n
        m_r = mimicked_rank if mimicked_rank is not None else n
        
        deception_matrix[i, j] = a_r - m_r
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠️ All deception scores zero; nothing to plot.")
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
                     '>0: Deception (RED), <0: Detection (BLUE)'
        }
    )
    
    plt.title(
        'XGBoost False-Flag Deception (30/70 Mixed Evidence, 6D)\n'
        'Rows=Actual Attacker | Columns=Mimicked Target',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    out_name = 'false_flag_deception_matrix_xgboost_30_70_6d_continuous.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved & displayed: {out_name}")


def main():
    print(" FALSE-FLAG EVALUATION: XGBOOST 30/70 MIX (PURE MODEL OUTPUT, 6D)")
    print("=" * 70)
    print("Each test uses XGBoost predict_proba cosine similarity ONLY")
    print("30% attacker + 70% mimicked features per evidence vector (T/S/CVE/CAPEC/CWE/Pulse)")
    print("=" * 70)
    
    saved_data = load_efficient_xgboost_model()
    if saved_data is None:
        return
    
    print(f"XGBoost model loaded: {saved_data['training_info']['unique_apts']} APTs")
    
    # FULL 30/70 evidence (6D)
    results_full, apt_name_mapping, profiles, apt_ids = evaluate_false_flag_attacks_efficient_parallel(
        saved_data, top_x_values=[1, 5, 10, 20]
    )

    # Visualizations (FULL evidence only)
    plot_false_flag_analysis_efficient(results_full, apt_name_mapping, profiles, apt_ids)

    print("\n✅ XGBOOST 30/70 FALSE-FLAG COMPLETE! (6D, pure model output)")
    
    # PARTIAL 30/70 + volume reduction (6D)
    results_partial = evaluate_false_flag_attacks_partial_evidence(
        saved_data, top_x_values=[1, 5, 10, 20]
    )



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
