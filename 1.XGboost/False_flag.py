import time
import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        print("\n" + "="*50)
        print("RESOURCE USAGE REPORT")
        print("="*50)
        
        # Time
        print(f"Execution Time: {end_time - self.start_time:.2f} seconds")
        
        # Memory
        print(f"Memory Usage: {end_memory - self.start_memory:.2f} MB")
        print(f"Current Memory: {end_memory:.2f} MB")
        
        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # System-wide CPU
        sys_cpu = psutil.cpu_percent(interval=0.1)
        print(f"System CPU: {sys_cpu}%")
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU {gpu.id} ({gpu.name}):")
                print(f"  Utilization: {gpu.load*100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        except:
            print("\nGPU: Not available or not detected")

# Usage
monitor = ResourceMonitor()
monitor.start()





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
# EFFICIENT MODEL PREDICTOR FOR XGBOOST PCA - UPDATED FOR PCA MODEL
#####################################################################

class EfficientXGBoostPredictorPCA:
    """Predictor for the efficient XGBoost model with PCA"""
    def __init__(self, model, apt_features, apt_encoder, feature_metadata):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
        self.feature_metadata = feature_metadata
    
    def create_vector_from_evidence(self, evidence):
        """Create PCA vector from evidence dictionary"""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        capec = evidence.get('capec', [])
        cwe = evidence.get('cwe', [])
        cve = evidence.get('cve', [])
        pulse = evidence.get('pulse', [])
        
        # Create binary vector
        vector = np.zeros(self.feature_metadata['pulse_offset'] + 
                        len(self.feature_metadata['all_pulse']), dtype=np.float32)
        
        # Mark all features from evidence
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
        
        # Apply PCA transformation
        if 'pca' in self.feature_metadata:
            pca_model = self.feature_metadata['pca']
            vector_pca = pca_model.transform([vector])[0]
            return vector_pca
        else:
            print("⚠  WARNING: No PCA model found")
            return vector
    
    def predict_vector_similarity(self, vector1, vector2):
        """Predict similarity between two PCA vectors using cosine similarity"""
        # Simple cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, 0.0, 1.0)


def load_efficient_xgboost_model():
    """Load the efficient XGBoost model and data"""
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('6feat_efficient_apt_similarity.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print(" Loaded efficient_apt_similarity.pkl")
        return saved_data
    except FileNotFoundError:
        print(" efficient_apt_similarity.pkl not found! Please run train.py first.")
        return None


def load_apt_profiles_from_data():
    """Load APT profiles from the encoded data file"""
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
    """Load mappings once to avoid repeated file reading"""
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    
    technique_mapping = {}
    software_mapping = {} 
    capec_mapping = {}
    cwe_mapping = {}
    cve_mapping = {}
    pulse_mapping = {}
    
    for _, row in df_mapping.iterrows():
        # Technique mapping
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            if row['techniques_Origin'] not in technique_mapping:
                technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        
        # Software mapping
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            if row['software_Origin'] not in software_mapping:
                software_mapping[row['software_Origin']] = row['software_Encoded']
        
        # Capec mapping
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            if row['CAPEC_ID_Origin'] not in capec_mapping:
                capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']

        # CWE mapping
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            if row['CWE_ID_Origin'] not in cwe_mapping:
                cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']

        # CVE mapping
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            if row['CVE_ID_Origin'] not in cve_mapping:
                cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']

        # Pulse mapping
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            if row['Pulse_ID_Origin'] not in pulse_mapping:
                pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']
    
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        if encoded_id not in apt_name_mapping:
            apt_name_mapping[encoded_id] = original_name
    
    return technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping


def create_pca_vector_matrix(saved_data, profiles, apt_ids):
    """Create PCA vector matrix for all APTs"""
    print(" Creating PCA vectors for all APTs...")
    
    # Create predictor for vector creation
    predictor = EfficientXGBoostPredictorPCA(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    
    # Get pre-computed APT vectors if available
    all_apt_vectors = saved_data.get('all_apt_vectors', {})
    
    apt_vectors = []
    for apt_id in apt_ids:
        if apt_id in all_apt_vectors:
            # Use pre-computed vector
            apt_vectors.append(all_apt_vectors[apt_id])
        else:
            # Create vector from profile
            profile = profiles[apt_id]
            evidence = {
                'techniques': list(profile['techniques']),
                'software': list(profile['software']),
                'capec': list(profile['capec']),
                'cwe': list(profile['cwe']),
                'cve': list(profile['cve']),
                'pulse': list(profile['pulse'])
            }
            vector = predictor.create_vector_from_evidence(evidence)
            apt_vectors.append(vector)
    
    return np.array(apt_vectors)


def sample_mixed_features(attacker_profile, mimicked_profile, rng=None, attacker_fraction=0.3):
    """
    Sample 30% attacker / 70% mimicked feature triple for false-flag testing
    Uses probabilistic sampling per feature slot
    """
    if rng is None:
        rng = random

    atk_tech = list(attacker_profile['techniques'])
    atk_soft = list(attacker_profile['software'])
    atk_capec  = list(attacker_profile['capec'])
    atk_cwe  = list(attacker_profile['cwe'])
    atk_cve  = list(attacker_profile['cve'])
    atk_pulse  = list(attacker_profile['pulse'])

    mim_tech = list(mimicked_profile['techniques'])
    mim_soft = list(mimicked_profile['software'])
    mim_capec  = list(mimicked_profile['capec'])
    mim_cwe  = list(mimicked_profile['cwe'])
    mim_cve  = list(mimicked_profile['cve'])
    mim_pulse  = list(mimicked_profile['pulse'])

    def pick_feature(atk_list, mim_list):
        if not atk_list and not mim_list:
            return []

        use_attacker = (rng.random() < attacker_fraction)
        if use_attacker and atk_list:
            # Return all attacker features for this category
            return list(atk_list)
        elif mim_list:
            # Return all mimicked features for this category
            return list(mim_list)
        elif atk_list:
            return list(atk_list)
        else:
            return []

    # Collect all features for each category
    techniques = pick_feature(atk_tech, mim_tech)
    software  = pick_feature(atk_soft, mim_soft)
    capec     = pick_feature(atk_capec,  mim_capec)
    cwe       = pick_feature(atk_cwe,  mim_cwe)
    cve       = pick_feature(atk_cve,  mim_cve)
    pulse     = pick_feature(atk_pulse,  mim_pulse)

    return {
        'techniques': techniques,
        'software': software,
        'capec': capec,
        'cwe': cwe,
        'cve': cve,
        'pulse': pulse
    }


def process_single_false_flag_pca(args):
    """Process single false-flag with 30/70 mixed evidence using PCA vectors"""
    attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping, profiles, apt_vectors, apt_ids, seed = args
    
    rng = random.Random(seed)  # Reproducible randomness
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # 30% ATTACKER / 70% MIMICKED mixed evidence (FULL volume)
    mixed_evidence = sample_mixed_features(
        attacker_profile, mimicked_profile, rng=rng, attacker_fraction=0.3
    )
    
    # Create predictor
    predictor = EfficientXGBoostPredictorPCA(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    
    # Create PCA vector from mixed evidence
    false_flag_vector = predictor.create_vector_from_evidence(mixed_evidence)
    
    # Calculate similarities with all APT PCA vectors
    similarities = []
    for apt_vector in apt_vectors:
        similarity = predictor.predict_vector_similarity(false_flag_vector, apt_vector)
        similarities.append(similarity)
    
    # Rank results
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
        'false_flag_evidence': mixed_evidence
    }


def process_single_false_flag_partial_pca(args):
    """Process single false-flag with 30/70 mixed evidence (HALVED after mixing) using PCA"""
    attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping, profiles, apt_vectors, apt_ids, seed = args
    
    rng = random.Random(seed)
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    def process_feature_category(atk_features, mim_features, feature_type):
        """Process a single feature category: get 30% attacker, 70% mimicked, then halve both"""
        # Get attacker features (30%)
        if atk_features:
            attacker_selected = list(atk_features)
            rng.shuffle(attacker_selected)
            attacker_count = max(1, int(len(attacker_selected) * 0.3))
            attacker_final = attacker_selected[:attacker_count]
            # Halve the attacker features
            attacker_final = attacker_final[:max(1, len(attacker_final) // 2)]
        else:
            attacker_final = []
        
        # Get mimicked features (70%)
        if mim_features:
            mimicked_selected = list(mim_features)
            rng.shuffle(mimicked_selected)
            mimicked_count = max(1, int(len(mimicked_selected) * 0.7))
            mimicked_final = mimicked_selected[:mimicked_count]
            # Halve the mimicked features
            mimicked_final = mimicked_final[:max(1, len(mimicked_final) // 2)]
        else:
            mimicked_final = []
        
        # Combine halved features
        return attacker_final + mimicked_final
    
    # Create evidence with halved features for each category
    partial_evidence = {
        'techniques': process_feature_category(
            attacker_profile['techniques'], 
            mimicked_profile['techniques'],
            'techniques'
        ),
        'software': process_feature_category(
            attacker_profile['software'], 
            mimicked_profile['software'],
            'software'
        ),
        'capec': process_feature_category(
            attacker_profile['capec'], 
            mimicked_profile['capec'],
            'capec'
        ),
        'cwe': process_feature_category(
            attacker_profile['cwe'], 
            mimicked_profile['cwe'],
            'cwe'
        ),
        'cve': process_feature_category(
            attacker_profile['cve'], 
            mimicked_profile['cve'],
            'cve'
        ),
        'pulse': process_feature_category(
            attacker_profile['pulse'], 
            mimicked_profile['pulse'],
            'pulse'
        )
    }
    
    # Create predictor
    predictor = EfficientXGBoostPredictorPCA(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    
    # Create PCA vector from partial evidence
    false_flag_vector = predictor.create_vector_from_evidence(partial_evidence)
    
    # Calculate similarities
    similarities = []
    for apt_vector in apt_vectors:
        similarity = predictor.predict_vector_similarity(false_flag_vector, apt_vector)
        similarities.append(similarity)
    
    # Rank results
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
        'false_flag_evidence': partial_evidence
    }


def evaluate_false_flag_attacks_pca_parallel(saved_data, top_x_values=[1, 5, 10, 20]):
    """FULL EVIDENCE: 30/70 mixed features using PCA vectors"""
    print(" EVALUATING FALSE-FLAG ATTACKS (XGBOOST PCA - 30/70 MIX FULL)")
    print("=" * 80)
    print(" FALSE-FLAG: 30% Attacker + 70% Mimicked features per category")
    print(" Using PCA-transformed vectors with cosine similarity")
    print("=" * 80)
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Create PCA vectors for all APTs
    apt_vectors = create_pca_vector_matrix(saved_data, profiles, apt_ids)
    
    technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping = load_mappings()
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} false-flag combinations (30/70 FULL PCA)...")
    print(f"Vector dimension: {apt_vectors.shape[1]}")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, 
                software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping,
                apt_name_mapping, profiles, apt_vectors, apt_ids, i)
        args_list.append(args)
    
    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {executor.submit(process_single_false_flag_pca, args): args 
                               for args in args_list}
        
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
    
    # Top-k analysis
    print("\n" + "="*80)
    print("FALSE-FLAG RESULTS - XGBOOST PCA 30/70 FULL EVIDENCE")
    print("="*80)
    
    for top_x in top_x_values:
        print(f"\n Top-{top_x} Analysis:")
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
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results, apt_name_mapping, profiles, apt_ids


def evaluate_false_flag_attacks_partial_pca(saved_data, top_x_values=[1, 5, 10, 20]):
    """PARTIAL EVIDENCE: 30/70 mix + volume reduction using PCA"""
    print("\n" + "="*80)
    print("FALSE-FLAG: 30/70 MIX + PARTIAL VOLUME (~33% reduction, PCA)")
    print("=" * 80)
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Create PCA vectors for all APTs
    apt_vectors = create_pca_vector_matrix(saved_data, profiles, apt_ids)
    
    technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping = load_mappings()
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} partial evidence combinations (PCA)...")
    print(f"Vector dimension: {apt_vectors.shape[1]}")
    
    args_list = []
    for i, (attacker_apt_id, mimicked_apt_id) in enumerate(all_combinations):
        args = (attacker_apt_id, mimicked_apt_id, saved_data, technique_mapping, 
                software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping,
                apt_name_mapping, profiles, apt_vectors, apt_ids, i)
        args_list.append(args)
    
    all_results = {}
    start_time = time.time()
    max_workers = min(mp.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {executor.submit(process_single_false_flag_partial_pca, args): args 
                               for args in args_list}
        
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
    
    # Same analysis format
    print("\n" + "="*80)
    print("FALSE-FLAG RESULTS - XGBOOST PCA 30/70 PARTIAL EVIDENCE")
    print("="*80)
    
    for top_x in top_x_values:
        print(f"\n Top-{top_x} Analysis:")
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
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results


def plot_false_flag_analysis_pca(results, apt_name_mapping, profiles, apt_ids):
    """
    XGBoost PCA continuous deception heatmap:
    deception_score = attacker_rank - mimicked_rank
    
    > 0: mimicked ranked better (smaller rank) -> successful deception (RED)
    < 0: attacker ranked better                -> successful detection (BLUE)
    """
    print("\n CREATING XGBOOST PCA FALSE-FLAG DECEPTION HEATMAP (30/70, CONTINUOUS)...")
    
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    
    # Continuous matrix: attacker_rank - mimicked_rank
    deception_matrix = np.zeros((n, n), dtype=float)
    id_to_idx = {apt_id: i for i, apt_id in enumerate(apt_ids)}
    
    for key, result in results.items():
        attacker_rank = result['attacker_rank']
        mimicked_rank = result['mimicked_rank']
        
        # Parse key: "AttackerName→MimickedName"
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
            
            # Find APT IDs from names
            attacker_id = next((aid for aid, name in apt_name_mapping.items() if name == attacker_name), None)
            mimicked_id = next((aid for aid, name in apt_name_mapping.items() if name == mimicked_name), None)
        except:
            continue
        
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        
        # Substitute n for missing ranks
        a_r = attacker_rank if attacker_rank is not None else n
        m_r = mimicked_rank if mimicked_rank is not None else n
        
        # Positive = deception success (mimicked better rank)
        deception_matrix[i, j] = a_r - m_r
    
    # Robust color scaling
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ All deception scores zero; nothing to plot.")
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
        cmap='RdBu_r',  # Red (positive=deception) → Blue (negative=detection)
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
        'XGBoost PCA False-Flag Deception (30/70 Mixed Evidence, Continuous)\n'
        'Rows=Actual Attacker | Columns=Mimicked Target',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save AND display
    out_name = 'false_flag_deception_matrix_xgboost_pca_30_70_continuous.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved & displayed: {out_name}")


def main():
    print(" FALSE-FLAG EVALUATION: XGBOOST PCA 30/70 MIX (PCA VECTORS)")
    print("=" * 70)
    print("Each test creates PCA vectors from mixed evidence")
    print("30% attacker + 70% mimicked features per category")
    print("Similarity = cosine similarity between PCA vectors")
    print("=" * 70)
    
    saved_data = load_efficient_xgboost_model()
    if saved_data is None:
        return
    
    print(f"XGBoost PCA model loaded: {saved_data['training_info']['unique_apts']} APTs")
    print(f"PCA components: {saved_data['feature_metadata']['pca'].n_components_}")
    
    # FULL 30/70 evidence with PCA
    results_full, apt_name_mapping, profiles, apt_ids = evaluate_false_flag_attacks_pca_parallel(
        saved_data, top_x_values=[1, 5, 10, 20]
    )

    # Visualizations (FULL evidence only)
    plot_false_flag_analysis_pca(results_full, apt_name_mapping, profiles, apt_ids)
    
    # PARTIAL 30/70 + volume reduction with PCA
    results_partial = evaluate_false_flag_attacks_partial_pca(
        saved_data, top_x_values=[1, 5, 10, 20]
    )
    
    
    print("\n XGBOOST PCA 30/70 FALSE-FLAG COMPLETE!")
    print("PCA vector similarity evaluation finished")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()





monitor.stop()