import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import islice


#####################################################################
# EFFICIENT XGBOOST PREDICTOR
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model"""
    def __init__(self, model, apt_features, apt_encoder):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
    
    def predict_vector_similarity(self, vector1, vector2):
        """Predict similarity between two vectors using cosine similarity"""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, 0.0, 1.0)
    
    def _feature_overlap_similarity(self, features1, features2):
        """Calculate similarity based on feature overlap (kept for compatibility)"""
        overlap_score = 0.0
        # Technique similarity
        if features1[0] == features2[0] and features1[0] != -1:
            overlap_score += 0.6
        # Software similarity  
        if features1[1] == features2[1] and features1[1] != -1:
            overlap_score += 0.5
        # CVE similarity
        if features1[2] == features2[2] and features1[2] != -1:
            overlap_score += 0.2
        return overlap_score
    
    def find_similar_apts(self, features, top_k=5):
        """Find most similar APTs to given features"""
        if all(f == -1 for f in features):
            return []
        probabilities = self.model.predict_proba([features])[0]
        similar_apts = []
        for apt_idx, prob in enumerate(probabilities):
            if prob > 0.01:
                original_apt = self.reverse_encoder[apt_idx]
                similar_apts.append((original_apt, prob))
        similar_apts.sort(key=lambda x: x[1], reverse=True)
        return similar_apts[:top_k]

#####################################################################
# EFFICIENT QUERY MODEL CLASS
#####################################################################

class EfficientQueryModel:
    def __init__(self, model, apt_features, apt_encoder):
        self.predictor = EfficientXGBoostPredictor(model, apt_features, apt_encoder)
        self.apt_features = apt_features
    
    @staticmethod
    def create_query_vector_from_evidence(evidence, feature_metadata):
        """Convert evidence to binary vector using feature metadata"""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        cve = evidence.get('cve', [])
        
        vector = np.zeros(feature_metadata['pulse_offset'] + 
                        len(feature_metadata['all_pulse']), dtype=np.float32)
        
        # Mark all features from evidence
        for t in techniques:
            if t in feature_metadata['all_techniques']:
                idx = feature_metadata['all_techniques'].index(t)
                vector[idx] = 1.0
        
        for s in software:
            if s in feature_metadata['all_software']:
                idx = feature_metadata['all_software'].index(s) + feature_metadata['software_offset']
                vector[idx] = 1.0
        
        for v in cve:
            if v in feature_metadata['all_cve']:
                idx = feature_metadata['all_cve'].index(v) + feature_metadata['cve_offset']
                vector[idx] = 1.0
        
        return vector
    
    def get_apt_profile(self, apt_id):
        """Get APT profile for similarity calculation"""
        if apt_id in self.apt_features:
            return self.apt_features[apt_id]
        return {'common_techniques': [], 'common_software': [], 'common_cve': [], 'count': 0}

def convert_evidence_to_encoded(evidence, mapping_file):
    """Convert human-readable evidence to encoded IDs using the mapping file"""
    print(" LOADING MAPPING DATA...")
    df_mapping = pd.read_csv(mapping_file)
    
    # Create mapping dictionaries
    technique_mapping = {}
    software_mapping = {}
    cve_mapping = {}
    
    for _, row in df_mapping.iterrows():
        # Technique mapping
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            if row['techniques_Origin'] not in technique_mapping:
                technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        # Software mapping
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            if row['software_Origin'] not in software_mapping:
                software_mapping[row['software_Origin']] = row['software_Encoded']
        # CVE mapping
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            if row['CVE_ID_Origin'] not in cve_mapping:
                cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
    
    print(f" Mappings loaded: {len(technique_mapping)} techniques, {len(software_mapping)} software, "
          f"{len(cve_mapping)} cve")
    
    # Convert evidence
    encoded_evidence = {
        'techniques': [],
        'software': [],
        'cve': []
    }
    
    for technique in evidence.get('techniques', []):
        if technique in technique_mapping:
            encoded_evidence['techniques'].append(technique_mapping[technique])
        else:
            print(f"⚠  Warning: Technique '{technique}' not found in mapping")
    
    for software in evidence.get('software', []):
        if software in software_mapping:
            encoded_evidence['software'].append(software_mapping[software])
        else:
            print(f"⚠  Warning: Software '{software}' not found in mapping")
    
    for cve in evidence.get('cve', []):
        if cve in cve_mapping:
            encoded_evidence['cve'].append(cve_mapping[cve])
        else:
            print(f"⚠  Warning: CVE '{cve}' not found in mapping")
    
    print(f"\n CONVERSION RESULTS:")
    print(f"   Input techniques: {len(evidence.get('techniques', []))} items → Encoded: {len(encoded_evidence['techniques'])}")
    print(f"   Input software: {len(evidence.get('software', []))} items → Encoded: {len(encoded_evidence['software'])}")
    print(f"   Input CVE: {len(evidence.get('cve', []))} items → Encoded: {len(encoded_evidence['cve'])}")
    
    return encoded_evidence

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
            'cve': set(apt_data['CVE_ID'].unique()),
            'total_features': len(apt_data)
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles

# ============================================================================
# The normalization function is kept but no longer used in the main query.
# It can be removed if desired.
# ============================================================================
def normalize_similarities_by_profile(similarities, profiles, apt_ids, query_features_count,
                                      query_techniques, query_software, query_capec,
                                      query_cwe, query_cve, query_pulse):
    """Enhanced normalization with better comprehensive profile weighting"""
    print(" Applying enhanced profile size normalization...")
    normalized_similarities = []
    query_size = query_features_count
    for i, apt_id in enumerate(apt_ids):
        target_size = profiles[apt_id]['total_features']
        original_sim = similarities[i]
        profile = profiles[apt_id]
        tech_matches = len(set(query_techniques) & profile['techniques'])
        soft_matches = len(set(query_software) & profile['software'])
        cve_matches = len(set(query_cve) & profile['cve'])
        total_matches = tech_matches + soft_matches + cve_matches
        
        if target_size > 0 and query_size > 0:
            size_ratio = min(query_size, target_size) / max(query_size, target_size)
        else:
            size_ratio = 0.5
        
        profile_boost = 0.0
        is_comprehensive = profile['total_features'] > 100
        if is_comprehensive:
            size_boost = np.log(profile['total_features'] / 50) * 0.05
            match_boost = (total_matches / max(1, query_size)) * 0.15
            profile_boost = size_boost + match_boost
        else:
            if total_matches < 3 and target_size < 20:
                profile_boost = -0.1
        
        adjusted_similarity = original_sim * (0.4 + 0.6 * size_ratio) + profile_boost
        normalized_similarities.append(min(1.0, max(0.0, adjusted_similarity)))
    
    if normalized_similarities:
        max_sim = max(normalized_similarities)
        if max_sim > 0:
            normalized_similarities = [sim / max_sim for sim in normalized_similarities]
    
    print(f"   Enhanced normalization applied")
    print(f"   Range: {np.min(similarities):.4f}->{np.min(normalized_similarities):.4f} to "
          f"{np.max(similarities):.4f}->{np.max(normalized_similarities):.4f}")
    return np.array(normalized_similarities)

def query_apt_similarity_efficient(evidence):
    """QUERY APT SIMILARITY USING EFFICIENT XGBOOST MODEL"""
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    
    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '6feat_efficient_apt_similarity_nopca.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        print(f" Loaded {model_path}")
    except FileNotFoundError:
        print(f" Model file not found at {model_path}. Please run train.py first.")
        return [], {}
    
    model = saved_data['model']
    apt_features = saved_data['apt_features']
    apt_encoder = saved_data['apt_encoder']
    feature_metadata = saved_data['feature_metadata']
    
    # Get the stored vectors – try both keys
    if 'all_binary_vectors' in saved_data:
        stored_vectors = saved_data['all_binary_vectors']
        print(" Using binary vectors (no PCA).")
    else:
        print(" No stored vectors found in saved data!")
        return [], {}
    
    # CREATE QUERY VECTOR
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    cve = evidence.get('cve', [])
    
    query_vector = EfficientQueryModel.create_query_vector_from_evidence(
        {
            'techniques': techniques,
            'software': software,
            'cve': cve
        },
        feature_metadata
    )
    print(f" Query vector created with length: {len(query_vector)}")
    
    
    # CREATE PREDICTOR
    predictor = EfficientXGBoostPredictor(model, apt_features, apt_encoder)
    
    # LOAD APT PROFILES
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    print(f" USING EFFICIENT XGBOOST MODEL")
    print(f" Features: Direct classification + Feature overlap")
    print(f" Comparing query against {len(stored_vectors)} APT vectors")
    
    # PREDICT SIMILARITIES
    print("PREDICTING SIMILARITIES...")
    similarities = []
    valid_apt_ids = []
    for apt_id, apt_vector in stored_vectors.items():
        # Ensure same dimensions
        if len(query_vector) != len(apt_vector):
            # This shouldn't happen if we loaded correctly, but just in case
            min_len = min(len(query_vector), len(apt_vector))
            qv = query_vector[:min_len]
            av = apt_vector[:min_len]
            sim = predictor.predict_vector_similarity(qv, av)
        else:
            sim = predictor.predict_vector_similarity(query_vector, apt_vector)
        similarities.append(sim)
        valid_apt_ids.append(apt_id)
    
    similarities = np.array(similarities)
    print(f" Raw similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
    
    
    # SORT (using raw similarities)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_indices]
    sorted_apt_ids = [valid_apt_ids[i] for i in sorted_indices]
    
    # LOAD APT NAME MAPPINGS
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        apt_name_mapping[row['APTnameEncoded']] = row['APTnameOrigin']
    
    # PREPARE RESULTS
    results = []
    for i, (similarity, apt_id) in enumerate(zip(sorted_similarities, sorted_apt_ids)):
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile = profiles[apt_id]
        tech_matches = len(set(techniques) & profile['techniques'])
        soft_matches = len(set(software) & profile['software'])
        capec_matches = len(set(capec) & profile['capec'])
        cwe_matches = len(set(cwe) & profile['cwe'])
        cve_matches = len(set(cve) & profile['cve'])
        pulse_matches = len(set(pulse) & profile['pulse'])
        match_score = tech_matches + soft_matches + capec_matches + cwe_matches + cve_matches + pulse_matches
        is_comprehensive = profile['total_features'] > 100
        results.append({
            'rank': i + 1,
            'apt_name': apt_name,
            'similarity': similarity,
            'tech_match': f"{tech_matches}/{len(techniques)}",
            'soft_match': f"{soft_matches}/{len(software)}",
            'capec_match': f"{capec_matches}/{len(capec)}",
            'cwe_match': f"{cwe_matches}/{len(cwe)}",
            'cve_match': f"{cve_matches}/{len(cve)}",
            'pulse_match': f"{pulse_matches}/{len(pulse)}",
            'match_score': match_score,
            'profile_size': f"{len(profile['techniques'])}T/{len(profile['software'])}S/{len(profile['capec'])}Ca/{len(profile['cwe'])}Cw/{len(profile['cve'])}Cv/{len(profile['pulse'])}P",
            'apt_id': apt_id,
            'is_comprehensive': is_comprehensive,
            'total_features': profile['total_features']
        })
    
    return results, {'query_vector': query_vector, 'stored_vectors': stored_vectors}

def print_efficient_results_table(results, evidence):
    """PRINT RESULTS TABLE FOR EFFICIENT MODEL"""
    print()
    print(" QUERY EVIDENCE:")
    print(f"   Techniques: {len(evidence['techniques'])} items")
    print(f"   Software: {len(evidence['software'])} items")
    print(f"   CAPEC: {len(evidence['capec'])} items")
    print(f"   cwe: {len(evidence['cwe'])} items")
    print(f"   cve: {len(evidence['cve'])} items")
    print(f"   Pulse: {len(evidence['pulse'])} items")
    print()
    print(f" EFFICIENT XGBOOST SIMILARITY RANKINGS")
    print(f" SHOWING ALL {len(results)} APTs")
    print("=" * 130)
    
    header = (f"{'Rank':<4} {'APT Name':<35} {'Similarity':<10} {'Tech':<8} {'Soft':<8} "
              f"{'CAPEC':<8} {'CWE':<8} {'CVE':<8} {'Pulse':<8} {'Matches':<8} {'Profile':<12} {'Size'}")
    print(header)
    print("-" * 150)
    
    for result in results:
        apt_name = result['apt_name']
        if len(apt_name) > 34:
            apt_name = apt_name[:31] + "..."
        comp_marker = "★" if result['is_comprehensive'] else " "
        row = (f"{result['rank']:<4} {apt_name:<35} {result['similarity']:.4f}    "
               f"{result['tech_match']:<8} {result['soft_match']:<8} {result['capec_match']:<8} "
               f"{result['cwe_match']:<8} {result['cve_match']:<8} {result['pulse_match']:<8}"
               f"{result['match_score']:<8} {result['profile_size']:<12} {result['total_features']:>4}{comp_marker}")
        print(row)
    
    print("-" * 150)
    print("★ = Comprehensive profile (>100 features)")

def analyze_feature_matching_classification(encoded_evidence, results, profiles):
    """ANALYZE FEATURE MATCHING BY CATEGORY - SEPARATE TABLES FOR EACH FEATURE TYPE"""
    print("\n" + "="*80)
    print(" FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    # Extract encoded evidence
    techniques = encoded_evidence.get('techniques', [])
    software = encoded_evidence.get('software', [])
    capec = encoded_evidence.get('capec', [])
    cwe = encoded_evidence.get('cwe', [])
    cve = encoded_evidence.get('cve', [])
    pulse = encoded_evidence.get('pulse', [])
    
    # Create mapping from APT name to profile key
    apt_name_to_profile = {}
    for result in results:
        apt_name = result['apt_name']
        apt_id = result['apt_id']
        if apt_id in profiles:
            apt_name_to_profile[apt_name] = profiles[apt_id]
    
    # Calculate feature matching scores for all APTs
    all_feature_scores = []
    
    for result in results:
        apt_name = result['apt_name']
        profile = apt_name_to_profile.get(apt_name)
        
        if profile:
            # Calculate individual feature matches
            tech_matches = len(set(techniques) & profile['techniques'])
            soft_matches = len(set(software) & profile['software'])
            capec_matches = len(set(capec) & profile['capec'])
            cwe_matches = len(set(cwe) & profile['cwe'])
            cve_matches = len(set(cve) & profile['cve'])
            pulse_matches = len(set(pulse) & profile['pulse'])
            
            # Calculate normalized scores (0-1)
            # Evidence coverage: how much of the evidence is covered by the APT profile
            tech_evidence_coverage = tech_matches / max(1, len(techniques))
            soft_evidence_coverage = soft_matches / max(1, len(software))
            capec_evidence_coverage = capec_matches / max(1, len(capec))
            cwe_evidence_coverage = cwe_matches / max(1, len(cwe))
            cve_evidence_coverage = cve_matches / max(1, len(cve))
            pulse_evidence_coverage = pulse_matches / max(1, len(pulse))
            
            # Profile coverage: how much of the APT profile matches the evidence
            tech_profile_coverage = tech_matches / max(1, len(profile['techniques']))
            soft_profile_coverage = soft_matches / max(1, len(profile['software']))
            capec_profile_coverage = capec_matches / max(1, len(profile['capec']))
            cwe_profile_coverage = cwe_matches / max(1, len(profile['cwe']))
            cve_profile_coverage = cve_matches / max(1, len(profile['cve']))
            pulse_profile_coverage = pulse_matches / max(1, len(profile['pulse']))
            
            # Combined score: harmonic mean of both coverages
            def harmonic_mean(a, b):
                if a + b == 0:
                    return 0
                return (2 * a * b) / (a + b)
            
            tech_score = harmonic_mean(tech_evidence_coverage, tech_profile_coverage)
            soft_score = harmonic_mean(soft_evidence_coverage, soft_profile_coverage)
            capec_score = harmonic_mean(capec_evidence_coverage, capec_profile_coverage)
            cwe_score = harmonic_mean(cwe_evidence_coverage, cwe_profile_coverage)
            cve_score = harmonic_mean(cve_evidence_coverage, cve_profile_coverage)
            pulse_score = harmonic_mean(pulse_evidence_coverage, pulse_profile_coverage)
            
            all_feature_scores.append({
                'apt_name': apt_name,
                'tech_score': tech_score,
                'soft_score': soft_score,
                'capec_score': capec_score,
                'cwe_score': cwe_score,
                'cve_score': cve_score,
                'pulse_score': pulse_score,

                'tech_matches': tech_matches,
                'soft_matches': soft_matches,
                'capec_matches': capec_matches,
                'cwe_matches': cwe_matches,
                'cve_matches': cve_matches,
                'pulse_matches': pulse_matches,

                'tech_evidence_coverage': tech_evidence_coverage,
                'soft_evidence_coverage': soft_evidence_coverage,
                'capec_evidence_coverage': capec_evidence_coverage,
                'cwe_evidence_coverage': cwe_evidence_coverage,
                'cve_evidence_coverage': cve_evidence_coverage,
                'pulse_evidence_coverage': pulse_evidence_coverage,

                'tech_profile_coverage': tech_profile_coverage,
                'soft_profile_coverage': soft_profile_coverage,
                'capec_profile_coverage': capec_profile_coverage,
                'cwe_profile_coverage': cwe_profile_coverage,
                'cve_profile_coverage': cve_profile_coverage,
                'pulse_profile_coverage': pulse_profile_coverage,

                'total_matches': tech_matches + soft_matches + capec_matches + cwe_matches + cve_matches + pulse_matches
            })
    
    # TABLE 1: TOP 5 TTP MATCHING
    print("\n TOP 5 APTs BY TTP MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by TTP score
    ttp_sorted = sorted(all_feature_scores, key=lambda x: x['tech_score'], reverse=True)[:5]
    for i, score in enumerate(ttp_sorted, 1):
        total_techniques = len(apt_name_to_profile[score['apt_name']]['techniques'])
        row = f"{i:<4} {score['apt_name']:<30} {score['tech_score']:.4f}    " \
              f"{score['tech_matches']:>2}/{len(techniques):<2} ({score['tech_evidence_coverage']:.3f})  " \
              f"{score['tech_matches']:>2}/{total_techniques:<3} ({score['tech_profile_coverage']:.3f})  " \
              f"{score['tech_score']:.4f}"
        print(row)
    
    # TABLE 2: TOP 5 SOFTWARE MATCHING
    print("\n  TOP 5 APTs BY SOFTWARE MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by Software score
    sw_sorted = sorted(all_feature_scores, key=lambda x: x['soft_score'], reverse=True)[:5]
    for i, score in enumerate(sw_sorted, 1):
        total_software = len(apt_name_to_profile[score['apt_name']]['software'])
        row = f"{i:<4} {score['apt_name']:<30} {score['soft_score']:.4f}    " \
              f"{score['soft_matches']:>2}/{len(software):<2} ({score['soft_evidence_coverage']:.3f})  " \
              f"{score['soft_matches']:>2}/{total_software:<3} ({score['soft_profile_coverage']:.3f})  " \
              f"{score['soft_score']:.4f}"
        print(row)
    
    # TABLE 3: TOP 5 CAPEC MATCHING
    print("\n TOP 5 APTs BY CAPEC MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CAPEC score
    capec_sorted = sorted(all_feature_scores, key=lambda x: x['capec_score'], reverse=True)[:5]
    for i, score in enumerate(capec_sorted, 1):
        total_capec = len(apt_name_to_profile[score['apt_name']]['capec'])
        row = f"{i:<4} {score['apt_name']:<30} {score['capec_score']:.4f}    " \
              f"{score['capec_matches']:>2}/{len(capec):<2} ({score['capec_evidence_coverage']:.3f})  " \
              f"{score['capec_matches']:>2}/{total_capec:<3} ({score['capec_profile_coverage']:.3f})  " \
              f"{score['capec_score']:.4f}"
        print(row)

    # TABLE 4: TOP 5 CWE MATCHING
    print("\n TOP 5 APTs BY CWE MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CWE score
    cwe_sorted = sorted(all_feature_scores, key=lambda x: x['cwe_score'], reverse=True)[:5]
    for i, score in enumerate(cwe_sorted, 1):
        total_cwe = len(apt_name_to_profile[score['apt_name']]['cwe'])
        row = f"{i:<4} {score['apt_name']:<30} {score['cwe_score']:.4f}    " \
              f"{score['cwe_matches']:>2}/{len(cwe):<2} ({score['cwe_evidence_coverage']:.3f})  " \
              f"{score['cwe_matches']:>2}/{total_cwe:<3} ({score['cwe_profile_coverage']:.3f})  " \
              f"{score['cwe_score']:.4f}"
        print(row)

    # TABLE 5: TOP 5 CVE MATCHING
    print("\n TOP 5 APTs BY CVE MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CVE score
    cve_sorted = sorted(all_feature_scores, key=lambda x: x['cve_score'], reverse=True)[:5]
    for i, score in enumerate(cve_sorted, 1):
        total_cve = len(apt_name_to_profile[score['apt_name']]['cve'])
        row = f"{i:<4} {score['apt_name']:<30} {score['cve_score']:.4f}    " \
              f"{score['cve_matches']:>2}/{len(cve):<2} ({score['cve_evidence_coverage']:.3f})  " \
              f"{score['cve_matches']:>2}/{total_cve:<3} ({score['cve_profile_coverage']:.3f})  " \
              f"{score['cve_score']:.4f}"
        print(row)

    # TABLE 6: TOP 5 Pulse MATCHING
    print("\n TOP 5 APTs BY Pulse MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by Pulse score
    pulse_sorted = sorted(all_feature_scores, key=lambda x: x['pulse_score'], reverse=True)[:5]
    for i, score in enumerate(pulse_sorted, 1):
        total_pulse = len(apt_name_to_profile[score['apt_name']]['pulse'])
        row = f"{i:<4} {score['apt_name']:<30} {score['pulse_score']:.4f}    " \
              f"{score['pulse_matches']:>2}/{len(pulse):<2} ({score['pulse_evidence_coverage']:.3f})  " \
              f"{score['pulse_matches']:>2}/{total_pulse:<3} ({score['pulse_profile_coverage']:.3f})  " \
              f"{score['pulse_score']:.4f}"
        print(row)
    
    # SUMMARY STATISTICS
    print("\n FEATURE MATCHING SUMMARY")
    print("-" * 50)
    
    # Calculate averages
    avg_tech = np.mean([s['tech_score'] for s in all_feature_scores])
    avg_soft = np.mean([s['soft_score'] for s in all_feature_scores])
    avg_capec = np.mean([s['capec_score'] for s in all_feature_scores])
    avg_cwe = np.mean([s['cwe_score'] for s in all_feature_scores])
    avg_cve = np.mean([s['cve_score'] for s in all_feature_scores])
    avg_pulse = np.mean([s['pulse_score'] for s in all_feature_scores])
    
    print(f"Average TTP matching score: {avg_tech:.4f}")
    print(f"Average Software matching score: {avg_soft:.4f}")
    print(f"Average CAPEC matching score: {avg_capec:.4f}")
    print(f"Average CWE matching score: {avg_cwe:.4f}")
    print(f"Average CVE matching score: {avg_cve:.4f}")
    print(f"Average Pulse matching score: {avg_pulse:.4f}")
    
    # Identify most distinctive feature
    max_avg = max(avg_tech, avg_soft, avg_capec, avg_cwe, avg_cve, avg_pulse)
    if max_avg == avg_tech:
        distinctive = "TTPs"
    elif max_avg == avg_soft:
        distinctive = "Software"
    elif max_avg == avg_capec:
        distinctive = "CAPEC"
    elif max_avg == avg_cwe:
        distinctive = "CWE"
    elif max_avg == avg_cve:
        distinctive = "CVE"
    elif max_avg == avg_pulse:
        distinctive = "PULSE"
    else:
        distinctive = "Unknown"
    
    print(f"Most distinctive feature: {distinctive}")
    
    # FALSE FLAG DETECTION
    print("\n FALSE FLAG DETECTION ANALYSIS")
    print("-" * 50)
    
    # Check top APTs for inconsistent patterns
    top_apts_feature_scores = all_feature_scores[:10]  # Top 10 by overall similarity
    
    inconsistent_groups = []
    for score in top_apts_feature_scores:
        scores = [score['tech_score'], score['soft_score'], score['capec_score'], 
                  score['cwe_score'], score['cve_score'], score['pulse_score']]
        variance = np.var(scores)
        
        # High variance might indicate false flag
        if variance > 0.05:
            inconsistent_groups.append({
                'apt_name': score['apt_name'],
                'variance': variance,
                'tech_score': score['tech_score'],
                'soft_score': score['soft_score'],
                'capec_score': score['capec_score'],
                'cwe_score': score['cwe_score'],
                'cve_score': score['cve_score'],
                'pulse_score': score['pulse_score']
            })
    
    if inconsistent_groups:
        print("⚠  POTENTIAL FALSE FLAG INDICATORS:")
        for group in inconsistent_groups[:3]:
            print(f"   • {group['apt_name']}: Variance={group['variance']:.4f}")
            print(f"     TTPs: {group['tech_score']:.4f}, SW: {group['soft_score']:.4f}, CAPEC: {group['capec_score']:.4f}, CWE: {group['cwe_score']:.4f}, CVE: {group['cve_score']:.4f}, Pulse: {group['pulse_score']:.4f}")
    else:
        print(" No strong false flag indicators detected")
    
    return all_feature_scores

def hybrid_query(evidence):
    encodedEvidence = convert_evidence_to_encoded(evidence, "../BaseData/comparison_Encoded_df.csv")
    results, model_info = query_apt_similarity_efficient(encodedEvidence)
    return results

if __name__ == "__main__":
    import json
    with open('../query_input.json', 'r') as f:
        evidence_humanreadable = json.load(f)

    print(" RUNNING EFFICIENT XGBOOST APT SIMILARITY QUERY")
    print("=" * 70)
    
    evidence = convert_evidence_to_encoded(evidence_humanreadable, "../BaseData/comparison_Encoded_df.csv")
    results, model_info = query_apt_similarity_efficient(evidence)
    print_efficient_results_table(results, evidence)
    
    profiles = load_apt_profiles_from_data()
    feature_scores = analyze_feature_matching_classification(evidence, results, profiles)
    
    print(f"\n EFFICIENT QUERY COMPLETE!")
    print(f" Most similar: {results[0]['apt_name']} (score: {results[0]['similarity']:.4f})")
    print(f" Least similar: {results[-1]['apt_name']} (score: {results[-1]['similarity']:.4f})")