import pickle
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from collections import defaultdict
from itertools import islice
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)

# ============================================================================
# 🔽  PicklableXGBWrapper – MUST be defined here so pickle can load the model
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
# EFFICIENT XGBOOST PREDICTOR
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model"""
    def __init__(self, model, apt_features, apt_encoder):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
    
    def predict_proba_vector(self, vector):
        """Return class probabilities for a single binary vector using the XGBoost model."""
        return self.model.predict_proba(vector.reshape(1, -1))[0]
    
    def predict_vector_similarity(self, vector1, vector2):
        """Cosine similarity on binary vectors (kept for compatibility)."""
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
        if features1[0] == features2[0] and features1[0] != -1:
            overlap_score += 0.6
        if features1[1] == features2[1] and features1[1] != -1:
            overlap_score += 0.5
        if features1[2] == features2[2] and features1[2] != -1:
            overlap_score += 0.4
        if features1[3] == features2[3] and features1[3] != -1:
            overlap_score += 0.3
        if features1[4] == features2[4] and features1[4] != -1:
            overlap_score += 0.2
        if features1[5] == features2[5] and features1[5] != -1:
            overlap_score += 0.1
        return overlap_score
    
    def find_similar_apts(self, features, top_k=5):
        """Find most similar APTs to given features (not used in main query)."""
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
        capec = evidence.get('capec', [])
        cwe = evidence.get('cwe', [])
        cve = evidence.get('cve', [])
        pulse = evidence.get('pulse', [])
        
        vector = np.zeros(feature_metadata['pulse_offset'] + 
                        len(feature_metadata['all_pulse']), dtype=np.float32)
        
        for t in techniques:
            if t in feature_metadata['all_techniques']:
                idx = feature_metadata['all_techniques'].index(t)
                vector[idx] = 1.0
        
        for s in software:
            if s in feature_metadata['all_software']:
                idx = feature_metadata['all_software'].index(s) + feature_metadata['software_offset']
                vector[idx] = 1.0
        
        for c in capec:
            if c in feature_metadata['all_capec']:
                idx = feature_metadata['all_capec'].index(c) + feature_metadata['capec_offset']
                vector[idx] = 1.0
        
        for w in cwe:
            if w in feature_metadata['all_cwe']:
                idx = feature_metadata['all_cwe'].index(w) + feature_metadata['cwe_offset']
                vector[idx] = 1.0
        
        for v in cve:
            if v in feature_metadata['all_cve']:
                idx = feature_metadata['all_cve'].index(v) + feature_metadata['cve_offset']
                vector[idx] = 1.0
        
        for p in pulse:
            if p in feature_metadata['all_pulse']:
                idx = feature_metadata['all_pulse'].index(p) + feature_metadata['pulse_offset']
                vector[idx] = 1.0
        
        return vector
    
    def get_apt_profile(self, apt_id):
        """Get APT profile for similarity calculation"""
        if apt_id in self.apt_features:
            return self.apt_features[apt_id]
        return {'common_techniques': [], 'common_software': [], 'common_capec': [],
                'common_cwe': [], 'common_cve': [], 'common_pulse': [], 'count': 0}


# ==============================================
# SINGLE SCORING PATH (used by hybrid_query, evaluation and false flag)
# ==============================================
class XGBScorer:
    """XGBoost similarity (thesis Appendix 8.3.2): the evidence is turned into the binary feature
    vector used in training (1 = feature present) and the softmax class probability the model
    assigns to each APT is its similarity score."""

    def __init__(self, model_path=None):
        model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '6feat_efficient_apt_similarity_nopca.pkl')
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        self.feature_metadata = saved['feature_metadata']
        self.predictor = EfficientXGBoostPredictor(saved['model'], saved['apt_features'], saved['apt_encoder'])
        names = pd.read_csv(os.path.join(P.BASE, 'comparison_Encoded_df.csv'),
                            usecols=['APTnameEncoded', 'APTnameOrigin']).drop_duplicates('APTnameEncoded')
        name_of = dict(zip(names['APTnameEncoded'], names['APTnameOrigin']))
        self.apt_ids = sorted(self.predictor.apt_encoder.keys())
        self.class_index = np.array([self.predictor.apt_encoder[a] for a in self.apt_ids])
        self.apt_names = [name_of.get(a, f"APT{a}") for a in self.apt_ids]

    def score(self, encoded_evidence):
        vector = EfficientQueryModel.create_query_vector_from_evidence(P.non_empty(encoded_evidence),
                                                                      self.feature_metadata)
        probs = self.predictor.predict_proba_vector(vector)
        return np.asarray(probs)[self.class_index]


_SCORER = None


def get_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = XGBScorer()
    return _SCORER

_MAPPING_CACHE = {}
_MAP_COLUMNS = {'techniques': 'techniques', 'software': 'software', 'capec': 'CAPEC_ID',
                'cwe': 'CWE_ID', 'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}


def _evidence_mappings(mapping_file):
    """Origin -> encoded ID dictionaries, built once (first occurrence wins, as before)."""
    if mapping_file not in _MAPPING_CACHE:
        print(" LOADING MAPPING DATA...")
        df_mapping = pd.read_csv(mapping_file)
        maps = {}
        for key in P.FEATURES:
            col = _MAP_COLUMNS[key]
            pair = df_mapping[[f"{col}_Origin", f"{col}_Encoded"]].dropna()
            pair = pair.drop_duplicates(subset=f"{col}_Origin", keep="first")
            maps[key] = dict(zip(pair[f"{col}_Origin"], (int(v) for v in pair[f"{col}_Encoded"])))
        _MAPPING_CACHE[mapping_file] = maps
    return _MAPPING_CACHE[mapping_file]


def convert_evidence_to_encoded(evidence, mapping_file):
    """Convert human-readable evidence to encoded IDs using the mapping file"""
    maps = _evidence_mappings(mapping_file)
    encoded_evidence = {key: [] for key in P.FEATURES}
    for key in P.FEATURES:
        for item in evidence.get(key, []):
            if item in maps[key]:
                encoded_evidence[key].append(maps[key][item])
            else:
                print(f"⚠  Warning: {key} '{item}' not found in mapping")
    print(" CONVERSION RESULTS: " + ", ".join(
        f"{key} {len(evidence.get(key, []))}->{len(encoded_evidence[key])}" for key in P.FEATURES))
    return encoded_evidence

def load_apt_profiles_from_data():
    """Load APT profiles from the encoded data file"""
    print(" LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].unique()) - {-1},
            'software': set(apt_data['software_str'].unique()) - {-1},
            'capec': set(apt_data['CAPEC_ID'].unique()) - {-1},
            'cwe': set(apt_data['CWE_ID'].unique()) - {-1},
            'cve': set(apt_data['CVE_ID'].unique()) - {-1},
            'pulse': set(apt_data['Pulse_ID'].unique()) - {-1},
            'total_features': len(apt_data)
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles

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
        capec_matches = len(set(query_capec) & profile['capec'])
        cwe_matches = len(set(query_cwe) & profile['cwe'])
        cve_matches = len(set(query_cve) & profile['cve'])
        pulse_matches = len(set(query_pulse) & profile['pulse'])
        total_matches = tech_matches + soft_matches + capec_matches + cwe_matches + cve_matches + pulse_matches
        
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
    """QUERY APT SIMILARITY USING EFFICIENT XGBOOST MODEL (NOW USES MODEL PROBABILITIES)"""
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    
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
    
    if 'all_binary_vectors' in saved_data:
        stored_vectors = saved_data['all_binary_vectors']
        print(" Binary vectors loaded (not used for scoring).")
    else:
        print(" No stored vectors found in saved data!")
        return [], {}
    
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    capec = evidence.get('capec', [])
    cwe = evidence.get('cwe', [])
    cve = evidence.get('cve', [])
    pulse = evidence.get('pulse', [])
    
    query_vector = EfficientQueryModel.create_query_vector_from_evidence(
        {'techniques': techniques, 'software': software, 'capec': capec,
         'cwe': cwe, 'cve': cve, 'pulse': pulse},
        feature_metadata
    )
    print(f" Query vector created with length: {len(query_vector)}")
    
    predictor = EfficientXGBoostPredictor(model, apt_features, apt_encoder)
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    print(f" USING EFFICIENT XGBOOST MODEL (CLASS PROBABILITIES)")
    print(f" Comparing query against {len(apt_encoder)} APT classes")
    
    print("PREDICTING SIMILARITIES...")
    probs = predictor.predict_proba_vector(query_vector)
    
    similarities = []
    valid_apt_ids = []
    for apt_id in apt_ids:
        if apt_id in predictor.apt_encoder:
            idx = predictor.apt_encoder[apt_id]
            sim = probs[idx]
            similarities.append(sim)
            valid_apt_ids.append(apt_id)
        else:
            similarities.append(0.0)
            valid_apt_ids.append(apt_id)
    
    similarities = np.array(similarities)
    print(f" Raw similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
    
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_indices]
    sorted_apt_ids = [valid_apt_ids[i] for i in sorted_indices]
    
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        apt_name_mapping[row['APTnameEncoded']] = row['APTnameOrigin']
    
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
    print(f" EFFICIENT XGBOOST SIMILARITY RANKINGS (USING MODEL PROBABILITIES)")
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
    
    techniques = encoded_evidence.get('techniques', [])
    software = encoded_evidence.get('software', [])
    capec = encoded_evidence.get('capec', [])
    cwe = encoded_evidence.get('cwe', [])
    cve = encoded_evidence.get('cve', [])
    pulse = encoded_evidence.get('pulse', [])
    
    apt_name_to_profile = {}
    for result in results:
        apt_name = result['apt_name']
        apt_id = result['apt_id']
        if apt_id in profiles:
            apt_name_to_profile[apt_name] = profiles[apt_id]
    
    all_feature_scores = []
    for result in results:
        apt_name = result['apt_name']
        profile = apt_name_to_profile.get(apt_name)
        if profile:
            tech_matches = len(set(techniques) & profile['techniques'])
            soft_matches = len(set(software) & profile['software'])
            capec_matches = len(set(capec) & profile['capec'])
            cwe_matches = len(set(cwe) & profile['cwe'])
            cve_matches = len(set(cve) & profile['cve'])
            pulse_matches = len(set(pulse) & profile['pulse'])
            
            tech_evidence_coverage = tech_matches / max(1, len(techniques))
            soft_evidence_coverage = soft_matches / max(1, len(software))
            capec_evidence_coverage = capec_matches / max(1, len(capec))
            cwe_evidence_coverage = cwe_matches / max(1, len(cwe))
            cve_evidence_coverage = cve_matches / max(1, len(cve))
            pulse_evidence_coverage = pulse_matches / max(1, len(pulse))
            
            tech_profile_coverage = tech_matches / max(1, len(profile['techniques']))
            soft_profile_coverage = soft_matches / max(1, len(profile['software']))
            capec_profile_coverage = capec_matches / max(1, len(profile['capec']))
            cwe_profile_coverage = cwe_matches / max(1, len(profile['cwe']))
            cve_profile_coverage = cve_matches / max(1, len(profile['cve']))
            pulse_profile_coverage = pulse_matches / max(1, len(profile['pulse']))
            
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
    
    top_apts_feature_scores = all_feature_scores[:10]
    
    inconsistent_groups = []
    for score in top_apts_feature_scores:
        scores = [score['tech_score'], score['soft_score'], score['capec_score'], 
                  score['cwe_score'], score['cve_score'], score['pulse_score']]
        variance = np.var(scores)
        
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
    """CTI / hybrid entry point: human-readable evidence -> ranked list of all APTs (mid-ranks)."""
    encoded_evidence = convert_evidence_to_encoded(evidence, os.path.join(P.BASE, "comparison_Encoded_df.csv"))
    scorer = get_scorer()
    return P.ranked_results(scorer.apt_names, scorer.score(encoded_evidence), score_key='similarity')

if __name__ == "__main__":
    import json
    with open('../query_input.json', 'r') as f:
        evidence_humanreadable = json.load(f)

    print(" RUNNING EFFICIENT XGBOOST APT SIMILARITY QUERY (USING MODEL)")
    print("=" * 70)
    
    evidence = convert_evidence_to_encoded(evidence_humanreadable, "../BaseData/comparison_Encoded_df.csv")
    results, model_info = query_apt_similarity_efficient(evidence)
    print_efficient_results_table(results, evidence)
    
    profiles = load_apt_profiles_from_data()
    feature_scores = analyze_feature_matching_classification(evidence, results, profiles)
    
    print(f"\n EFFICIENT QUERY COMPLETE!")
    print(f" Most similar: {results[0]['apt_name']} (score: {results[0]['similarity']:.4f})")
    print(f" Least similar: {results[-1]['apt_name']} (score: {results[-1]['similarity']:.4f})")