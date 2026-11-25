import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)

def analyze_feature_matching_classification(encoded_evidence, results, apt_profiles, apt_name_dict):
    """ANALYZE FEATURE MATCHING BY CATEGORY - SEPARATE TABLES FOR TTP, SOFTWARE, CVE"""
    print("\n" + "="*80)
    print(" FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    # Prepare feature matching breakdown
    techniques = encoded_evidence.get('techniques', [])
    software = encoded_evidence.get('software', [])
    cve = encoded_evidence.get('cve', [])
    
    # Create a mapping from APT name to profile key
    apt_name_to_key = {}
    for result in results:
        apt_name = result['APT']
        # Find the apt profile key that matches this APT name
        for apt_key, profile in apt_profiles.items():
            if apt_name_dict[profile['encoded_id']] == apt_name:
                apt_name_to_key[apt_name] = apt_key
                break
    
    # Calculate feature matching scores for all APTs
    all_feature_scores = []
    
    for result in results:
        apt_name = result['APT']
        apt_key = apt_name_to_key.get(apt_name)
        
        if apt_key and apt_key in apt_profiles:
            profile = apt_profiles[apt_key]
            
            # Calculate individual feature matches
            tech_matches = len(set(techniques) & set(profile.get('techniques', [])))
            soft_matches = len(set(software) & set(profile.get('software', [])))
            cve_matches = len(set(cve) & set(profile.get('cve', [])))
            
            # Calculate normalized scores (0-1)
            # Evidence coverage: how much of the evidence is covered by the APT profile
            tech_evidence_coverage = tech_matches / max(1, len(techniques))
            soft_evidence_coverage = soft_matches / max(1, len(software))
            cve_evidence_coverage = cve_matches / max(1, len(cve))
            
            # Profile coverage: how much of the APT profile matches the evidence
            tech_profile_coverage = tech_matches / max(1, len(profile.get('techniques', [])))
            soft_profile_coverage = soft_matches / max(1, len(profile.get('software', [])))
            cve_profile_coverage = cve_matches / max(1, len(profile.get('cve', [])))
            
            # Combined score: harmonic mean of both coverages
            def harmonic_mean(a, b):
                if a + b == 0:
                    return 0
                return (2 * a * b) / (a + b)
            
            tech_score = harmonic_mean(tech_evidence_coverage, tech_profile_coverage)
            soft_score = harmonic_mean(soft_evidence_coverage, soft_profile_coverage)
            cve_score = harmonic_mean(cve_evidence_coverage, cve_profile_coverage)
            
            all_feature_scores.append({
                'apt_name': apt_name,
                'apt_key': apt_key,
                'tech_score': tech_score,
                'soft_score': soft_score,
                'cve_score': cve_score,

                'tech_matches': tech_matches,
                'soft_matches': soft_matches,
                'cve_matches': cve_matches,

                'tech_evidence_coverage': tech_evidence_coverage,
                'soft_evidence_coverage': soft_evidence_coverage,
                'cve_evidence_coverage': cve_evidence_coverage,

                'tech_profile_coverage': tech_profile_coverage,
                'soft_profile_coverage': soft_profile_coverage,
                'cve_profile_coverage': cve_profile_coverage,

                'total_matches': tech_matches + soft_matches + cve_matches
            })
    
    # TABLE 1: TOP 5 TTP MATCHING
    print("\n TOP 5 APTs BY TTP MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by TTP score
    ttp_sorted = sorted(all_feature_scores, key=lambda x: x['tech_score'], reverse=True)[:5]
    for i, score in enumerate(ttp_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_techniques = len(profile.get('techniques', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['tech_score']:.4f}    " \
              f"{score['tech_matches']:>2}/{len(techniques):<2} ({score['tech_evidence_coverage']:.3f})  " \
              f"{score['tech_matches']:>2}/{total_techniques:<3} ({score['tech_profile_coverage']:.3f})  " \
              f"{score['tech_score']:.4f}"
        print(row)
    
    # TABLE 2: TOP 5 SOFTWARE MATCHING
    print("\n  TOP 5 APTs BY SOFTWARE MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by Software score
    sw_sorted = sorted(all_feature_scores, key=lambda x: x['soft_score'], reverse=True)[:5]
    for i, score in enumerate(sw_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_software = len(profile.get('software', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['soft_score']:.4f}    " \
              f"{score['soft_matches']:>2}/{len(software):<2} ({score['soft_evidence_coverage']:.3f})  " \
              f"{score['soft_matches']:>2}/{total_software:<3} ({score['soft_profile_coverage']:.3f})  " \
              f"{score['soft_score']:.4f}"
        print(row)

    # TABLE 5: TOP 5 CVE MATCHING
    print("\n TOP 5 APTs BY CVE MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CVE score
    cve_sorted = sorted(all_feature_scores, key=lambda x: x['cve_score'], reverse=True)[:5]
    for i, score in enumerate(cve_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_cve = len(profile.get('cve', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['cve_score']:.4f}    " \
              f"{score['cve_matches']:>2}/{len(cve):<2} ({score['cve_evidence_coverage']:.3f})  " \
              f"{score['cve_matches']:>2}/{total_cve:<3} ({score['cve_profile_coverage']:.3f})  " \
              f"{score['cve_score']:.4f}"
        print(row)

    # SUMMARY STATISTICS
    print("\n FEATURE MATCHING SUMMARY")
    print("-" * 50)
    
    # Calculate averages
    avg_tech = np.mean([s['tech_score'] for s in all_feature_scores])
    avg_soft = np.mean([s['soft_score'] for s in all_feature_scores])
    avg_cve = np.mean([s['cve_score'] for s in all_feature_scores])
    
    print(f"Average TTP matching score: {avg_tech:.4f}")
    print(f"Average Software matching score: {avg_soft:.4f}")
    print(f"Average CVE matching score: {avg_cve:.4f}")
    
    # Identify most distinctive feature
    max_avg = max(avg_tech, avg_soft, avg_cve)
    if max_avg == avg_tech:
        distinctive = "TTPs"
    elif max_avg == avg_soft:
        distinctive = "Software"
    elif max_avg == avg_cve:
        distinctive = "CVE"
    else:
        pass
    
    print(f"Most distinctive feature: {distinctive}")
    
    # FALSE FLAG DETECTION
    print("\n FALSE FLAG DETECTION ANALYSIS")
    print("-" * 50)
    
    # Check top APTs for inconsistent patterns
    top_apts_feature_scores = all_feature_scores[:10]  # Top 10 by overall similarity
    
    inconsistent_groups = []
    for score in top_apts_feature_scores:
        scores = [score['tech_score'], score['soft_score'], score['cve_score']]
        variance = np.var(scores)
        
        # High variance might indicate false flag
        if variance > 0.05:
            inconsistent_groups.append({
                'apt_name': score['apt_name'],
                'variance': variance,
                'tech_score': score['tech_score'],
                'soft_score': score['soft_score'],
                'cve_score': score['cve_score']
            })
    
    if inconsistent_groups:
        print("⚠  POTENTIAL FALSE FLAG INDICATORS:")
        for group in inconsistent_groups[:3]:
            print(f"   • {group['apt_name']}: Variance={group['variance']:.4f}")
            print(f"     TTPs: {group['tech_score']:.4f}, SW: {group['soft_score']:.4f}, cve: {group['cve_score']:.4f}")
    else:
        print(" No strong false flag indicators detected")
    
    return all_feature_scores

def load_model(status=None):
    """Load the trained Random Forest and its data (path independent of the working directory)."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3feat_apt_similarity_model.joblib')
    model_data = joblib.load(model_path)
    print("Model loaded successfully")
    return model_data

def sanitize_evidence(evidence, df_mapping):
    """Keep only evidence features that exist in our database"""
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques': ('techniques_Origin', 'techniques_Encoded'),
        'software': ('software_Origin', 'software_Encoded'),
        'cve': ('CVE_ID_Origin', 'CVE_ID_Encoded')
    }
    
    for evidence_key, (origin_col, _) in column_mapping.items():
        if evidence_key in evidence:
            valid_values = set(df_mapping[origin_col].dropna())
            valid_values.discard('')
            
            valid = []
            invalid = []
            
            for item in evidence[evidence_key]:
                if item == '':
                    continue
                if str(item) in valid_values:
                    valid.append(item)
                else:
                    invalid.append(item)
            
            if valid:
                sanitized[evidence_key] = valid
            if invalid:
                missing[evidence_key] = invalid
    
    if missing:
        print("WARNING: These features were not found in our database:")
        for feature_type, items in missing.items():
            print(f"- {feature_type}: {items[:5]}")
            if len(items) > 5:
                print(f"  (and {len(items)-5} more)")
    
    return sanitized

def translate_to_encoded(evidence, df_mapping):
    """Convert sanitized evidence to encoded IDs"""
    encoded_evidence = {}
    
    column_mapping = {
        'techniques': ('techniques_Encoded', 'techniques'),
        'software': ('software_Encoded', 'software'),
        'cve': ('CVE_ID_Encoded', 'cve')
    }
    
    
    for evidence_key, (encoded_col, output_key) in column_mapping.items():
        if evidence_key in evidence:
            origin_col = encoded_col.replace('_Encoded', '_Origin')
            mapping = dict(zip(df_mapping[origin_col], df_mapping[encoded_col]))
            
            encoded_values = []
            for val in evidence[evidence_key]:
                if val in mapping:
                    encoded_values.append(mapping[val])
            
            if encoded_values:
                encoded_evidence[output_key] = encoded_values
    
    return encoded_evidence

def create_evidence_vector(encoded_evidence, binarizers, all_features, feature_types):
    """Create a feature vector for the evidence"""
    evidence_vector = []
    
    for ft in feature_types:
        if ft in encoded_evidence:
            valid_features = [f for f in encoded_evidence[ft] if f in all_features[ft]]
            binary_vec = binarizers[ft].transform([valid_features]).flatten()
        else:
            binary_vec = np.zeros(len(all_features[ft]))
        evidence_vector.extend(binary_vec)
    
    return np.array(evidence_vector)

def calculate_rf_similarity(evidence_vector, rf_model, X, apt_names):
    """Calculate similarity scores using Random Forest"""
    evidence_leaves = np.array([tree.apply(evidence_vector.reshape(1, -1)) for tree in rf_model.estimators_]).flatten()
    
    apt_leaves = []
    for tree in rf_model.estimators_:
        apt_leaves.append(tree.apply(X))
    apt_leaves = np.array(apt_leaves)
    
    similarities = np.mean(apt_leaves == evidence_leaves.reshape(-1, 1), axis=0)
    
    return similarities


# ==============================================
# SINGLE SCORING PATH (used by hybrid_query, evaluation and false flag)
# ==============================================
class RFScorer:
    """Leaf-based similarity (thesis Appendix 8.3.2): the share of trees in which the evidence
    vector and an APT profile vector end in the same leaf. Same computation as
    calculate_rf_similarity, with the APT leaves computed once."""

    def __init__(self, model_data=None):
        self.md = model_data if model_data is not None else load_model("hybrid")
        self.rf = self.md['rf_model']
        self.apt_names = [self.md['apt_name_dict'][self.md['apt_profiles'][a]['encoded_id']]
                          for a in self.md['apt_names']]
        self.apt_leaves = np.array([tree.apply(self.md['X']) for tree in self.rf.estimators_])
        self.df_mapping = self.md['df_mapping']

    def score(self, encoded_evidence):
        vector = create_evidence_vector(P.non_empty(encoded_evidence), self.md['binarizers'],
                                        self.md['all_features'], self.md['feature_types'])
        evidence_leaves = np.array([tree.apply(vector.reshape(1, -1))
                                    for tree in self.rf.estimators_]).flatten()
        return np.mean(self.apt_leaves == evidence_leaves.reshape(-1, 1), axis=0)


_SCORER = None


def get_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = RFScorer()
    return _SCORER

def print_rf_results_table(results, encoded_evidence, apt_profiles, apt_name_dict):
    """Print results table for RF model"""
    print("\n QUERY EVIDENCE:")
    print(f"   Techniques: {len(encoded_evidence.get('techniques', []))} items")
    print(f"   Software: {len(encoded_evidence.get('software', []))} items")
    print(f"   cve: {len(encoded_evidence.get('cve', []))} items")
    print()
    print(f" RANDOM FOREST SIMILARITY RANKINGS")
    print("=" * 120)
    
    # Header
    header = f"{'rank':<4} {'APT Name':<35} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CVE':<8} {'Matches':<8} {'Profile'}"
    print(header)
    print("-" * 120)
    
    for result in results:
        # Truncate long APT names
        apt_name = result['APT']
        if len(apt_name) > 34:
            apt_name = apt_name[:31] + "..."
        
        # Get profile information
        total_matches = result['techniques'] + result['software'] + result['cve']
        
        # Find profile for this APT
        profile_info = ""
        for apt_key, profile in apt_profiles.items():
            if apt_name_dict[profile['encoded_id']] == result['APT']:
                tech_count = len(profile.get('techniques', []))
                soft_count = len(profile.get('software', []))
                cve_count = len(profile.get('cve', []))
                profile_info = f"{tech_count}T/{soft_count}S/{cve_count}Cv"
                break
        
        row = (f"{result['rank']:<4} {apt_name:<35} {result['Score']:.4f}    "
               f"{result['techniques']}/{len(encoded_evidence.get('techniques', [])):<7} "
               f"{result['software']}/{len(encoded_evidence.get('software', [])):<7} "
               f"{result['cve']}/{len(encoded_evidence.get('cve', [])):<7} "
               f"{total_matches:<8} {profile_info}")
        print(row)
    
    print("-" * 120)


def hybrid_query(evidence):
    """CTI / hybrid entry point: human-readable evidence -> ranked list of all APTs (mid-ranks)."""
    scorer = get_scorer()
    clean_evidence = sanitize_evidence(evidence, scorer.df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, scorer.df_mapping)
    scores = scorer.score(encoded_evidence)
    return P.ranked_results(scorer.apt_names, scores, score_key='Score')

def main():
    # Load the model
    model_data = load_model("nonhybrid")
    
    # Extract components from model data
    rf = model_data['rf_model']
    binarizers = model_data['binarizers']
    all_features = model_data['all_features']
    apt_profiles = model_data['apt_profiles']
    apt_names = model_data['apt_names']
    apt_name_dict = model_data['apt_name_dict']
    X = model_data['X']
    feature_types = model_data['feature_types']
    df_mapping = model_data['df_mapping']
    
    # Process custom evidence
    print("\n" + "="*80)
    print("PROCESSING CUSTOM EVIDENCE")
    print("="*80)
    
    # Define your evidence here
    # evidence = {
    #     'techniques': ['T1001.002', 'T1003.002', 'T1003.004', 'T1003.006', 'T1005', 'T1016.001', 'T1018', 'T1021.001', 'T1021.002', 'T1021.006', 'T1021.007', 'T1027.001', 'T1027.002', 'T1027.003', 'T1027.006', 'T1036.004', 'T1036.005', 'T1037', 'T1037.004', 'T1047', 'T1048.002', 'T1053.005', 'T1057', 'T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1059.009', 'T1068', 'T1069.002', 'T1070.004', 'T1070.006', 'T1070.008', 'T1071.001', 'T1074.002', 'T1078', 'T1078.002', 'T1078.003', 'T1078.004', 'T1082', 'T1083', 'T1087.002', 'T1087.004', 'T1090.001', 'T1090.002', 'T1090.003', 'T1090.004', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.005', 'T1102.002', 'T1105', 'T1110.001', 'T1110.003', 'T1114.002', 'T1133', 'T1136.003', 'T1140', 'T1190', 'T1195.002', 'T1199', 'T1203', 'T1204.001', 'T1204.002', 'T1213', 'T1213.003', 'T1218.005', 'T1218.011', 'T1482', 'T1484.002', 'T1505.003', 'T1528', 'T1539', 'T1546.003', 'T1546.008', 'T1547.001', 'T1548.002', 'T1550.001', 'T1550.003', 'T1550.004', 'T1552.004', 'T1553.002', 'T1553.005', 'T1555.003', 'T1556.007', 'T1558.003', 'T1560.001', 'T1562.001', 'T1562.002', 'T1562.004', 'T1562.008', 'T1566.001', 'T1566.002', 'T1566.003', 'T1568', 'T1573', 'T1583.001', 'T1583.006', 'T1584.001', 'T1585.001', 'T1586.002', 'T1586.003', 'T1587.001', 'T1587.003', 'T1588.002', 'T1589.001', 'T1595.002', 'T1606.001', 'T1606.002', 'T1621', 'T1649', 'T1651', 'T1665'],
    #     'software': ['S0002', 'S0029', 'S0037', 'S0039', 'S0046', 'S0048', 'S0049', 'S0050', 'S0051', 'S0052', 'S0053', 'S0054', 'S0057', 'S0096', 'S0100', 'S0139', 'S0150', 'S0154', 'S0175', 'S0183', 'S0195', 'S0357', 'S0511', 'S0512', 'S0513', 'S0514', 'S0515', 'S0516', 'S0518', 'S0521', 'S0552', 'S0559', 'S0560', 'S0562', 'S0565', 'S0588', 'S0589', 'S0597', 'S0633', 'S0634', 'S0635', 'S0636', 'S0637', 'S0661', 'S0677', 'S0682', 'S0684', 'S1084'],
    #     'cve': ['CVE-2012-0433', 'CVE-2013-10007', 'CVE-2016-2124', 'CVE-2016-8612', 'CVE-2017-12233', 'CVE-2017-12234', 'CVE-2017-12235', 'CVE-2017-12240', 'CVE-2017-12319', 'CVE-2017-3167', 'CVE-2017-5174', 'CVE-2017-7501', 'CVE-2017-7517', 'CVE-2017-7924', 'CVE-2018-0125', 'CVE-2018-0147', 'CVE-2018-0158', 'CVE-2018-0159', 'CVE-2018-0171', 'CVE-2018-0172', 'CVE-2018-0173', 'CVE-2018-0174', 'CVE-2018-0296', 'CVE-2018-10863', 'CVE-2018-10867', 'CVE-2018-1270', 'CVE-2018-1273', 'CVE-2018-14667', 'CVE-2018-15454', 'CVE-2018-5383', 'CVE-2019-10938', 'CVE-2019-12643', 'CVE-2019-14840', 'CVE-2019-14866', 'CVE-2019-15976', 'CVE-2019-1622', 'CVE-2019-1652', 'CVE-2019-1653', 'CVE-2019-1821', 'CVE-2019-18998', 'CVE-2019-3738', 'CVE-2019-7609', 'CVE-2020-10148', 'CVE-2020-10684', 'CVE-2020-10693', 'CVE-2020-10754', 'CVE-2020-12145', 'CVE-2020-13942', 'CVE-2020-14343', 'CVE-2020-15250', 'CVE-2020-15256', 'CVE-2020-16096', 'CVE-2020-16098', 'CVE-2020-16099', 'CVE-2020-16102', 'CVE-2020-1747', 'CVE-2020-17519', 'CVE-2020-1926', 'CVE-2020-2000', 'CVE-2020-2022', 'CVE-2020-2035', 'CVE-2020-2050', 'CVE-2020-2506', 'CVE-2020-25662', 'CVE-2020-25717', 'CVE-2020-26291', 'CVE-2020-27218', 'CVE-2020-27833', 'CVE-2020-27836', 'CVE-2020-3161', 'CVE-2020-3259', 'CVE-2020-3452', 'CVE-2020-3495', 'CVE-2020-35167', 'CVE-2020-35168', 'CVE-2020-35509', 'CVE-2020-36197', 'CVE-2020-36319', 'CVE-2020-5258', 'CVE-2020-5259', 'CVE-2020-5396', 'CVE-2020-5412', 'CVE-2020-5414', 'CVE-2020-5415', 'CVE-2020-5417', 'CVE-2020-5425', 'CVE-2020-7069', 'CVE-2020-7070', 'CVE-2020-7943', 'CVE-2020-8193', 'CVE-2020-8195', 'CVE-2020-8196', 'CVE-2020-8218', 'CVE-2020-8243', 'CVE-2020-8255', 'CVE-2021-1577', 'CVE-2021-1580', 'CVE-2021-20260', 'CVE-2021-21587', 'CVE-2021-21703', 'CVE-2021-21705', 'CVE-2021-21964', 'CVE-2021-21965', 'CVE-2021-21968', 'CVE-2021-22044', 'CVE-2021-22047', 'CVE-2021-22053', 'CVE-2021-22117', 'CVE-2021-22132', 'CVE-2021-22134', 'CVE-2021-22284', 'CVE-2021-22288', 'CVE-2021-22289', 'CVE-2021-22817', 'CVE-2021-22876', 'CVE-2021-22893', 'CVE-2021-22894', 'CVE-2021-22900', 'CVE-2021-22924', 'CVE-2021-22925', 'CVE-2021-22941', 'CVE-2021-22946', 'CVE-2021-23136', 'CVE-2021-23140', 'CVE-2021-23177', 'CVE-2021-23204', 'CVE-2021-24122', 'CVE-2021-25003', 'CVE-2021-25122', 'CVE-2021-25369', 'CVE-2021-25444', 'CVE-2021-25489', 'CVE-2021-25490', 'CVE-2021-25745', 'CVE-2021-25746', 'CVE-2021-26117', 'CVE-2021-26606', 'CVE-2021-26726', 'CVE-2021-26727', 'CVE-2021-26728', 'CVE-2021-26729', 'CVE-2021-26731', 'CVE-2021-26732', 'CVE-2021-26733', 'CVE-2021-27853', 'CVE-2021-27854', 'CVE-2021-27861', 'CVE-2021-27862', 'CVE-2021-28169', 'CVE-2021-28170', 'CVE-2021-28799', 'CVE-2021-30299', 'CVE-2021-3031', 'CVE-2021-3038', 'CVE-2021-3044', 'CVE-2021-3046', 'CVE-2021-3048', 'CVE-2021-3049', 'CVE-2021-3062', 'CVE-2021-31403', 'CVE-2021-31404', 'CVE-2021-31406', 'CVE-2021-3155', 'CVE-2021-31566', 'CVE-2021-31917', 'CVE-2021-32029', 'CVE-2021-32039', 'CVE-2021-32076', 'CVE-2021-32648', 'CVE-2021-32706', 'CVE-2021-32819', 'CVE-2021-33592', 'CVE-2021-33593', 'CVE-2021-33723', 'CVE-2021-33727', 'CVE-2021-3426', 'CVE-2021-34401', 'CVE-2021-34429', 'CVE-2021-3452', 'CVE-2021-34571', 'CVE-2021-34577', 'CVE-2021-34588', 'CVE-2021-34589', 'CVE-2021-3490', 'CVE-2021-3493', 'CVE-2021-3503', 'CVE-2021-3513', 'CVE-2021-35221', 'CVE-2021-35223', 'CVE-2021-35232', 'CVE-2021-35245', 'CVE-2021-35247', 'CVE-2021-35248', 'CVE-2021-35249', 'CVE-2021-35254', 'CVE-2021-3572', 'CVE-2021-3580', 'CVE-2021-3585', 'CVE-2021-35937', 'CVE-2021-35938', 'CVE-2021-35939', 'CVE-2021-3602', 'CVE-2021-3612', 'CVE-2021-3631', 'CVE-2021-3675', 'CVE-2021-3677', 'CVE-2021-3688', 'CVE-2021-3701', 'CVE-2021-3714', 'CVE-2021-37147', 'CVE-2021-3722', 'CVE-2021-3732', 'CVE-2021-3736', 'CVE-2021-37533', 'CVE-2021-37580', 'CVE-2021-37701', 'CVE-2021-37712', 'CVE-2021-3773', 'CVE-2021-3784', 'CVE-2021-3798', 'CVE-2021-3800', 'CVE-2021-3827', 'CVE-2021-38454', 'CVE-2021-3849', 'CVE-2021-3856', 'CVE-2021-3864', 'CVE-2021-3897', 'CVE-2021-39008', 'CVE-2021-39144', 'CVE-2021-3917', 'CVE-2021-39226', 'CVE-2021-39327', 'CVE-2021-3970', 'CVE-2021-3981', 'CVE-2021-3995', 'CVE-2021-3996', 'CVE-2021-4037', 'CVE-2021-40390', 'CVE-2021-40404', 'CVE-2021-40405', 'CVE-2021-40413', 'CVE-2021-40416', 'CVE-2021-40423', 'CVE-2021-40690', 'CVE-2021-41079', 'CVE-2021-4112', 'CVE-2021-41124', 'CVE-2021-4120', 'CVE-2021-4125', 'CVE-2021-41277', 'CVE-2021-4135', 'CVE-2021-4142', 'CVE-2021-41769', 'CVE-2021-41807', 'CVE-2021-4197', 'CVE-2021-41992', 'CVE-2021-4210', 'CVE-2021-4211', 'CVE-2021-43588', 'CVE-2021-43762', 'CVE-2021-43767', 'CVE-2021-43937', 'CVE-2021-43938', 'CVE-2021-43939', 'CVE-2021-43999', 'CVE-2021-44040', 'CVE-2021-44228', 'CVE-2021-4428', 'CVE-2021-44354', 'CVE-2021-44419', 'CVE-2021-44467', 'CVE-2021-44521', 'CVE-2021-44529', 'CVE-2021-44739', 'CVE-2021-44759', 'CVE-2021-44776', 'CVE-2021-44832', 'CVE-2021-45029', 'CVE-2021-45074', 'CVE-2021-45105', 'CVE-2021-46270', 'CVE-2021-46279', 'CVE-2022-0019', 'CVE-2022-0026', 'CVE-2022-0027', 'CVE-2022-0029', 'CVE-2022-0030', 'CVE-2022-0073', 'CVE-2022-0155', 'CVE-2022-0217', 'CVE-2022-0336', 'CVE-2022-0492', 'CVE-2022-0494', 'CVE-2022-0564', 'CVE-2022-0567', 'CVE-2022-0656', 'CVE-2022-0708', 'CVE-2022-0715', 'CVE-2022-0718', 'CVE-2022-0811', 'CVE-2022-0813', 'CVE-2022-0850', 'CVE-2022-0851', 'CVE-2022-0852', 'CVE-2022-0862', 'CVE-2022-0910', 'CVE-2022-0987', 'CVE-2022-1117', 'CVE-2022-1159', 'CVE-2022-1353', 'CVE-2022-1471', 'CVE-2022-1596', 'CVE-2022-1655', 'CVE-2022-1833', 'CVE-2022-2031', 'CVE-2022-2048', 'CVE-2022-20630', 'CVE-2022-20658', 'CVE-2022-20660', 'CVE-2022-20680', 'CVE-2022-20696', 'CVE-2022-20698', 'CVE-2022-20733', 'CVE-2022-20742', 'CVE-2022-20745', 'CVE-2022-20748', 'CVE-2022-20762', 'CVE-2022-20777', 'CVE-2022-20779', 'CVE-2022-20780', 'CVE-2022-20797', 'CVE-2022-20798', 'CVE-2022-20805', 'CVE-2022-21194', 'CVE-2022-21196', 'CVE-2022-21686', 'CVE-2022-21712', 'CVE-2022-21800', 'CVE-2022-21820', 'CVE-2022-21831', 'CVE-2022-2191', 'CVE-2022-22110', 'CVE-2022-22141', 'CVE-2022-22148', 'CVE-2022-22151', 'CVE-2022-22241', 'CVE-2022-22243', 'CVE-2022-22282', 'CVE-2022-22521', 'CVE-2022-22537', 'CVE-2022-22538', 'CVE-2022-22539', 'CVE-2022-22545', 'CVE-2022-22553', 'CVE-2022-22576', 'CVE-2022-22722', 'CVE-2022-22806', 'CVE-2022-22807', 'CVE-2022-22813', 'CVE-2022-22947', 'CVE-2022-22963', 'CVE-2022-22965', 'CVE-2022-22990', 'CVE-2022-22998', 'CVE-2022-23008', 'CVE-2022-23014', 'CVE-2022-23019', 'CVE-2022-23067', 'CVE-2022-23131', 'CVE-2022-23134', 'CVE-2022-23402', 'CVE-2022-23469', 'CVE-2022-23634', 'CVE-2022-23635', 'CVE-2022-23648', 'CVE-2022-23711', 'CVE-2022-2394', 'CVE-2022-24044', 'CVE-2022-24075', 'CVE-2022-24086', 'CVE-2022-24112', 'CVE-2022-24280', 'CVE-2022-24385', 'CVE-2022-24398', 'CVE-2022-24663', 'CVE-2022-24664', 'CVE-2022-24665', 'CVE-2022-24720', 'CVE-2022-24731', 'CVE-2022-24734', 'CVE-2022-24735', 'CVE-2022-24768', 'CVE-2022-24769', 'CVE-2022-24853', 'CVE-2022-24866', 'CVE-2022-24867', 'CVE-2022-24881', 'CVE-2022-24882', 'CVE-2022-24885', 'CVE-2022-24886', 'CVE-2022-24972', 'CVE-2022-2503', 'CVE-2022-25167', 'CVE-2022-25246', 'CVE-2022-25248', 'CVE-2022-2551', 'CVE-2022-25823', 'CVE-2022-25824', 'CVE-2022-25825', 'CVE-2022-25943', 'CVE-2022-25989', 'CVE-2022-26071', 'CVE-2022-26106', 'CVE-2022-26107', 'CVE-2022-26109', 'CVE-2022-26138', 'CVE-2022-26336', 'CVE-2022-26340', 'CVE-2022-26531', 'CVE-2022-2663', 'CVE-2022-26672', 'CVE-2022-26839', 'CVE-2022-26851', 'CVE-2022-26854', 'CVE-2022-26855', 'CVE-2022-26865', 'CVE-2022-2735', 'CVE-2022-27506', 'CVE-2022-27511', 'CVE-2022-27512', 'CVE-2022-27518', 'CVE-2022-27634', 'CVE-2022-27649', 'CVE-2022-27655', 'CVE-2022-27667', 'CVE-2022-27774', 'CVE-2022-27775', 'CVE-2022-27875', 'CVE-2022-28127', 'CVE-2022-28186', 'CVE-2022-28188', 'CVE-2022-28193', 'CVE-2022-28195', 'CVE-2022-28196', 'CVE-2022-28214', 'CVE-2022-2827', 'CVE-2022-28614', 'CVE-2022-28695', 'CVE-2022-28708', 'CVE-2022-28774', 'CVE-2022-29160', 'CVE-2022-29217', 'CVE-2022-29248', 'CVE-2022-29263', 'CVE-2022-29417', 'CVE-2022-29479', 'CVE-2022-29613', 'CVE-2022-29813', 'CVE-2022-29814', 'CVE-2022-29815', 'CVE-2022-29818', 'CVE-2022-29819', 'CVE-2022-29821', 'CVE-2022-29901', 'CVE-2022-30115', 'CVE-2022-30556', 'CVE-2022-30670', 'CVE-2022-30730', 'CVE-2022-31024', 'CVE-2022-31036', 'CVE-2022-31042', 'CVE-2022-31043', 'CVE-2022-31055', 'CVE-2022-31083', 'CVE-2022-31162', 'CVE-2022-31248', 'CVE-2022-3140', 'CVE-2022-31465', 'CVE-2022-31479', 'CVE-2022-31480', 'CVE-2022-31485', 'CVE-2022-31619', 'CVE-2022-31629', 'CVE-2022-3171', 'CVE-2022-3180', 'CVE-2022-32143', 'CVE-2022-32144', 'CVE-2022-32154', 'CVE-2022-32158', 'CVE-2022-32169', 'CVE-2022-32170', 'CVE-2022-32212', 'CVE-2022-32255', 'CVE-2022-32513', 'CVE-2022-32514', 'CVE-2022-32515', 'CVE-2022-32517', 'CVE-2022-32518', 'CVE-2022-32520', 'CVE-2022-32534', 'CVE-2022-32742', 'CVE-2022-32744', 'CVE-2022-32747', 'CVE-2022-3284', 'CVE-2022-33714', 'CVE-2022-33715', 'CVE-2022-33719', 'CVE-2022-33720', 'CVE-2022-33721', 'CVE-2022-33723', 'CVE-2022-33724', 'CVE-2022-33725', 'CVE-2022-33727', 'CVE-2022-33729', 'CVE-2022-33731', 'CVE-2022-33732', 'CVE-2022-34255', 'CVE-2022-34256', 'CVE-2022-34259', 'CVE-2022-35290', 'CVE-2022-35296', 'CVE-2022-35666', 'CVE-2022-35668', 'CVE-2022-3676', 'CVE-2022-36960', 'CVE-2022-37393', 'CVE-2022-37400', 'CVE-2022-37436', 'CVE-2022-3787', 'CVE-2022-38112', 'CVE-2022-38113', 'CVE-2022-39013', 'CVE-2022-39016', 'CVE-2022-39017', 'CVE-2022-39018', 'CVE-2022-39019', 'CVE-2022-39042', 'CVE-2022-39236', 'CVE-2022-39249', 'CVE-2022-39250', 'CVE-2022-39251', 'CVE-2022-39253', 'CVE-2022-39327', 'CVE-2022-40259', 'CVE-2022-40293', 'CVE-2022-4057', 'CVE-2022-4060', 'CVE-2022-40751', 'CVE-2022-41205', 'CVE-2022-41214', 'CVE-2022-41264', 'CVE-2022-4140', 'CVE-2022-41694', 'CVE-2022-41813', 'CVE-2022-41836', 'CVE-2022-41915', 'CVE-2022-42340', 'CVE-2022-4236', 'CVE-2022-42972', 'CVE-2022-45052', 'CVE-2022-45787', 'CVE-2022-46363', 'CVE-2022-47391', 'CVE-2022-47508', 'CVE-2022-47618', 'CVE-2022-47894', 'CVE-2022-4862', 'CVE-2022-4904', 'CVE-2023-0002', 'CVE-2023-0012', 'CVE-2023-0014', 'CVE-2023-0020', 'CVE-2023-0022', 'CVE-2023-0023', 'CVE-2023-0297', 'CVE-2023-0386', 'CVE-2023-1007', 'CVE-2023-1453', 'CVE-2023-1486', 'CVE-2023-20867', 'CVE-2023-21434', 'CVE-2023-21524', 'CVE-2023-21531', 'CVE-2023-21540', 'CVE-2023-21542', 'CVE-2023-21549', 'CVE-2023-21550', 'CVE-2023-21553', 'CVE-2023-21554', 'CVE-2023-21558', 'CVE-2023-21559', 'CVE-2023-21574', 'CVE-2023-21596', 'CVE-2023-21607', 'CVE-2023-21678', 'CVE-2023-21685', 'CVE-2023-21717', 'CVE-2023-21721', 'CVE-2023-21722', 'CVE-2023-21726', 'CVE-2023-21742', 'CVE-2023-21749', 'CVE-2023-21751', 'CVE-2023-21760', 'CVE-2023-21767', 'CVE-2023-21777', 'CVE-2023-21816', 'CVE-2023-21818', 'CVE-2023-22232', 'CVE-2023-2283', 'CVE-2023-22858', 'CVE-2023-23382', 'CVE-2023-23397', 'CVE-2023-2359', 'CVE-2023-23857', 'CVE-2023-23934', 'CVE-2023-24484', 'CVE-2023-24485', 'CVE-2023-24486', 'CVE-2023-24489', 'CVE-2023-2454', 'CVE-2023-2455', 'CVE-2023-24807', 'CVE-2023-24881', 'CVE-2023-24893', 'CVE-2023-24904', 'CVE-2023-24905', 'CVE-2023-24950', 'CVE-2023-25661', 'CVE-2023-26049', 'CVE-2023-26360', 'CVE-2023-26367', 'CVE-2023-26406', 'CVE-2023-26407', 'CVE-2023-26408', 'CVE-2023-26460', 'CVE-2023-27268', 'CVE-2023-27350', 'CVE-2023-27351', 'CVE-2023-27893', 'CVE-2023-27894', 'CVE-2023-27897', 'CVE-2023-28121', 'CVE-2023-28221', 'CVE-2023-28222', 'CVE-2023-28246', 'CVE-2023-28271', 'CVE-2023-28274', 'CVE-2023-28291', 'CVE-2023-28303', 'CVE-2023-28304', 'CVE-2023-28312', 'CVE-2023-28322', 'CVE-2023-28432', 'CVE-2023-2868', 'CVE-2023-28762', 'CVE-2023-28764', 'CVE-2023-28765', 'CVE-2023-28856', 'CVE-2023-28900', 'CVE-2023-29108', 'CVE-2023-29111', 'CVE-2023-29298', 'CVE-2023-29335', 'CVE-2023-29338', 'CVE-2023-29343', 'CVE-2023-3006', 'CVE-2023-30440', 'CVE-2023-30445', 'CVE-2023-30446', 'CVE-2023-30447', 'CVE-2023-30448', 'CVE-2023-30449', 'CVE-2023-30535', 'CVE-2023-30740', 'CVE-2023-30861', 'CVE-2023-30991', 'CVE-2023-30996', 'CVE-2023-31064', 'CVE-2023-31066', 'CVE-2023-31098', 'CVE-2023-31404', 'CVE-2023-31453', 'CVE-2023-32037', 'CVE-2023-32043', 'CVE-2023-32053', 'CVE-2023-32056', 'CVE-2023-32057', 'CVE-2023-32113', 'CVE-2023-32243', 'CVE-2023-3247', 'CVE-2023-3262', 'CVE-2023-32697', 'CVE-2023-33148', 'CVE-2023-33155', 'CVE-2023-33165', 'CVE-2023-33174', 'CVE-2023-33246', 'CVE-2023-33990', 'CVE-2023-3466', 'CVE-2023-3519', 'CVE-2023-35303', 'CVE-2023-35320', 'CVE-2023-35333', 'CVE-2023-35336', 'CVE-2023-35347', 'CVE-2023-35348', 'CVE-2023-35353', 'CVE-2023-35365', 'CVE-2023-35367', 'CVE-2023-35619', 'CVE-2023-35624', 'CVE-2023-35625', 'CVE-2023-35633', 'CVE-2023-35841', 'CVE-2023-35870', 'CVE-2023-36018', 'CVE-2023-36021', 'CVE-2023-36046', 'CVE-2023-36047', 'CVE-2023-36049', 'CVE-2023-36052', 'CVE-2023-36391', 'CVE-2023-36394', 'CVE-2023-36399', 'CVE-2023-36404', 'CVE-2023-36407', 'CVE-2023-36437', 'CVE-2023-36563', 'CVE-2023-36705', 'CVE-2023-36719', 'CVE-2023-36761', 'CVE-2023-36872', 'CVE-2023-36874', 'CVE-2023-36917', 'CVE-2023-36920', 'CVE-2023-36923', 'CVE-2023-36924', 'CVE-2023-37464', 'CVE-2023-37486', 'CVE-2023-37582', 'CVE-2023-38205', 'CVE-2023-38206', 'CVE-2023-38220', 'CVE-2023-38245', 'CVE-2023-38720', 'CVE-2023-38728', 'CVE-2023-38740', 'CVE-2023-39422', 'CVE-2023-39440', 'CVE-2023-40029', 'CVE-2023-40049', 'CVE-2023-40273', 'CVE-2023-4039', 'CVE-2023-4043', 'CVE-2023-40621', 'CVE-2023-40622', 'CVE-2023-40743', 'CVE-2023-41313', 'CVE-2023-4164', 'CVE-2023-42480', 'CVE-2023-42503', 'CVE-2023-43123', 'CVE-2023-43804', 'CVE-2023-4481', 'CVE-2023-45143', 'CVE-2023-45236', 'CVE-2023-45648', 'CVE-2023-46747', 'CVE-2023-4759', 'CVE-2023-49081', 'CVE-2023-49082', 'CVE-2023-49093', 'CVE-2023-49198', 'CVE-2023-49578', 'CVE-2023-49580', 'CVE-2023-49582', 'CVE-2023-50164', 'CVE-2023-5043', 'CVE-2023-5044', 'CVE-2023-6548', 'CVE-2024-0057', 'CVE-2024-0204', 'CVE-2024-0411', 'CVE-2024-10327', 'CVE-2024-10924', 'CVE-2024-11234', 'CVE-2024-11680', 'CVE-2024-1709', 'CVE-2024-20656', 'CVE-2024-20657', 'CVE-2024-20665', 'CVE-2024-20666', 'CVE-2024-20669', 'CVE-2024-20670', 'CVE-2024-20673', 'CVE-2024-20675', 'CVE-2024-20684', 'CVE-2024-20695', 'CVE-2024-20709', 'CVE-2024-20721', 'CVE-2024-20733', 'CVE-2024-20767', 'CVE-2024-21302', 'CVE-2024-21304', 'CVE-2024-21305', 'CVE-2024-21312', 'CVE-2024-21315', 'CVE-2024-21316', 'CVE-2024-21319', 'CVE-2024-21320', 'CVE-2024-21329', 'CVE-2024-21351', 'CVE-2024-21364', 'CVE-2024-21374', 'CVE-2024-21376', 'CVE-2024-21378', 'CVE-2024-21380', 'CVE-2024-21390', 'CVE-2024-21397', 'CVE-2024-21401', 'CVE-2024-21402', 'CVE-2024-21403', 'CVE-2024-21410', 'CVE-2024-21412', 'CVE-2024-21424', 'CVE-2024-21427', 'CVE-2024-21447', 'CVE-2024-21448', 'CVE-2024-21737', 'CVE-2024-21742', 'CVE-2024-21793', 'CVE-2024-22029', 'CVE-2024-22131', 'CVE-2024-22245', 'CVE-2024-23944', 'CVE-2024-24549', 'CVE-2024-24683', 'CVE-2024-24795', 'CVE-2024-24919', 'CVE-2024-25153', 'CVE-2024-25641', 'CVE-2024-25645', 'CVE-2024-25646', 'CVE-2024-26026', 'CVE-2024-26063', 'CVE-2024-26119', 'CVE-2024-26126', 'CVE-2024-26127', 'CVE-2024-26158', 'CVE-2024-26164', 'CVE-2024-26167', 'CVE-2024-26177', 'CVE-2024-26189', 'CVE-2024-26193', 'CVE-2024-26197', 'CVE-2024-26199', 'CVE-2024-26201', 'CVE-2024-26203', 'CVE-2024-26216', 'CVE-2024-26234', 'CVE-2024-26238', 'CVE-2024-26240', 'CVE-2024-26250', 'CVE-2024-27135', 'CVE-2024-27438', 'CVE-2024-27458', 'CVE-2024-27894', 'CVE-2024-28163', 'CVE-2024-28164', 'CVE-2024-28185', 'CVE-2024-28189', 'CVE-2024-28253', 'CVE-2024-28255', 'CVE-2024-28847', 'CVE-2024-28848', 'CVE-2024-28849', 'CVE-2024-28897', 'CVE-2024-28907', 'CVE-2024-28916', 'CVE-2024-28917', 'CVE-2024-28919', 'CVE-2024-28921', 'CVE-2024-29006', 'CVE-2024-29014', 'CVE-2024-29054', 'CVE-2024-29055', 'CVE-2024-29056', 'CVE-2024-29060', 'CVE-2024-29187', 'CVE-2024-29988', 'CVE-2024-29992', 'CVE-2024-29993', 'CVE-2024-29995', 'CVE-2024-29998', 'CVE-2024-30002', 'CVE-2024-30018', 'CVE-2024-30033', 'CVE-2024-30040', 'CVE-2024-30041', 'CVE-2024-30050', 'CVE-2024-30052', 'CVE-2024-30054', 'CVE-2024-30059', 'CVE-2024-30065', 'CVE-2024-30076', 'CVE-2024-30078', 'CVE-2024-30087', 'CVE-2024-30093', 'CVE-2024-30096', 'CVE-2024-30104', 'CVE-2024-31141', 'CVE-2024-31227', 'CVE-2024-31860', 'CVE-2024-31862', 'CVE-2024-31863', 'CVE-2024-31864', 'CVE-2024-31867', 'CVE-2024-32007', 'CVE-2024-33003', 'CVE-2024-33065', 'CVE-2024-33066', 'CVE-2024-3383', 'CVE-2024-3385', 'CVE-2024-34340', 'CVE-2024-34684', 'CVE-2024-35253', 'CVE-2024-35254', 'CVE-2024-35263', 'CVE-2024-37180', 'CVE-2024-37325', 'CVE-2024-37341', 'CVE-2024-3746', 'CVE-2024-37965', 'CVE-2024-38046', 'CVE-2024-38084', 'CVE-2024-38098', 'CVE-2024-38112', 'CVE-2024-38163', 'CVE-2024-38180', 'CVE-2024-38188', 'CVE-2024-38189', 'CVE-2024-38194', 'CVE-2024-38195', 'CVE-2024-38196', 'CVE-2024-38197', 'CVE-2024-38200', 'CVE-2024-38201', 'CVE-2024-38203', 'CVE-2024-38213', 'CVE-2024-38216', 'CVE-2024-38220', 'CVE-2024-38223', 'CVE-2024-38225', 'CVE-2024-38226', 'CVE-2024-38230', 'CVE-2024-38231', 'CVE-2024-38234', 'CVE-2024-38241', 'CVE-2024-38243', 'CVE-2024-38244', 'CVE-2024-38245', 'CVE-2024-38367', 'CVE-2024-38425', 'CVE-2024-38646', 'CVE-2024-38647', 'CVE-2024-38810', 'CVE-2024-39398', 'CVE-2024-39404', 'CVE-2024-39405', 'CVE-2024-39407', 'CVE-2024-39412', 'CVE-2024-39413', 'CVE-2024-39414', 'CVE-2024-39415', 'CVE-2024-39416', 'CVE-2024-39417', 'CVE-2024-39418', 'CVE-2024-39573', 'CVE-2024-39593', 'CVE-2024-39597', 'CVE-2024-39599', 'CVE-2024-39600', 'CVE-2024-40766', 'CVE-2024-41729', 'CVE-2024-41732', 'CVE-2024-41733', 'CVE-2024-41736', 'CVE-2024-41806', 'CVE-2024-41856', 'CVE-2024-42494', 'CVE-2024-43461', 'CVE-2024-43469', 'CVE-2024-43470', 'CVE-2024-43479', 'CVE-2024-43482', 'CVE-2024-43487', 'CVE-2024-43492', 'CVE-2024-4358', 'CVE-2024-43602', 'CVE-2024-43645', 'CVE-2024-44113', 'CVE-2024-45195', 'CVE-2024-45216', 'CVE-2024-45219', 'CVE-2024-46901', 'CVE-2024-47076', 'CVE-2024-47175', 'CVE-2024-47592', 'CVE-2024-48862', 'CVE-2024-49039', 'CVE-2024-49040', 'CVE-2024-49044', 'CVE-2024-49048', 'CVE-2024-49049', 'CVE-2024-49051', 'CVE-2024-49056', 'CVE-2024-49328', 'CVE-2024-49604', 'CVE-2024-5082', 'CVE-2024-5535', 'CVE-2024-56326', 'CVE-2024-5806', 'CVE-2024-5913', 'CVE-2024-6385', 'CVE-2024-6915', 'CVE-2024-7014', 'CVE-2024-7254', 'CVE-2024-7593', 'CVE-2024-8260', 'CVE-2024-9264', 'CVE-2024-9486', 'CVE-2025-0064', 'CVE-2025-0411', 'CVE-2025-0683', 'CVE-2025-1097', 'CVE-2025-1098', 'CVE-2025-21293', 'CVE-2025-23209', 'CVE-2025-24061', 'CVE-2025-24071', 'CVE-2025-24408', 'CVE-2025-24411', 'CVE-2025-24418', 'CVE-2025-24422', 'CVE-2025-24424', 'CVE-2025-24426', 'CVE-2025-24427', 'CVE-2025-24429', 'CVE-2025-24435', 'CVE-2025-24437', 'CVE-2025-24513', 'CVE-2025-24514', 'CVE-2025-24876', 'CVE-2025-24970', 'CVE-2025-24989', 'CVE-2025-26658', 'CVE-2025-2746', 'CVE-2025-2747', 'CVE-2025-29927']
    #     }

    import json

    # Read evidence from JSON file
    with open('../query_input.json', 'r') as f:
        evidence = json.load(f)

    
    # Process evidence
    print("Processing evidence...")
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
    
    print("Evidence processed:")
    print({k: len(v) for k, v in encoded_evidence.items()})
    
    # Calculate similarities using Random Forest
    print("Calculating similarities...")
    evidence_vector = create_evidence_vector(encoded_evidence, binarizers, all_features, feature_types)
    rf_similarities = calculate_rf_similarity(evidence_vector, rf, X, apt_names)
    
    # Create results
    full_results = []
    sorted_indices = np.argsort(rf_similarities)[::-1]
    
    for idx in sorted_indices:
        apt_name = apt_names[idx]
        apt_data = apt_profiles[apt_name]
        
        common_features = {
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve'])),
        }
        
        full_results.append({
            'rank': len(full_results)+1,
            'APT': apt_name_dict[apt_data['encoded_id']],
            'Score': rf_similarities[idx],
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })
    
    # PRINT RF RESULTS TABLE (ALL APTs)
    print_rf_results_table(full_results, encoded_evidence, apt_profiles, apt_name_dict)
    # PERFORM FEATURE MATCHING CLASSIFICATION ANALYSIS
    feature_scores = analyze_feature_matching_classification(
        encoded_evidence, 
        full_results, 
        apt_profiles, 
        apt_name_dict
    )
    
    print(f"\n RANDOM FOREST QUERY COMPLETE!")
    print(f" Most similar: {full_results[0]['APT']} (score: {full_results[0]['Score']:.4f})")
    print(f" Least similar: {full_results[-1]['APT']} (score: {full_results[-1]['Score']:.4f})")
    print(f" Total APTs ranked: {len(full_results)}")
    print(f" Using Random Forest with leaf similarity")

if __name__ == "__main__":
    main()

