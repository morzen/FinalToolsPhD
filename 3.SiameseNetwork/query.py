import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def harmonic_mean(a, b):
    if a + b == 0: return 0.0
    return 2 * a * b / (a + b)

# ---------- FeatureEncoder (unchanged) ----------
class FeatureEncoder(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwe, num_cve, num_pulse, embed_dim=128):
        super().__init__()
        embed_per_feature = embed_dim // 6
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature)
        self.capec_embed = nn.Embedding(num_capec+1, embed_per_feature)
        self.cwe_embed = nn.Embedding(num_cwe+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cve+1, embed_per_feature)
        self.pulse_embed = nn.Embedding(num_pulse+1, embed_per_feature)
        total_embed_size = embed_per_feature * 6
        self.fc1 = nn.Linear(total_embed_size, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech, soft, capec, cwe, cve, pulse = features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4], features[:, 5]
        t_emb = self.tech_embed(tech.long() + 1)
        s_emb = self.soft_embed(soft.long() + 1)
        ca_emb = self.capec_embed(capec.long() + 1)
        cw_emb = self.cwe_embed(cwe.long() + 1)
        cv_emb = self.cve_embed(cve.long() + 1)
        P_emb = self.pulse_embed(pulse.long() + 1)
        x = torch.cat([t_emb, s_emb, ca_emb, cw_emb, cv_emb, P_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

# ---------- Load everything ----------
def load_model_and_data(status):

    if status == "hybrid":
        with open("../3.SiameseNetwork/6feat_apt_siamese_feature_model.pkl", 'rb') as f:
            data = pickle.load(f)
    else:
        with open("6feat_apt_siamese_feature_model.pkl", 'rb') as f:
            data = pickle.load(f)
    
    config = data['model_config']
    model = FeatureEncoder(config['num_techniques'], config['num_software'], config['num_capec'], config['num_cwe'], config['num_cve'], config['num_pulse'], config['embedding_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    apt_embeddings = data['apt_embeddings'].float()
    apt_ids = data['apt_ids']
    apt_to_idx = data['apt_to_idx']
    return model, data, apt_embeddings, apt_ids, apt_to_idx

def load_mapping_and_profiles():
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    df_encoded = pd.read_csv("../BaseData/EncodedData.csv")
    
    # Build APT profiles
    profiles = {}
    for apt_id in df_encoded['APTname'].unique():
        apt_data = df_encoded[df_encoded['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software': set(apt_data['software_str'].dropna().unique()), 
            'capec': set(apt_data['CAPEC_ID'].dropna().unique()),
            'cwe': set(apt_data['CWE_ID'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'pulse': set(apt_data['Pulse_ID'].dropna().unique()),
            'total_features': len(apt_data)
        }
    
    # APT name mapping
    apt_name_mapping = dict(zip(df_mapping['APTnameEncoded'], df_mapping['APTnameOrigin'].fillna(df_mapping['APTnameEncoded'])))
    
    print(f" Loaded profiles for {len(profiles)} APTs")
    return df_mapping, profiles, apt_name_mapping

# ---------- Evidence processing (XGBoost-style) ----------
def process_evidence(evidence, df_mapping):
    print(" CONVERSION RESULTS:")
    
    column_pairs = {
        'techniques': ('techniques_Origin', 'techniques_Encoded', 'techniques'),
        'software': ('software_Origin', 'software_Encoded', 'software'),
        'capec': ('CAPEC_ID_Origin', 'CAPEC_ID_Encoded', 'capec'),
        'cwe': ('CWE_ID_Origin', 'CWE_ID_Encoded', 'cwe'),
        'cve': ('CVE_ID_Origin', 'CVE_ID_Encoded', 'cve'),
        'pulse': ('Pulse_ID_Origin', 'Pulse_ID_Encoded', 'pulse')
    }
    
    encoded_evidence = {}
    for evidence_key, (origin_col, encoded_col, output_key) in column_pairs.items():
        if evidence_key in evidence:
            print(f"   Input {output_key}: {len(evidence[evidence_key])} items")
            mapping = dict(zip(df_mapping[origin_col].astype(str), df_mapping[encoded_col]))
            encoded_values = []
            for val in evidence[evidence_key]:
                encoded_id = mapping.get(str(val), None)
                if encoded_id is not None and pd.notna(encoded_id):
                    try:
                        encoded_values.append(int(encoded_id))
                    except:
                        pass
            encoded_evidence[output_key] = encoded_values
            print(f"   → Encoded {output_key}: {len(encoded_values)} items")
    
    return encoded_evidence

# ---------- Siamese similarity + feature matching ----------
def query_siamese_with_features(model, apt_embeddings, encoded_evidence, apt_ids, profiles, apt_name_mapping):
    model.eval()
    
    # Siamese embedding similarity
    evidence_vectors = []
    n_samples = 50
    for _ in range(n_samples):
        tech = np.random.choice(encoded_evidence.get('techniques', [-1]))
        soft = np.random.choice(encoded_evidence.get('software', [-1]))
        capec = np.random.choice(encoded_evidence.get('capec', [-1]))
        cwe = np.random.choice(encoded_evidence.get('cwe', [-1]))
        cve = np.random.choice(encoded_evidence.get('cve', [-1]))
        pulse = np.random.choice(encoded_evidence.get('pulse', [-1]))
        evidence_vectors.append([float(tech), float(soft), float(capec), float(cwe), float(cve), float(pulse)])
    
    evidence_vectors = torch.tensor(evidence_vectors, dtype=torch.float32)
    evidence_emb = model(evidence_vectors).mean(dim=0)
    similarities = F.cosine_similarity(evidence_emb.unsqueeze(0), apt_embeddings, dim=1).detach().cpu().numpy()
    
    # Feature matching (XGBoost-style)
    results = []
    for idx, apt_id in enumerate(apt_ids):
        profile = profiles.get(apt_id, {})
        tech_matches = len(set(encoded_evidence.get('techniques', [])) & profile.get('techniques', set()))
        soft_matches = len(set(encoded_evidence.get('software', [])) & profile.get('software', set()))
        capec_matches = len(set(encoded_evidence.get('capec', [])) & profile.get('capec', set()))
        cwe_matches = len(set(encoded_evidence.get('cwe', [])) & profile.get('cwe', set()))
        cve_matches = len(set(encoded_evidence.get('cve', [])) & profile.get('cve', set()))
        pulse_matches = len(set(encoded_evidence.get('pulse', [])) & profile.get('pulse', set()))
        total_matches = tech_matches + soft_matches + capec_matches + cwe_matches + cve_matches + pulse_matches
        
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile_size = f"{len(profile.get('techniques',[])):.0f}T/{len(profile.get('software',[])):.0f}S/{len(profile.get('capec',[])):.0f}Ca/{len(profile.get('cwe',[])):.0f}Cw/{len(profile.get('cve',[])):.0f}Cv/{len(profile.get('pulse',[])):.0f}P"
        
        results.append({
            'apt_id': apt_id,
            'apt_name': apt_name,
            'similarity': similarities[idx],
            'tech_matches': tech_matches,
            'soft_matches': soft_matches, 
            'capec_matches': capec_matches,
            'cwe_matches': cwe_matches,
            'cve_matches': cve_matches,
            'pulse_matches': pulse_matches,
            'total_matches': total_matches,
            'profile_size': profile_size,
            'profile_total': len(profile.get('techniques',[])) + len(profile.get('software',[])) + len(profile.get('capec',[])) + len(profile.get('cwe',[])) + len(profile.get('cve',[])) + len(profile.get('pulse',[]))
        })
    
    # Sort by similarity and add rank
    results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
    for i, result in enumerate(results_sorted, 1):
        result['rank'] = i
    
    return results_sorted

# ---------- XGBoost-style output functions ----------
def print_results_table(results, encoded_evidence):
    print("\n QUERY EVIDENCE:")
    print(f"   Techniques: {len(encoded_evidence.get('techniques', []))} items")
    print(f"   Software: {len(encoded_evidence.get('software', []))} items")
    print(f"   CAPEC: {len(encoded_evidence.get('capec', []))} items")
    print(f"   cwe: {len(encoded_evidence.get('cwe', []))} items")
    print(f"   cve: {len(encoded_evidence.get('cve', []))} items")
    print(f"   Pulse: {len(encoded_evidence.get('pulse', []))} items")
    
    print("\n SIAMESE NETWORK SIMILARITY RANKINGS")
    print(" Features: FeatureEncoder embeddings + Feature overlap")
    print("=" * 160)
    
    # Show ALL APTs, not just top 50
    header = f"{'Rank':<4} {'APT Name':<28} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CAPEC':<8} {'CWE':<8} {'CVE':<8} {'Pulse':<8} {'Matches':<8} {'Profile'}"
    print(header)
    print("-" * 160)
    
    # Get evidence lengths for each feature
    tech_len = len(encoded_evidence.get('techniques', []))
    soft_len = len(encoded_evidence.get('software', []))
    capec_len = len(encoded_evidence.get('capec', []))
    cwe_len = len(encoded_evidence.get('cwe', []))
    cve_len = len(encoded_evidence.get('cve', []))
    pulse_len = len(encoded_evidence.get('pulse', []))

    for i, result in enumerate(results, 1):  # REMOVE [:50] to show all
        tech_str = f"{result['tech_matches']}/{tech_len}"
        soft_str = f"{result['soft_matches']}/{soft_len}"
        capec_str = f"{result['capec_matches']}/{capec_len}"
        cwe_str = f"{result['cwe_matches']}/{cwe_len}"
        cve_str = f"{result['cve_matches']}/{cve_len}"
        pulse_str = f"{result['pulse_matches']}/{pulse_len}"
        
        # Format with ALL 6 features correctly
        row = (f"{i:<4} {result['apt_name']:<28} {result['similarity']:<10.4f} "
               f"{tech_str:<8} {soft_str:<8} {capec_str:<8} {cwe_str:<8} "
               f"{cve_str:<8} {pulse_str:<8} {result['total_matches']:<10} {result['profile_size']}")
        print(row)
    
    print("-" * 160)

def analyze_feature_matching(results, encoded_evidence, profiles):
    print("\n" + "="*80)
    print(" FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    all_feature_scores = []
    
    # Calculate evidence lengths for each category
    tech_len = len(encoded_evidence.get('techniques', []))
    soft_len = len(encoded_evidence.get('software', []))
    capec_len = len(encoded_evidence.get('capec', []))
    cwe_len = len(encoded_evidence.get('cwe', []))
    cve_len = len(encoded_evidence.get('cve', []))
    pulse_len = len(encoded_evidence.get('pulse', []))
    
    # Create evidence length dictionary
    evid_lengths = {
        'tech_score': tech_len,
        'soft_score': soft_len,
        'capec_score': capec_len,
        'cwe_score': cwe_len,
        'cve_score': cve_len,
        'pulse_score': pulse_len
    }
    
    for result in results:
        apt_id = result['apt_id']
        profile = profiles.get(apt_id, {})
        
        tech_matches = len(set(encoded_evidence.get('techniques', [])) & profile.get('techniques', set()))
        soft_matches = len(set(encoded_evidence.get('software', [])) & profile.get('software', set()))
        capec_matches = len(set(encoded_evidence.get('capec', [])) & profile.get('capec', set()))
        cwe_matches = len(set(encoded_evidence.get('cwe', [])) & profile.get('cwe', set()))
        cve_matches = len(set(encoded_evidence.get('cve', [])) & profile.get('cve', set()))
        pulse_matches = len(set(encoded_evidence.get('pulse', [])) & profile.get('pulse', set()))
        
        # Harmonic mean scores
        def harmonic(a, b):
            return 2*a*b/(a+b) if a+b > 0 else 0
        
        tech_score = harmonic(tech_matches/max(1,tech_len), tech_matches/max(1,len(profile.get('techniques',[]))))
        soft_score = harmonic(soft_matches/max(1,soft_len), soft_matches/max(1,len(profile.get('software',[]))))
        capec_score = harmonic(capec_matches/max(1,capec_len), capec_matches/max(1,len(profile.get('capec',[]))))
        cwe_score = harmonic(cwe_matches/max(1,cwe_len), cwe_matches/max(1,len(profile.get('cwe',[]))))
        cve_score = harmonic(cve_matches/max(1,cve_len), cve_matches/max(1,len(profile.get('cve',[]))))
        pulse_score = harmonic(pulse_matches/max(1,pulse_len), pulse_matches/max(1,len(profile.get('pulse',[]))))
        
        all_feature_scores.append({
            'apt_name': result['apt_name'],
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

            'tech_evid_cov': tech_matches/max(1,tech_len),
            'soft_evid_cov': soft_matches/max(1,soft_len),
            'capec_evid_cov': capec_matches/max(1,capec_len),
            'cwe_evid_cov': cwe_matches/max(1,cwe_len),
            'cve_evid_cov': cve_matches/max(1,cve_len),
            'pulse_evid_cov': pulse_matches/max(1,pulse_len),
            
            'tech_prof_cov': tech_matches/max(1,len(profile.get('techniques',[]))),
            'soft_prof_cov': soft_matches/max(1,len(profile.get('software',[]))),
            'capec_prof_cov': capec_matches/max(1,len(profile.get('capec',[]))),
            'cwe_prof_cov': cwe_matches/max(1,len(profile.get('cwe',[]))),
            'cve_prof_cov': cve_matches/max(1,len(profile.get('cve',[]))),
            'pulse_prof_cov': pulse_matches/max(1,len(profile.get('pulse',[])))
        })
    
    # Print top 5 by category with ACTUAL NUMBERS (not "...")
    categories = [
        ('TTP TOP 5 APTs BY TTP MATCHING', 'tech_score', 'tech_matches', 'tech_evid_cov', 'tech_prof_cov'),
        ('Software TOP 5 APTs BY Software MATCHING', 'soft_score', 'soft_matches', 'soft_evid_cov', 'soft_prof_cov'),
        ('CAPEC TOP 5 APTs BY CAPEC MATCHING', 'capec_score', 'capec_matches', 'capec_evid_cov', 'capec_prof_cov'),
        ('CWE TOP 5 APTs BY CWE MATCHING', 'cwe_score', 'cwe_matches', 'cwe_evid_cov', 'cwe_prof_cov'),
        ('CVE TOP 5 APTs BY CVE MATCHING', 'cve_score', 'cve_matches', 'cve_evid_cov', 'cve_prof_cov'),
        ('Pulse TOP 5 APTs BY Pulse MATCHING', 'pulse_score', 'pulse_matches', 'pulse_evid_cov', 'pulse_prof_cov')
    ]

    for cat_name, score_key, matches_key, evid_key, prof_key in categories:
        print(f"\n{cat_name}")
        print("-" * 85)
        header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<16} {'Profile':<15} {'Harmonic':<10}"
        print(header)
        print("-" * 85)
        
        top5 = sorted(all_feature_scores, key=lambda x: x[score_key], reverse=True)[:5]
        for i, score in enumerate(top5, 1):
            apt_name = score['apt_name'][:29] + "..." if len(score['apt_name']) > 29 else score['apt_name']
            
            # Get the correct evidence length for this category
            evid_len = evid_lengths[score_key]
            
            # Calculate coverage values
            evidence_cov = f"{score[matches_key]}/{evid_len} ({score[evid_key]:.3f})"
            
            # For profile coverage, we need the actual profile total from the original results
            # We'll use the matches as numerator and the profile total from the results
            profile_total = 0
            for r in results:
                if r['apt_name'] == score['apt_name']:
                    # Extract profile total from profile_size string
                    try:
                        # The profile_size format is "16T/1S/12Ca/13Cw/223Cv/282P"
                        # We need to parse this to get the total
                        parts = r['profile_size'].split('/')
                        total = 0
                        for part in parts:
                            if 'T' in part:
                                total += int(part.replace('T', ''))
                            elif 'S' in part:
                                total += int(part.replace('S', ''))
                            elif 'Ca' in part:
                                total += int(part.replace('Ca', ''))
                            elif 'Cw' in part:
                                total += int(part.replace('Cw', ''))
                            elif 'Cv' in part:
                                total += int(part.replace('Cv', ''))
                            elif 'P' in part:
                                total += int(part.replace('P', ''))
                        profile_total = total
                    except:
                        profile_total = score[matches_key] + 1  # fallback
                    break
            
            profile_cov = f"{score[matches_key]}/{profile_total} ({score[prof_key]:.3f})"
            
            print(f"{i:<4} {apt_name:<30} {score[score_key]:<10.4f} {evidence_cov:<16} {profile_cov:<15} {score[score_key]:<10.4f}")


def hybrid_query(evidence):

    model, data, apt_embeddings, apt_ids, apt_to_idx = load_model_and_data("hybrid")
    df_mapping, profiles, apt_name_mapping = load_mapping_and_profiles()

    encoded_evidence = process_evidence(evidence, df_mapping)
    
    results = query_siamese_with_features(model, apt_embeddings, encoded_evidence, apt_ids, profiles, apt_name_mapping)

    return results
    

# ---------- MAIN ----------
def main():
    print(" RUNNING SIAMESE FEATURE ENCODER APT SIMILARITY QUERY")
    print("="*70)
    print(" FEATURES: FeatureEncoder embeddings + Feature overlap")
    print("="*70)
    
    # Load everything
    model, data, apt_embeddings, apt_ids, apt_to_idx = load_model_and_data("nonhybrid")
    df_mapping, profiles, apt_name_mapping = load_mapping_and_profiles()
    
    # Your FULL evidence
    # evidence = {
    #     'techniques': ['T1001.002', 'T1003.002', 'T1003.004', 'T1003.006', 'T1005', 'T1016.001', 'T1018', 'T1021.001', 'T1021.002', 'T1021.006', 'T1021.007', 'T1027.001', 'T1027.002', 'T1027.003', 'T1027.006', 'T1036.004', 'T1036.005', 'T1037', 'T1037.004', 'T1047', 'T1048.002', 'T1053.005', 'T1057', 'T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1059.009', 'T1068', 'T1069.002', 'T1070.004', 'T1070.006', 'T1070.008', 'T1071.001', 'T1074.002', 'T1078', 'T1078.002', 'T1078.003', 'T1078.004', 'T1082', 'T1083', 'T1087.002', 'T1087.004', 'T1090.001', 'T1090.002', 'T1090.003', 'T1090.004', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.005', 'T1102.002', 'T1105', 'T1110.001', 'T1110.003', 'T1114.002', 'T1133', 'T1136.003', 'T1140', 'T1190', 'T1195.002', 'T1199', 'T1203', 'T1204.001', 'T1204.002', 'T1213', 'T1213.003', 'T1218.005', 'T1218.011', 'T1482', 'T1484.002', 'T1505.003', 'T1528', 'T1539', 'T1546.003', 'T1546.008', 'T1547.001', 'T1548.002', 'T1550.001', 'T1550.003', 'T1550.004', 'T1552.004', 'T1553.002', 'T1553.005', 'T1555.003', 'T1556.007', 'T1558.003', 'T1560.001', 'T1562.001', 'T1562.002', 'T1562.004', 'T1562.008', 'T1566.001', 'T1566.002', 'T1566.003', 'T1568', 'T1573', 'T1583.001', 'T1583.006', 'T1584.001', 'T1585.001', 'T1586.002', 'T1586.003', 'T1587.001', 'T1587.003', 'T1588.002', 'T1589.001', 'T1595.002', 'T1606.001', 'T1606.002', 'T1621', 'T1649', 'T1651', 'T1665'],
    #     'software': ['S0002', 'S0029', 'S0037', 'S0039', 'S0046', 'S0048', 'S0049', 'S0050', 'S0051', 'S0052', 'S0053', 'S0054', 'S0057', 'S0096', 'S0100', 'S0139', 'S0150', 'S0154', 'S0175', 'S0183', 'S0195', 'S0357', 'S0511', 'S0512', 'S0513', 'S0514', 'S0515', 'S0516', 'S0518', 'S0521', 'S0552', 'S0559', 'S0560', 'S0562', 'S0565', 'S0588', 'S0589', 'S0597', 'S0633', 'S0634', 'S0635', 'S0636', 'S0637', 'S0661', 'S0677', 'S0682', 'S0684', 'S1084'],
    #     'capec': ['CAPEC-127', 'CAPEC-150', 'CAPEC-163', 'CAPEC-186', 'CAPEC-204', 'CAPEC-206', 'CAPEC-21', 'CAPEC-268', 'CAPEC-292', 'CAPEC-31', 'CAPEC-312', 'CAPEC-313', 'CAPEC-35', 'CAPEC-37', 'CAPEC-443', 'CAPEC-464', 'CAPEC-465', 'CAPEC-473', 'CAPEC-474', 'CAPEC-481', 'CAPEC-485', 'CAPEC-49', 'CAPEC-497', 'CAPEC-504', 'CAPEC-509', 'CAPEC-523', 'CAPEC-542', 'CAPEC-543', 'CAPEC-545', 'CAPEC-555', 'CAPEC-558', 'CAPEC-560', 'CAPEC-561', 'CAPEC-564', 'CAPEC-565', 'CAPEC-571', 'CAPEC-572', 'CAPEC-573', 'CAPEC-578', 'CAPEC-580', 'CAPEC-593', 'CAPEC-60', 'CAPEC-616', 'CAPEC-639', 'CAPEC-645', 'CAPEC-647', 'CAPEC-650', 'CAPEC-655', 'CAPEC-669', 'CAPEC-673', 'CAPEC-678', 'CAPEC-68', 'CAPEC-691', 'CAPEC-70'],
    #     'cwe': ['CWE-1021', 'CWE-113', 'CWE-117', 'CWE-1258', 'CWE-1278', 'CWE-20', 'CWE-200', 'CWE-204', 'CWE-208', 'CWE-226', 'CWE-257', 'CWE-270', 'CWE-272', 'CWE-276', 'CWE-282', 'CWE-284', 'CWE-285', 'CWE-287', 'CWE-288', 'CWE-290', 'CWE-294', 'CWE-302', 'CWE-307', 'CWE-311', 'CWE-312', 'CWE-318', 'CWE-325', 'CWE-327', 'CWE-330', 'CWE-346', 'CWE-359', 'CWE-384', 'CWE-425', 'CWE-441', 'CWE-451', 'CWE-472', 'CWE-488', 'CWE-494', 'CWE-521', 'CWE-522', 'CWE-524', 'CWE-539', 'CWE-552', 'CWE-59', 'CWE-602', 'CWE-664', 'CWE-693', 'CWE-732', 'CWE-798', 'CWE-923', 'CWE-94'],
    #     'cve': ['CVE-2012-0433', 'CVE-2013-10007', 'CVE-2016-2124', 'CVE-2016-8612', 'CVE-2017-12233', 'CVE-2017-12234', 'CVE-2017-12235', 'CVE-2017-12240', 'CVE-2017-12319', 'CVE-2017-3167', 'CVE-2017-5174', 'CVE-2017-7501', 'CVE-2017-7517', 'CVE-2017-7924', 'CVE-2018-0125', 'CVE-2018-0147', 'CVE-2018-0158', 'CVE-2018-0159', 'CVE-2018-0171', 'CVE-2018-0172', 'CVE-2018-0173', 'CVE-2018-0174', 'CVE-2018-0296', 'CVE-2018-10863', 'CVE-2018-10867', 'CVE-2018-1270', 'CVE-2018-1273', 'CVE-2018-14667', 'CVE-2018-15454', 'CVE-2018-5383', 'CVE-2019-10938', 'CVE-2019-12643', 'CVE-2019-14840', 'CVE-2019-14866', 'CVE-2019-15976', 'CVE-2019-1622', 'CVE-2019-1652', 'CVE-2019-1653', 'CVE-2019-1821', 'CVE-2019-18998', 'CVE-2019-3738', 'CVE-2019-7609', 'CVE-2020-10148', 'CVE-2020-10684', 'CVE-2020-10693', 'CVE-2020-10754', 'CVE-2020-12145', 'CVE-2020-13942', 'CVE-2020-14343', 'CVE-2020-15250', 'CVE-2020-15256', 'CVE-2020-16096', 'CVE-2020-16098', 'CVE-2020-16099', 'CVE-2020-16102', 'CVE-2020-1747', 'CVE-2020-17519', 'CVE-2020-1926', 'CVE-2020-2000', 'CVE-2020-2022', 'CVE-2020-2035', 'CVE-2020-2050', 'CVE-2020-2506', 'CVE-2020-25662', 'CVE-2020-25717', 'CVE-2020-26291', 'CVE-2020-27218', 'CVE-2020-27833', 'CVE-2020-27836', 'CVE-2020-3161', 'CVE-2020-3259', 'CVE-2020-3452', 'CVE-2020-3495', 'CVE-2020-35167', 'CVE-2020-35168', 'CVE-2020-35509', 'CVE-2020-36197', 'CVE-2020-36319', 'CVE-2020-5258', 'CVE-2020-5259', 'CVE-2020-5396', 'CVE-2020-5412', 'CVE-2020-5414', 'CVE-2020-5415', 'CVE-2020-5417', 'CVE-2020-5425', 'CVE-2020-7069', 'CVE-2020-7070', 'CVE-2020-7943', 'CVE-2020-8193', 'CVE-2020-8195', 'CVE-2020-8196', 'CVE-2020-8218', 'CVE-2020-8243', 'CVE-2020-8255', 'CVE-2021-1577', 'CVE-2021-1580', 'CVE-2021-20260', 'CVE-2021-21587', 'CVE-2021-21703', 'CVE-2021-21705', 'CVE-2021-21964', 'CVE-2021-21965', 'CVE-2021-21968', 'CVE-2021-22044', 'CVE-2021-22047', 'CVE-2021-22053', 'CVE-2021-22117', 'CVE-2021-22132', 'CVE-2021-22134', 'CVE-2021-22284', 'CVE-2021-22288', 'CVE-2021-22289', 'CVE-2021-22817', 'CVE-2021-22876', 'CVE-2021-22893', 'CVE-2021-22894', 'CVE-2021-22900', 'CVE-2021-22924', 'CVE-2021-22925', 'CVE-2021-22941', 'CVE-2021-22946', 'CVE-2021-23136', 'CVE-2021-23140', 'CVE-2021-23177', 'CVE-2021-23204', 'CVE-2021-24122', 'CVE-2021-25003', 'CVE-2021-25122', 'CVE-2021-25369', 'CVE-2021-25444', 'CVE-2021-25489', 'CVE-2021-25490', 'CVE-2021-25745', 'CVE-2021-25746', 'CVE-2021-26117', 'CVE-2021-26606', 'CVE-2021-26726', 'CVE-2021-26727', 'CVE-2021-26728', 'CVE-2021-26729', 'CVE-2021-26731', 'CVE-2021-26732', 'CVE-2021-26733', 'CVE-2021-27853', 'CVE-2021-27854', 'CVE-2021-27861', 'CVE-2021-27862', 'CVE-2021-28169', 'CVE-2021-28170', 'CVE-2021-28799', 'CVE-2021-30299', 'CVE-2021-3031', 'CVE-2021-3038', 'CVE-2021-3044', 'CVE-2021-3046', 'CVE-2021-3048', 'CVE-2021-3049', 'CVE-2021-3062', 'CVE-2021-31403', 'CVE-2021-31404', 'CVE-2021-31406', 'CVE-2021-3155', 'CVE-2021-31566', 'CVE-2021-31917', 'CVE-2021-32029', 'CVE-2021-32039', 'CVE-2021-32076', 'CVE-2021-32648', 'CVE-2021-32706', 'CVE-2021-32819', 'CVE-2021-33592', 'CVE-2021-33593', 'CVE-2021-33723', 'CVE-2021-33727', 'CVE-2021-3426', 'CVE-2021-34401', 'CVE-2021-34429', 'CVE-2021-3452', 'CVE-2021-34571', 'CVE-2021-34577', 'CVE-2021-34588', 'CVE-2021-34589', 'CVE-2021-3490', 'CVE-2021-3493', 'CVE-2021-3503', 'CVE-2021-3513', 'CVE-2021-35221', 'CVE-2021-35223', 'CVE-2021-35232', 'CVE-2021-35245', 'CVE-2021-35247', 'CVE-2021-35248', 'CVE-2021-35249', 'CVE-2021-35254', 'CVE-2021-3572', 'CVE-2021-3580', 'CVE-2021-3585', 'CVE-2021-35937', 'CVE-2021-35938', 'CVE-2021-35939', 'CVE-2021-3602', 'CVE-2021-3612', 'CVE-2021-3631', 'CVE-2021-3675', 'CVE-2021-3677', 'CVE-2021-3688', 'CVE-2021-3701', 'CVE-2021-3714', 'CVE-2021-37147', 'CVE-2021-3722', 'CVE-2021-3732', 'CVE-2021-3736', 'CVE-2021-37533', 'CVE-2021-37580', 'CVE-2021-37701', 'CVE-2021-37712', 'CVE-2021-3773', 'CVE-2021-3784', 'CVE-2021-3798', 'CVE-2021-3800', 'CVE-2021-3827', 'CVE-2021-38454', 'CVE-2021-3849', 'CVE-2021-3856', 'CVE-2021-3864', 'CVE-2021-3897', 'CVE-2021-39008', 'CVE-2021-39144', 'CVE-2021-3917', 'CVE-2021-39226', 'CVE-2021-39327', 'CVE-2021-3970', 'CVE-2021-3981', 'CVE-2021-3995', 'CVE-2021-3996', 'CVE-2021-4037', 'CVE-2021-40390', 'CVE-2021-40404', 'CVE-2021-40405', 'CVE-2021-40413', 'CVE-2021-40416', 'CVE-2021-40423', 'CVE-2021-40690', 'CVE-2021-41079', 'CVE-2021-4112', 'CVE-2021-41124', 'CVE-2021-4120', 'CVE-2021-4125', 'CVE-2021-41277', 'CVE-2021-4135', 'CVE-2021-4142', 'CVE-2021-41769', 'CVE-2021-41807', 'CVE-2021-4197', 'CVE-2021-41992', 'CVE-2021-4210', 'CVE-2021-4211', 'CVE-2021-43588', 'CVE-2021-43762', 'CVE-2021-43767', 'CVE-2021-43937', 'CVE-2021-43938', 'CVE-2021-43939', 'CVE-2021-43999', 'CVE-2021-44040', 'CVE-2021-44228', 'CVE-2021-4428', 'CVE-2021-44354', 'CVE-2021-44419', 'CVE-2021-44467', 'CVE-2021-44521', 'CVE-2021-44529', 'CVE-2021-44739', 'CVE-2021-44759', 'CVE-2021-44776', 'CVE-2021-44832', 'CVE-2021-45029', 'CVE-2021-45074', 'CVE-2021-45105', 'CVE-2021-46270', 'CVE-2021-46279', 'CVE-2022-0019', 'CVE-2022-0026', 'CVE-2022-0027', 'CVE-2022-0029', 'CVE-2022-0030', 'CVE-2022-0073', 'CVE-2022-0155', 'CVE-2022-0217', 'CVE-2022-0336', 'CVE-2022-0492', 'CVE-2022-0494', 'CVE-2022-0564', 'CVE-2022-0567', 'CVE-2022-0656', 'CVE-2022-0708', 'CVE-2022-0715', 'CVE-2022-0718', 'CVE-2022-0811', 'CVE-2022-0813', 'CVE-2022-0850', 'CVE-2022-0851', 'CVE-2022-0852', 'CVE-2022-0862', 'CVE-2022-0910', 'CVE-2022-0987', 'CVE-2022-1117', 'CVE-2022-1159', 'CVE-2022-1353', 'CVE-2022-1471', 'CVE-2022-1596', 'CVE-2022-1655', 'CVE-2022-1833', 'CVE-2022-2031', 'CVE-2022-2048', 'CVE-2022-20630', 'CVE-2022-20658', 'CVE-2022-20660', 'CVE-2022-20680', 'CVE-2022-20696', 'CVE-2022-20698', 'CVE-2022-20733', 'CVE-2022-20742', 'CVE-2022-20745', 'CVE-2022-20748', 'CVE-2022-20762', 'CVE-2022-20777', 'CVE-2022-20779', 'CVE-2022-20780', 'CVE-2022-20797', 'CVE-2022-20798', 'CVE-2022-20805', 'CVE-2022-21194', 'CVE-2022-21196', 'CVE-2022-21686', 'CVE-2022-21712', 'CVE-2022-21800', 'CVE-2022-21820', 'CVE-2022-21831', 'CVE-2022-2191', 'CVE-2022-22110', 'CVE-2022-22141', 'CVE-2022-22148', 'CVE-2022-22151', 'CVE-2022-22241', 'CVE-2022-22243', 'CVE-2022-22282', 'CVE-2022-22521', 'CVE-2022-22537', 'CVE-2022-22538', 'CVE-2022-22539', 'CVE-2022-22545', 'CVE-2022-22553', 'CVE-2022-22576', 'CVE-2022-22722', 'CVE-2022-22806', 'CVE-2022-22807', 'CVE-2022-22813', 'CVE-2022-22947', 'CVE-2022-22963', 'CVE-2022-22965', 'CVE-2022-22990', 'CVE-2022-22998', 'CVE-2022-23008', 'CVE-2022-23014', 'CVE-2022-23019', 'CVE-2022-23067', 'CVE-2022-23131', 'CVE-2022-23134', 'CVE-2022-23402', 'CVE-2022-23469', 'CVE-2022-23634', 'CVE-2022-23635', 'CVE-2022-23648', 'CVE-2022-23711', 'CVE-2022-2394', 'CVE-2022-24044', 'CVE-2022-24075', 'CVE-2022-24086', 'CVE-2022-24112', 'CVE-2022-24280', 'CVE-2022-24385', 'CVE-2022-24398', 'CVE-2022-24663', 'CVE-2022-24664', 'CVE-2022-24665', 'CVE-2022-24720', 'CVE-2022-24731', 'CVE-2022-24734', 'CVE-2022-24735', 'CVE-2022-24768', 'CVE-2022-24769', 'CVE-2022-24853', 'CVE-2022-24866', 'CVE-2022-24867', 'CVE-2022-24881', 'CVE-2022-24882', 'CVE-2022-24885', 'CVE-2022-24886', 'CVE-2022-24972', 'CVE-2022-2503', 'CVE-2022-25167', 'CVE-2022-25246', 'CVE-2022-25248', 'CVE-2022-2551', 'CVE-2022-25823', 'CVE-2022-25824', 'CVE-2022-25825', 'CVE-2022-25943', 'CVE-2022-25989', 'CVE-2022-26071', 'CVE-2022-26106', 'CVE-2022-26107', 'CVE-2022-26109', 'CVE-2022-26138', 'CVE-2022-26336', 'CVE-2022-26340', 'CVE-2022-26531', 'CVE-2022-2663', 'CVE-2022-26672', 'CVE-2022-26839', 'CVE-2022-26851', 'CVE-2022-26854', 'CVE-2022-26855', 'CVE-2022-26865', 'CVE-2022-2735', 'CVE-2022-27506', 'CVE-2022-27511', 'CVE-2022-27512', 'CVE-2022-27518', 'CVE-2022-27634', 'CVE-2022-27649', 'CVE-2022-27655', 'CVE-2022-27667', 'CVE-2022-27774', 'CVE-2022-27775', 'CVE-2022-27875', 'CVE-2022-28127', 'CVE-2022-28186', 'CVE-2022-28188', 'CVE-2022-28193', 'CVE-2022-28195', 'CVE-2022-28196', 'CVE-2022-28214', 'CVE-2022-2827', 'CVE-2022-28614', 'CVE-2022-28695', 'CVE-2022-28708', 'CVE-2022-28774', 'CVE-2022-29160', 'CVE-2022-29217', 'CVE-2022-29248', 'CVE-2022-29263', 'CVE-2022-29417', 'CVE-2022-29479', 'CVE-2022-29613', 'CVE-2022-29813', 'CVE-2022-29814', 'CVE-2022-29815', 'CVE-2022-29818', 'CVE-2022-29819', 'CVE-2022-29821', 'CVE-2022-29901', 'CVE-2022-30115', 'CVE-2022-30556', 'CVE-2022-30670', 'CVE-2022-30730', 'CVE-2022-31024', 'CVE-2022-31036', 'CVE-2022-31042', 'CVE-2022-31043', 'CVE-2022-31055', 'CVE-2022-31083', 'CVE-2022-31162', 'CVE-2022-31248', 'CVE-2022-3140', 'CVE-2022-31465', 'CVE-2022-31479', 'CVE-2022-31480', 'CVE-2022-31485', 'CVE-2022-31619', 'CVE-2022-31629', 'CVE-2022-3171', 'CVE-2022-3180', 'CVE-2022-32143', 'CVE-2022-32144', 'CVE-2022-32154', 'CVE-2022-32158', 'CVE-2022-32169', 'CVE-2022-32170', 'CVE-2022-32212', 'CVE-2022-32255', 'CVE-2022-32513', 'CVE-2022-32514', 'CVE-2022-32515', 'CVE-2022-32517', 'CVE-2022-32518', 'CVE-2022-32520', 'CVE-2022-32534', 'CVE-2022-32742', 'CVE-2022-32744', 'CVE-2022-32747', 'CVE-2022-3284', 'CVE-2022-33714', 'CVE-2022-33715', 'CVE-2022-33719', 'CVE-2022-33720', 'CVE-2022-33721', 'CVE-2022-33723', 'CVE-2022-33724', 'CVE-2022-33725', 'CVE-2022-33727', 'CVE-2022-33729', 'CVE-2022-33731', 'CVE-2022-33732', 'CVE-2022-34255', 'CVE-2022-34256', 'CVE-2022-34259', 'CVE-2022-35290', 'CVE-2022-35296', 'CVE-2022-35666', 'CVE-2022-35668', 'CVE-2022-3676', 'CVE-2022-36960', 'CVE-2022-37393', 'CVE-2022-37400', 'CVE-2022-37436', 'CVE-2022-3787', 'CVE-2022-38112', 'CVE-2022-38113', 'CVE-2022-39013', 'CVE-2022-39016', 'CVE-2022-39017', 'CVE-2022-39018', 'CVE-2022-39019', 'CVE-2022-39042', 'CVE-2022-39236', 'CVE-2022-39249', 'CVE-2022-39250', 'CVE-2022-39251', 'CVE-2022-39253', 'CVE-2022-39327', 'CVE-2022-40259', 'CVE-2022-40293', 'CVE-2022-4057', 'CVE-2022-4060', 'CVE-2022-40751', 'CVE-2022-41205', 'CVE-2022-41214', 'CVE-2022-41264', 'CVE-2022-4140', 'CVE-2022-41694', 'CVE-2022-41813', 'CVE-2022-41836', 'CVE-2022-41915', 'CVE-2022-42340', 'CVE-2022-4236', 'CVE-2022-42972', 'CVE-2022-45052', 'CVE-2022-45787', 'CVE-2022-46363', 'CVE-2022-47391', 'CVE-2022-47508', 'CVE-2022-47618', 'CVE-2022-47894', 'CVE-2022-4862', 'CVE-2022-4904', 'CVE-2023-0002', 'CVE-2023-0012', 'CVE-2023-0014', 'CVE-2023-0020', 'CVE-2023-0022', 'CVE-2023-0023', 'CVE-2023-0297', 'CVE-2023-0386', 'CVE-2023-1007', 'CVE-2023-1453', 'CVE-2023-1486', 'CVE-2023-20867', 'CVE-2023-21434', 'CVE-2023-21524', 'CVE-2023-21531', 'CVE-2023-21540', 'CVE-2023-21542', 'CVE-2023-21549', 'CVE-2023-21550', 'CVE-2023-21553', 'CVE-2023-21554', 'CVE-2023-21558', 'CVE-2023-21559', 'CVE-2023-21574', 'CVE-2023-21596', 'CVE-2023-21607', 'CVE-2023-21678', 'CVE-2023-21685', 'CVE-2023-21717', 'CVE-2023-21721', 'CVE-2023-21722', 'CVE-2023-21726', 'CVE-2023-21742', 'CVE-2023-21749', 'CVE-2023-21751', 'CVE-2023-21760', 'CVE-2023-21767', 'CVE-2023-21777', 'CVE-2023-21816', 'CVE-2023-21818', 'CVE-2023-22232', 'CVE-2023-2283', 'CVE-2023-22858', 'CVE-2023-23382', 'CVE-2023-23397', 'CVE-2023-2359', 'CVE-2023-23857', 'CVE-2023-23934', 'CVE-2023-24484', 'CVE-2023-24485', 'CVE-2023-24486', 'CVE-2023-24489', 'CVE-2023-2454', 'CVE-2023-2455', 'CVE-2023-24807', 'CVE-2023-24881', 'CVE-2023-24893', 'CVE-2023-24904', 'CVE-2023-24905', 'CVE-2023-24950', 'CVE-2023-25661', 'CVE-2023-26049', 'CVE-2023-26360', 'CVE-2023-26367', 'CVE-2023-26406', 'CVE-2023-26407', 'CVE-2023-26408', 'CVE-2023-26460', 'CVE-2023-27268', 'CVE-2023-27350', 'CVE-2023-27351', 'CVE-2023-27893', 'CVE-2023-27894', 'CVE-2023-27897', 'CVE-2023-28121', 'CVE-2023-28221', 'CVE-2023-28222', 'CVE-2023-28246', 'CVE-2023-28271', 'CVE-2023-28274', 'CVE-2023-28291', 'CVE-2023-28303', 'CVE-2023-28304', 'CVE-2023-28312', 'CVE-2023-28322', 'CVE-2023-28432', 'CVE-2023-2868', 'CVE-2023-28762', 'CVE-2023-28764', 'CVE-2023-28765', 'CVE-2023-28856', 'CVE-2023-28900', 'CVE-2023-29108', 'CVE-2023-29111', 'CVE-2023-29298', 'CVE-2023-29335', 'CVE-2023-29338', 'CVE-2023-29343', 'CVE-2023-3006', 'CVE-2023-30440', 'CVE-2023-30445', 'CVE-2023-30446', 'CVE-2023-30447', 'CVE-2023-30448', 'CVE-2023-30449', 'CVE-2023-30535', 'CVE-2023-30740', 'CVE-2023-30861', 'CVE-2023-30991', 'CVE-2023-30996', 'CVE-2023-31064', 'CVE-2023-31066', 'CVE-2023-31098', 'CVE-2023-31404', 'CVE-2023-31453', 'CVE-2023-32037', 'CVE-2023-32043', 'CVE-2023-32053', 'CVE-2023-32056', 'CVE-2023-32057', 'CVE-2023-32113', 'CVE-2023-32243', 'CVE-2023-3247', 'CVE-2023-3262', 'CVE-2023-32697', 'CVE-2023-33148', 'CVE-2023-33155', 'CVE-2023-33165', 'CVE-2023-33174', 'CVE-2023-33246', 'CVE-2023-33990', 'CVE-2023-3466', 'CVE-2023-3519', 'CVE-2023-35303', 'CVE-2023-35320', 'CVE-2023-35333', 'CVE-2023-35336', 'CVE-2023-35347', 'CVE-2023-35348', 'CVE-2023-35353', 'CVE-2023-35365', 'CVE-2023-35367', 'CVE-2023-35619', 'CVE-2023-35624', 'CVE-2023-35625', 'CVE-2023-35633', 'CVE-2023-35841', 'CVE-2023-35870', 'CVE-2023-36018', 'CVE-2023-36021', 'CVE-2023-36046', 'CVE-2023-36047', 'CVE-2023-36049', 'CVE-2023-36052', 'CVE-2023-36391', 'CVE-2023-36394', 'CVE-2023-36399', 'CVE-2023-36404', 'CVE-2023-36407', 'CVE-2023-36437', 'CVE-2023-36563', 'CVE-2023-36705', 'CVE-2023-36719', 'CVE-2023-36761', 'CVE-2023-36872', 'CVE-2023-36874', 'CVE-2023-36917', 'CVE-2023-36920', 'CVE-2023-36923', 'CVE-2023-36924', 'CVE-2023-37464', 'CVE-2023-37486', 'CVE-2023-37582', 'CVE-2023-38205', 'CVE-2023-38206', 'CVE-2023-38220', 'CVE-2023-38245', 'CVE-2023-38720', 'CVE-2023-38728', 'CVE-2023-38740', 'CVE-2023-39422', 'CVE-2023-39440', 'CVE-2023-40029', 'CVE-2023-40049', 'CVE-2023-40273', 'CVE-2023-4039', 'CVE-2023-4043', 'CVE-2023-40621', 'CVE-2023-40622', 'CVE-2023-40743', 'CVE-2023-41313', 'CVE-2023-4164', 'CVE-2023-42480', 'CVE-2023-42503', 'CVE-2023-43123', 'CVE-2023-43804', 'CVE-2023-4481', 'CVE-2023-45143', 'CVE-2023-45236', 'CVE-2023-45648', 'CVE-2023-46747', 'CVE-2023-4759', 'CVE-2023-49081', 'CVE-2023-49082', 'CVE-2023-49093', 'CVE-2023-49198', 'CVE-2023-49578', 'CVE-2023-49580', 'CVE-2023-49582', 'CVE-2023-50164', 'CVE-2023-5043', 'CVE-2023-5044', 'CVE-2023-6548', 'CVE-2024-0057', 'CVE-2024-0204', 'CVE-2024-0411', 'CVE-2024-10327', 'CVE-2024-10924', 'CVE-2024-11234', 'CVE-2024-11680', 'CVE-2024-1709', 'CVE-2024-20656', 'CVE-2024-20657', 'CVE-2024-20665', 'CVE-2024-20666', 'CVE-2024-20669', 'CVE-2024-20670', 'CVE-2024-20673', 'CVE-2024-20675', 'CVE-2024-20684', 'CVE-2024-20695', 'CVE-2024-20709', 'CVE-2024-20721', 'CVE-2024-20733', 'CVE-2024-20767', 'CVE-2024-21302', 'CVE-2024-21304', 'CVE-2024-21305', 'CVE-2024-21312', 'CVE-2024-21315', 'CVE-2024-21316', 'CVE-2024-21319', 'CVE-2024-21320', 'CVE-2024-21329', 'CVE-2024-21351', 'CVE-2024-21364', 'CVE-2024-21374', 'CVE-2024-21376', 'CVE-2024-21378', 'CVE-2024-21380', 'CVE-2024-21390', 'CVE-2024-21397', 'CVE-2024-21401', 'CVE-2024-21402', 'CVE-2024-21403', 'CVE-2024-21410', 'CVE-2024-21412', 'CVE-2024-21424', 'CVE-2024-21427', 'CVE-2024-21447', 'CVE-2024-21448', 'CVE-2024-21737', 'CVE-2024-21742', 'CVE-2024-21793', 'CVE-2024-22029', 'CVE-2024-22131', 'CVE-2024-22245', 'CVE-2024-23944', 'CVE-2024-24549', 'CVE-2024-24683', 'CVE-2024-24795', 'CVE-2024-24919', 'CVE-2024-25153', 'CVE-2024-25641', 'CVE-2024-25645', 'CVE-2024-25646', 'CVE-2024-26026', 'CVE-2024-26063', 'CVE-2024-26119', 'CVE-2024-26126', 'CVE-2024-26127', 'CVE-2024-26158', 'CVE-2024-26164', 'CVE-2024-26167', 'CVE-2024-26177', 'CVE-2024-26189', 'CVE-2024-26193', 'CVE-2024-26197', 'CVE-2024-26199', 'CVE-2024-26201', 'CVE-2024-26203', 'CVE-2024-26216', 'CVE-2024-26234', 'CVE-2024-26238', 'CVE-2024-26240', 'CVE-2024-26250', 'CVE-2024-27135', 'CVE-2024-27438', 'CVE-2024-27458', 'CVE-2024-27894', 'CVE-2024-28163', 'CVE-2024-28164', 'CVE-2024-28185', 'CVE-2024-28189', 'CVE-2024-28253', 'CVE-2024-28255', 'CVE-2024-28847', 'CVE-2024-28848', 'CVE-2024-28849', 'CVE-2024-28897', 'CVE-2024-28907', 'CVE-2024-28916', 'CVE-2024-28917', 'CVE-2024-28919', 'CVE-2024-28921', 'CVE-2024-29006', 'CVE-2024-29014', 'CVE-2024-29054', 'CVE-2024-29055', 'CVE-2024-29056', 'CVE-2024-29060', 'CVE-2024-29187', 'CVE-2024-29988', 'CVE-2024-29992', 'CVE-2024-29993', 'CVE-2024-29995', 'CVE-2024-29998', 'CVE-2024-30002', 'CVE-2024-30018', 'CVE-2024-30033', 'CVE-2024-30040', 'CVE-2024-30041', 'CVE-2024-30050', 'CVE-2024-30052', 'CVE-2024-30054', 'CVE-2024-30059', 'CVE-2024-30065', 'CVE-2024-30076', 'CVE-2024-30078', 'CVE-2024-30087', 'CVE-2024-30093', 'CVE-2024-30096', 'CVE-2024-30104', 'CVE-2024-31141', 'CVE-2024-31227', 'CVE-2024-31860', 'CVE-2024-31862', 'CVE-2024-31863', 'CVE-2024-31864', 'CVE-2024-31867', 'CVE-2024-32007', 'CVE-2024-33003', 'CVE-2024-33065', 'CVE-2024-33066', 'CVE-2024-3383', 'CVE-2024-3385', 'CVE-2024-34340', 'CVE-2024-34684', 'CVE-2024-35253', 'CVE-2024-35254', 'CVE-2024-35263', 'CVE-2024-37180', 'CVE-2024-37325', 'CVE-2024-37341', 'CVE-2024-3746', 'CVE-2024-37965', 'CVE-2024-38046', 'CVE-2024-38084', 'CVE-2024-38098', 'CVE-2024-38112', 'CVE-2024-38163', 'CVE-2024-38180', 'CVE-2024-38188', 'CVE-2024-38189', 'CVE-2024-38194', 'CVE-2024-38195', 'CVE-2024-38196', 'CVE-2024-38197', 'CVE-2024-38200', 'CVE-2024-38201', 'CVE-2024-38203', 'CVE-2024-38213', 'CVE-2024-38216', 'CVE-2024-38220', 'CVE-2024-38223', 'CVE-2024-38225', 'CVE-2024-38226', 'CVE-2024-38230', 'CVE-2024-38231', 'CVE-2024-38234', 'CVE-2024-38241', 'CVE-2024-38243', 'CVE-2024-38244', 'CVE-2024-38245', 'CVE-2024-38367', 'CVE-2024-38425', 'CVE-2024-38646', 'CVE-2024-38647', 'CVE-2024-38810', 'CVE-2024-39398', 'CVE-2024-39404', 'CVE-2024-39405', 'CVE-2024-39407', 'CVE-2024-39412', 'CVE-2024-39413', 'CVE-2024-39414', 'CVE-2024-39415', 'CVE-2024-39416', 'CVE-2024-39417', 'CVE-2024-39418', 'CVE-2024-39573', 'CVE-2024-39593', 'CVE-2024-39597', 'CVE-2024-39599', 'CVE-2024-39600', 'CVE-2024-40766', 'CVE-2024-41729', 'CVE-2024-41732', 'CVE-2024-41733', 'CVE-2024-41736', 'CVE-2024-41806', 'CVE-2024-41856', 'CVE-2024-42494', 'CVE-2024-43461', 'CVE-2024-43469', 'CVE-2024-43470', 'CVE-2024-43479', 'CVE-2024-43482', 'CVE-2024-43487', 'CVE-2024-43492', 'CVE-2024-4358', 'CVE-2024-43602', 'CVE-2024-43645', 'CVE-2024-44113', 'CVE-2024-45195', 'CVE-2024-45216', 'CVE-2024-45219', 'CVE-2024-46901', 'CVE-2024-47076', 'CVE-2024-47175', 'CVE-2024-47592', 'CVE-2024-48862', 'CVE-2024-49039', 'CVE-2024-49040', 'CVE-2024-49044', 'CVE-2024-49048', 'CVE-2024-49049', 'CVE-2024-49051', 'CVE-2024-49056', 'CVE-2024-49328', 'CVE-2024-49604', 'CVE-2024-5082', 'CVE-2024-5535', 'CVE-2024-56326', 'CVE-2024-5806', 'CVE-2024-5913', 'CVE-2024-6385', 'CVE-2024-6915', 'CVE-2024-7014', 'CVE-2024-7254', 'CVE-2024-7593', 'CVE-2024-8260', 'CVE-2024-9264', 'CVE-2024-9486', 'CVE-2025-0064', 'CVE-2025-0411', 'CVE-2025-0683', 'CVE-2025-1097', 'CVE-2025-1098', 'CVE-2025-21293', 'CVE-2025-23209', 'CVE-2025-24061', 'CVE-2025-24071', 'CVE-2025-24408', 'CVE-2025-24411', 'CVE-2025-24418', 'CVE-2025-24422', 'CVE-2025-24424', 'CVE-2025-24426', 'CVE-2025-24427', 'CVE-2025-24429', 'CVE-2025-24435', 'CVE-2025-24437', 'CVE-2025-24513', 'CVE-2025-24514', 'CVE-2025-24876', 'CVE-2025-24970', 'CVE-2025-24989', 'CVE-2025-26658', 'CVE-2025-2746', 'CVE-2025-2747', 'CVE-2025-29927'],
    #     'pulse': ['5c88e255d5fbbe1cb7c54534', '5cb741fb29d3ac01c1ac01ea', '5cfa66ced7bc379b04017f36', '5e7b4a11d552fbcfce6c314d', '5f62318a419e133716eca72d', '5f8f08e03e906183f28915dc', '607f4734482486b74db7a2a5', '608725a144ffb0ec56af74be', '608c75593de1fe5ca6a8ca26', '60aff44e91865ceaac51e473', '6113690279e9eb9f64fac829', '61695dd13611f59694746cda', '61c00f661617d4a5cfef88d9', '61d4703b9c64b7783fc7cf75', '61d488268466c45883b5b38d', '61d9cb8ac71f29ee19314ece', '61d9cb8ea12bdeddcbcdb00a', '61dd26a576d13eb49a135cd5', '61def4fcbbf4bd2b73890632', '61def579cfa208c04c94ceb5', '61deff0101737ee62817cbd7', '61df005d9f0ff216afd15bdc', '61dfec9f252191bd5d552665', '61e0222c305103ca70237bab', '61e08c4bdb19d679585909f5', '61e08c6c3bc170cf83060753', '61e18c8c830dde350ac934f8', '61e19e733ec4215697d77f1a', '61e1b70ab9a4692ae469a090', '61e1d2a375e82957fdda03a8', '61e46aa082235e3bcc6ad202', '61e47d110b37788dcbe2f35f', '61e4fdc113fc0f2353a2d43b', '61e5048f1bf0dd2d148afa02', '61e50a56b5f3b90b1442ea87', '61e64b9d2f9e981dbc50de5b', '61e64c654f3052ea5488d22a', '61e7911210659b875d9c0d9f', '61e793dbb4223ec56dfa93ff', '61ea34e3de858b9e610392bb', '61ea3ebe8ac6c6f3529de234', '61ea41afa123e1c8a04444f0', '61ea60ef8dce498857059215', '61ee3acb8b2802a34a0274bb', '61ee3e2558564077bc3e5551', '61ee3fac42420886673fb260', '61ee9bf84cb4a10066f023ef', '61f21e5d66aafdcd9f8776ea', '61f235f56624436b3d73fc2b', '61f377bacff16dc1ab9d5d42', '61f4237effc10e171d4b9583', '61f4453f33e86346a02f961c', '61f552d47a02c57e5bad2afc', '61fa0c674cb4b79495d760b5', '61fa0e35d2248aeddcd4bf8f', '61fa196f26ed3f73cb7bab90', '61fa1ce236918d966a8ddcbf', '61fb65b36b041172547306bc', '61fb674df497a4a2069ba713', '61fcba241a6e01c39ac5fa9d', '61fcbb0f13c0a592646f942e', '61fcbbdc228d97b1dcccbd8a', '61fd164e9cffde58cab8f488', '61fedb5c8e35f997f199d652', '6200b1c61fb94f55bc8a328b', '62032c163f50d127ae2ae7f4', '62035bd4416992510a785958', '62035c322ce60d6af887a4ed', '62035cb62c4236e4db543b38', '6205691491c237cf45f56462', '6205f2a0cbb8d712f05e780b', '6209459378beb5e65e74c5dd', '6209551a609f62eb72ec9d86', '6209644c25c5c3dd3257e87b', '62096ce27d5d12f517aa7f22', '6209e70b2dbb80bd25b465a4', '6209e78a655a10bbb912d3bd', '620a7c61440979287380fca4', '620a84a754efb383e154fd59', '620aba8c6fc62088817bb924', '620b462ae88fcd19b5b785e2', '620ccc1182612e3c23c02e02', '620ccebbec1b8678ac232ff9', '620cd2d0fd8257b6f1f3c693', '620dd8b421678d92233a64ba', '620dd978bc1bb788531890af', '620ddbb05c95f995d24f2eaa', '620f44669638ae9dc791bf4f', '620f4468859dd4c838554a4c', '6214ddb6667a4f68da319336', '6214ddb8410d1a052959e11d', '6214fd63e85542d706b66fa8', '62161999458df32ab2104efe', '6217822a96829b893f0a40d9', '6217a0a2a1265ba0a16888bf', '6217c9963ee88f435dbff552', '62189de26a1fc8fefb03c4f7', '6218cc36c0cf3c8fbed6a14b', '6218e7a4f6061eea3f0bbd57', '621cfc11806aadc36056f312', '621f37be91db3cd3cbaabac8', '621f3a45711ff618fe803ae7', '621f40ad31a26cb530b7da08', '621f4747a5944ee1a9735a2d', '621f47bf73d522fa4cd60c35', '621f4bf86b1a97ea84d560b6', '6220f960cd2b607e485ce025', '6225fa23ea7c77caa06b6af8', '622603098ef6175f143528e7', '622603d018bb937d93f18b90', '6226054c02d688b554b9317a', '6226072d216f4527213a6925', '62260a5e980e25836cabb9c9', '62287e7abc5a7a2e46e78dbf', '62287f6d6adf4dac253fdd92', '62288a3e275f7fdd7d0f1a6b', '62288b7893bcc8ff41f4be6b', '6228a952b8527c6374dfec52', '6228b4f0b2d5ade72cc7a0ea', '6228c44bc3398c90a2da9e52', '6229b9238830549c0472862d', '622a49747442f9ac0473a26b', '622b4524de418d6d25e8c5dc', '622b45713e4bb01fd4ada846', '622c4e3b5aa331a587eca96d', '622f08d792011bf0d58d8b2f', '622f34890b534565792b6be6', '622fd6e4d2592fc24c87e51d', '62302c1b4725c9b58789e0ae', '62307f58a5146eb7ed2cd262', '6231b72b7dfa2b437c530577', '62348eb5a0db758c735a8c23', '62349926d63031bbfe012621', '6237acc3b44530d49d96b3b6', '623824e4bb9affd1ba2d9df1', '623827d256465459f941b04d', '623829a96a125e0a8820d340', '62382ae27ba6accc6cdaa37b', '62396cb86058a14deee24cf9', '623dc346d0ad386b5eaec184', '6240756ea8ce62d319a4ca85', '624298dada94224d29d4f8ee', '624aef7893eec93783a63647', '624af172f9234f0952066eba', '624af55e665dd4f6604f6072', '624afb43a77dd860b5ef3cd9', '624afbe8c4ccec2d11b4061b', '624b02c4f17c2c35747bc298', '624b078044fc61d3b93007ba', '624b1623e8a20d14f9371590', '624b65f3ea2b736604a874c6', '624b6621e5de8b4c5ec3c80f', '624b6754db2e3d7978efe124', '624b6782bd41664295824441', '624bc9eb7237c87b9c1ae4e9', '624d19fc5f4f437c253ff63e', '624d236b17ee234b636c6ba3', '624d5375489ddf48d249cf04', '624ed70df705969144ee4035', '624ede54e2e09887574c0233', '624ee6af51c7707137215465', '62518c06a25b32eddeb4175f', '6251a4eeb235f2fe5fd6564c', '625426e5135c942ef6211b89', '62545114a2bfb60d7097c646', '62567dcc06f78289866b7db1', '625bbe88cbafd6ac31442278', '625d4ab4eb07779bf6221298', '625d504e6f9c25e78f4ef442', '625d58dbc24dc678720b223e', '625e4f2a9c0b44120763d375', '625e50745ad599bc9c4a4f1e', '625e6af73b7b0982c6920d17', '625ed296796e491224686ab2', '625ed897984eab234c6ec944', '625efc8658db58d3dc0f9572', '625f30881bd6644fc33db9d6', '625f382c638b44b686f58520', '626040216e00cd388b1b9180', '6260f27ac5c698041930217d', '6261009aa8dc9b97ef200ece', '62616780d27cf09ab2efc522', '62616a9d2694eb915e176d13', '6261881c028c00eb681606ec', '626223e499a92549749bf634', '6267fbe9b137f786faa9e825', '6268e096d42c2393b2bddfb1', '6269318f62bf4e09c946bf7c', '626d5d5d0bcdf733d0dc8713', '626f0c649658fe779ac37eed', '626f8201336bba0c759bffba', '626f97ebc840c2313b628cd2', '626fa3a4b095edd2ef093d03', '626fa408c3386e70ea1248bf', '626fa939fa82f8a29fe000bf', '626fab9d1aa3320d9fc45af9', '626fb07d350d86386ed947c3', '626fb3b0b3db36e80f5c68d3', '626fb4d719669351ffce150f', '626fb62dcfbf03791a39137d', '626fb6a1a9e9bf1956a2193d', '626fda978a4b7b89ae4b5899', '626fdc127b6c30ef1d590dfe', '626fde7fd196c298d0950ce3', '626fdf0b2e864d3cc2dc3de6', '6271815d5b43a4e4c1dbdd72', '6272d37b82fd89ba51b6a4fa', '6272d5299bfc6bd578bc99d8', '62740fbffdf7dc216fa473d7', '6279148a03d4caaf2755f7ab', '62791b82e7f0509d49ee1e7c', '6279252f4be88550c90caa5a', '627928ce0fb65efed0571448', '62793fa7333ad7f37f39bdc5', '627ad362d31cfb1b8990ec09', '627ad41b9cba3f0eb4e4a12b', '627bb0d5947a6d828600f696', '627bb3b912e1e0413ade1cae', '627bb55d238ef2e36af1ee47', '62826e6e99e1c6ccf06b100e', '628270d8268710bc5b03df8b', '6282727ce39861b3d0555171', '6282753a341365fb4cb031b4', '6284e6a3ac03d713427764ce', '6285ac654db92004b788b4ae', '628c7cfb6b0413ece7218de4', '628c9749160e4f4fc41d24a0', '628c9cc7ce1561535fee2565', '628c9d73ad1c576d586a25d0', '62905a9b74a16aee543df2b8', '6290612818f1091768851934', '62914d2c3694e482fa57f83e', '6294b4c14eef09d36a0e0445', '6294baaa70a4d1293968b434', '6294c11f8328a3727356bc79', '6294c120b83fd0366a88c247', '6294c12de1db3082ba2f2182', '6294c130e1db3082ba2f2183', '6295c89ad9162c33ab43f4c1', '6295cdc589590f0ab7f34423', '6295f2c308cccc2f34088b6b', '6299fc621cec139fbcdf9429', '629bc1d7f254af2a03488207', '629d9efbf9ecafd1d11b5169', '62a203fb56848981bc483c00', '62a2099362e720e668ab2b75', '62a20cc53443f576b771d261', '62a21574df118d0db8450abd', '62a73dd11791cf25d96d3f43', '62ab365e312017ee94dae0dd', '62ab39aba4338f14de2f652d', '62ab7196f070398b299a2e71', '62ab76a4036fe53751e635a7', '62b1cc898c1bedfb0433d3ad', '62b290cadc0ccfc853312064', '62b2e664239bab694df94cfd', '62b46f6d15146077ffa78c29', '62b476a22f170838c498950d', '62b476a38f9ce6cc5e5b1328', '62b5dbc4edaa45d31f2a70df', '62b5e44021454de9a8327b21', '62b95560d7256a83c889a925', '62b95fd25f4f624dc5a5309c', '62b967df23d3968f62e3a1d3', '62b972d532b6ab4bcde78435', '62b977b2fc6b5ae4b01bd688', '62b97cef3753ae9e3e264358', '62b97d95c41324bdec5cc467', '62b982c42091a2a6c74ef0ad', '62b99d4d454e9f56606deaa8', '62ba0b4d192cdc7b9cd6cd5d', '62ba35736c97e9f5a1a14978', '62ba46bc291caee46d63336a', '62ba96d316f8d360a727f00b', '62ba972eb403bd2062972eaa', '62ba982d628d8282d78900d1', '62ba99a1b403bd2062972eab', '62ba9a315a22bce21dbe2f3e', '62ba9b3c246a2bef94d7108c', '62baa09ff007f0f1a3dba1a0', '62baa0dfffd5a0f9e2fc37b0', '62baa2d474d816c20c55a97b', '62baa8c470a5a96a2cd8ade1', '62baa9dab0a3d25c6da62330', '62baaaab1354776b08898b89', '62bab06be7c99b18481fd88e', '62badad4050ec43174fbe015', '62badfc332b3bf7b31a5e884', '62bae2fb9cb0441ece550940', '62bae3b8b780914023ad6b27', '62bae3f790d36a31b490d08c', '62bae7636beb3a9dac8037ef', '62bae97c0218f5a726c27c67', '62baebda7faf2a630fa21f1f', '62baec5262e01b84ab429ebe', '62bb3cd8e18ca5252dcdd059', '62bb9691168d324fb3511255', '62bc5b629a8c8dbea7b46ecd', '62bed8104578110a6837efba', '62cbba438824f6c661c84995', '62d1706692053cb2d6a5f1d8', '62d17953eef8d3fefbcc47af', '62d17f78496f956b9814a7d1', '62d596031528cabc959023a9', '62d74be1123f330b40d3de44', '62d74be21a58e328ee29b9d3', '62d74be339add8157494b214', '62d74bf1bcbb5dfa193005b7', '62dc43cfbbad6b6a25f66306', '62dd872ccdc8ffcb3daf74e1', '62e2d33a9a2821bc35ea3b74', '62e3846ac813896a49812b5d', '62e93f84f3c3ade7272d85ef', '62e9940930cfdfe2ffb490a6', '62ea78045e85e06b632b48ee', '62ebc01347d56a11788ee04e', '62ec0fd8b24fcb3bd7a4a81d', '62ed006b73d1a20d4c0b056c', '62f28e895658ab6b557f73c9', '62f29bc29532eee97146dce3', '62f6644b18fa9d4cf89bc34e', '62faa7ded1a93dfbe41cbca5', '62fbe809bae78cb304ba5a9b', '62fc921f3a9eb35e1d78d28a', '62fd01c900afccb5f81e8cce', '62fded0f1f5547c189eae6b9', '62ffc421748a66f8336acd30', '62ffc44e1d9cb0ffdd9d6640', '6301500e8bccbbd8bf1eb883', '63037a764f7474d9194f3365', '6304478b94b92e2ee69b1cc4', '630b731ed3d54d7dfc5ea8c9', '630b8ac028507ead20b9d68d', '630e439ea98624c5bb1d4545', '631562848a15051a53e5dd23', '6320bd73317c5c7d3116d4f6', '6320cbb3de76790891b0471a', '63218708c90625db457b79a9', '6321d318f7065a6be14383ee', '63222d1f5ef6a49c5d8baaed', '6322d260fce700b422bc8ded', '632323f7b974ea595174c847', '6324720fb38e5d3d757ca2f4', '63261cd6843d867201a8ba0c', '632793c5a997b7a080450d73', '6328a57ec461f46ac6a1b985', '632a7067fd5fee23983daa0d', '632a70740d70e21206a4df20', '632b23446eca84a6a0125395', '632c9caf3a5c7dcb35fc7bf1', '6331b8457dbc0865595f6b72', '6335a8ec3ea7eca7bcf26b8b', '6335bc540c4ca457ed9ad4d9', '633adacc74c9aca5bb26c159', '633b40b7ba504b51f2688631', '633b4e3178ebc39cc81850eb', '633bc352b7e7eb15b49bdb42', '633bc4ffe50b434caf1d7419', '633c0f200aa7a48f62fb6c4b', '633c6894b934a17dba8aec26', '633c689a6e806dad4ca1a418', '633ded26ee3223e405f58347', '633f43f576d385122d400bc4', '633fe22a6ca33b8627222c81', '633fe22ab4dc0246d354afbd', '633fe2c9fa8609c5fe071324', '633fe35929c8c9e67575a22f', '633fe3d636a91775abe43565', '633fe53261d78317ba30d0ae', '633fe57015ce47aef9ed1a6d', '633fe5a42d78d2b50ba7e22f', '633fe82030ce145c07a79659', '633fe867eec5ce6caca20c0d', '633fe8c2e0786e345062a224', '633fe93623ae21ebae3bbf1a', '633fe99756c380cc850752b3', '633fea22bb00c6af53b9ff9e', '633feaafcdc2a8272f80ab9e', '633feb2f587ad3acd6cf2eb1', '633feb94eb62c532f100a8c2', '633fec4e736bf458cc5844b9', '633fecd5c1889344ed7b9239', '633fed1cc7d448982d5da618', '633fed411670dbf4fd6f8dff', '633fed5d814d04c79c3062fa', '633fedaa199e524111cb474d', '633fee1aa85d24a1e908efcd', '633fee524dc2ff1c288f80de', '633fef92e96269876b026250', '633ff11f64a90e10a0bf2bf6', '633ff2316d4c53420efeb60e', '633ff2c045928a6c117714bb', '633ff33d075c36325d670f7f', '633ff9fefbb7406838a3982a', '6348072faf6478a8300d9ae6', '6348094901bc4c39102fdef3', '63482e0715473de3758f82c7', '63485e1e19d6812717d4c7d4', '63490fca70590efde5eb059e', '63490fdda990fc5d958fd617', '634adc0264d5f254855993c4', '634b172fec38187c4a5fecb3', '634e52cab9a22e636a50d4bd', '6350518f9e19238718c7a094', '6356c7f833f28e954ce8a98c', '6356c8b4eb4f28638a6ee358', '6356d70ffc70532355c9ad21', '6356dc0415dfcfe807e66dc5', '6356dcf36840def089318540', '63579137c1103ef3aec3b272', '6358e69da2869e416acebefe', '6358e6cb40f3a2413ff7033a', '6358f641f5e0bc963211b60f', '63599401cebf59aac273bb24', '635ab834356c6bbd9d864ae3', '635acc293fa156b5a2ed0e7e', '635acffa4cb150ebb855f10b', '635aec6ed6c5fd99b7bdf037', '635b31cb9fce0ef45ee73a05', '635c4f42512f22fad781cfee', '635fa5f161dfccb6105578ba', '6368baa67a71becb5d2f2137', '636e7355666b833d01c23968', '636e7ef49a9d54f8224724da', '636fa0b6c0cb63fbe063483b', '636fcfb1c098948655446dd5', '636fda2e4d2a01ee0cbb04c8', '63715b36685b2622ba56e2c3', '637255fcb67171f84e4f42e0', '63725869686e6026d3130cb5', '63727af162babceb5f53628f', '6373ca4e0734931a01366ae2', '637a66a831441f8116d9be6d', '637b30662e1183e22aee7371', '6381b362f909cb67f0954f1a', '638e5afa84088ec27f5381c3', '638f5c68aeef2aa5237526ba', '63909251ac70341084ecf17e', '6393454b38c36548afd75fc7', '6393841a0285223da672b179', '63978e361ee5943b516a8f75', '63978e64c3e913913c7ddfdc', '6399fdc966948f6b537b9762', '639b2253a3fc438980e859df', '639b26a907bdc943d7190917', '639b3b4013bc3754daf1beb4', '63a01d5249897d1ce5fcc4c1', '63a01d55919d8c212243d1b3', '63a33b422f7db17c51f6499b', '63a415ba9d7ddb4165d3c44b', '63a41dfc05aa84684084441f', '63a4374ba7126114dabb85b0', '63a4721874292cc0afb8a687', '63a482510d9a7fd0ed93702b', '63a5366d09707e5927d950b9', '63a5ae13d4a3401905954b4a', '63a5bd9e4b819d29d7b8f94c', '63a97aca4939331148964839', '63ac9a9b47285cdeaac87628', '63b46b640bb97a8b96c4b155', '63b83c0dadb77fcc68b0801f', '63c18489192c82bf0d1b31cd', '63c1ccda5d013cb9ebe876d7', '63c588485de7bfc1346b67bd', '63c5985bcb414d3b4f78eb25', '63c59f86df9e597cbbbc7c9c', '63c5aefe9898abb9e909a62a', '63c6ef62e0447a277b10b3cc', '63c9863c6368bd48043d03b9', '63c9dfe26d43fb6c45ae7c60', '63ce5685c96a21df0c652243', '63ed4641233889cbd248083a', '63ee3a204e550153f8eb051b', '63ee4d6829ad8e3a55688ade', '63efa93d496f186c0aa13837', '63f313a2ae5a3b4bb2f4ecbd', '640616fa50bfa6ab023e8af8', '640ee316c2bada8ce092da7e', '64109f7806ac151d8dbedc39', '6410abe4e8cd60f7b529effc', '6417f5dd46189cd7e1bee349', '641b0ca45b7985b1abc639c8', '641b243b14df3b7fd67f735f', '641b45b06f970570bc0e3871', '64215a8d20df58a42b27ed68', '6423db35e51cd6548e6f94d7', '6423e0829f9a081bb00dacaf', '642568f1d616a2fccd1571c3', '6425c0f7d4d42da77ec23573', '6433bc1a1abde825fcdd0285', '64364f52e24724c209614d95', '6438095933ac54a2698e7fe3', '6447f445138504b84219c3c9', '64487bc4e37834448d937956', '644a9f01819b7dac974d4f20', '644aadbc0076639c9c17c75d', '644be8ad5374d679806cbb50', '645a0d4c0e0c3cffd34ec23a', '645a3915e6e9987b081c9bfe', '645afe1242d054c87eed3d72', '645bdc01348c87bb8cb47ff2', '645ceb6d1676519c990e7bcd', '6461c30239f7f30e66299d2c', '646326f051ab2dd2077e9770', '64640499268ffa30abe51d5b', '6467f05ff97e226846df4856', '646b27c60c837b46a8b02007', '646ce6644a079d9b74d400e0', '646e4312c79d331a4ce5808f', '6475990488498b1458c33289', '64759910ee8ce851b484de0e', '6475994d5e4049a7220be941', '6475bdb4e413abd7004cce24', '64774fc29a882d005f7e6919', '647d961dcf05e86dab80565d', '647d9637ba5470ba10a40882', '647dd48bddca06330a62f1c4', '647dfc09c1378a98a1db5d4d', '647dfc0a2a002cff46ebf5e4', '647dfd0d0747cc4cb56969cf', '647dfd205259ad17ab92411f', '647edcd27f0215ef8e0027cd', '647edcf3c7af38b2f4883fa0', '64809a8188764abce778628c', '6480db093f4a3abcd042e873', '6481bc4b433216c3f35221a5', '6481bc4d04284fa1ad9abbe9', '648378c9f17245b3307e20f2', '648378eb0efbd5799fbb5306', '6488a812f6da4a86b319f2bd', '64894de8a6cc052b7f4d3a01', '648996e7f24d12d19f2f74a5', '648ae43b6aa0fee7638132be', '648af3fed854b4533b3295d9', '648b1300ccc64d3a1aa18f71', '648b45c5b203342c262673ef', '648b7ee010b41ae494305f6c', '648b85557b503d44f5bb45a0', '648bf1c4eae9fe585828e321', '648c072e78ca489bec6cb0c0', '648c21c01d8b78d4da9a4fc8', '64944a718c48be8bb9d2c315', '64a2b20319dcfb0d657b8997', '64a3f741fde1b02db29db5c5', '64ab412e96e4858e508978c7', '64ab46ec79ba3a3f9f859541', '64b193d3c01f259eab698535', '64b1c3b991eac8829b45988b', '64b6296ada990ed2e6250c6d', '64b654448e76a8cc1dce73a9', '64b66befca0676907a1ec075', '64b66f039bef7e8ae9bea494', '64b66f05461b548ac1de8ae5', '64b66f13e590a1d383e5bd93', '64b66f1f53f786ba87d14935', '64b66f2a81cdeec6d4a43f78', '64b66f39f2440e702c3ea6eb', '64b66f47c631b17dc9ef6cc5', '64b66f568d7a118efae035d1', '64b66f6193d0ca1616c68658', '64b66f71c631b17dc9ef6cc6', '64b66fa65c2d8251c879fe22', '64b66fbded9e45b04a5a10aa', '64b66fc9939bc4e5c90c914d', '64b66fd69ea9b0a8bd7820f9', '64b66fe366142f4996ccdf3d', '64b66fee96a10be2a210f0f3', '64b66ffd64acc975f12a0ed6', '64b6700e08a09b059533713b', '64b67011f81ffbe2e414060b', '64b6701beb060de0f901c761', '64b670280f11e7ed6fa94fe7', '64b6703771f31cea1d4f38b6', '64b67047808acc81b0d7cf42', '64b6705550626d399d6bf81a', '64b67062acdd506d39f74381', '64b670713d561de2917128bd', '64b6707d5137b8e8060d6f61', '64b670963633dacac5ad0850', '64b670a43c8f1168fffc2646', '64b670a53e6daa83893cda0a', '64b670b2101de92706312678', '64b670bd534ebfd17a3a2638', '64b670c86ca6ea9cd0d790fb', '64b8f4d80b0e4b622175ebd5', '64b8f4e0b7008cc32a008d4f', '64bace8d5d470a6a7baaa8a9', '64bc3dbad1071a81f7f9279e', '64bc42b07eabfe6151e757be', '64bf8eca368443ad46d6779d', '64bf8efcb3db1c8514f133aa', '64bf8efcf15003f6facc65df', '64bf8f0b71e678fd1894a854', '64bf8f0bb4844b4e2e6f3946', '64c0b4920a40bb8a96162adc', '64c0e9db04ed02765f336f16', '64c17dc34265fd1359962a8a', '64c17dc55bd8ed9bca3d4c02', '64c3210e7bb5f0a9972175ba', '64c3b8a568674f257291b042', '64c446d9fa11016096a06f61', '64c446dacd742b956b571f9a', '64c5271070f8335935dd112c', '64c7cd0ac6333d36a4419d85', '64c81be68818f4ae0f8ac172', '64c8267397a7cce9adeecaa0', '64c826748caebf09e24fbd12', '64c82712d7810b852cabc855', '64c827144620e1502824a501', '64c8271531ba066a327381f4', '64c827169fd4e55ea5b8075d', '64c82716f08b75eef0ccad22', '64c8271853824d2e96b63e76', '64c8271a118ad7ca6ad1cc1c', '64c8271c154fb0e795a4eed4', '64c8c155b612ca1be9ffd792', '64c8c1d3034553d56b5d3793', '64c9e2ae307d1b04caab7f40', '64c9e39a743ff437fab9d14e', '64cb264b88cebe1ecc36d70f', '64cb273717400f994a91de5b', '64cbc9c43e7f0344f99a9bb2', '64cd58f4e715d3035aca9e58', '64d09dd4b50ecb4b950db1df', '64d0bd4146099d65cddfbcc1', '64d141ac89e4830b54ed17f2', '64d1684316ea591fa2318f03', '64d16b2947e56dfa72dba0fd', '64d2519e0f3e4adf85e1cbda', '64d3531afdc6fb6c0cbe7ee3', '64d35332303ad4fc9dffe197', '64d5195c92a53a23e310f7e9', '64d524e4061d965bb44d7f1a', '64d55989c84b56b6d1151699', '64d6444b0040e1240107e3d7', '64d668fdf57445a68d4c6450', '64d95dd61226ddac00c1f23f', '64d9cba801043bab44290764', '64da86601c59b7dd980df35f', '64db35c1535927c8f68655d8', '64dd9c1d76a7807782a691d3', '64ddcaaaacaf6369b87613fd', '64de4b9811c31a3894fe78cb', '64e37791808502819a914e44', '64e45bbc705cbb44b096f4d5', '64e45bbd393461f633a01b21', '64e4692e7edecd7dcf00a75d', '64e61ce0a14dddeaec112098', '64e773ac2f554f69ce9bb27e', '64e79b3ca146c6ea8460b59a', '64e85599afbecba4a3b09b3e', '64e8d29fb3add9b751ad9843', '64ec3691ba8ffb01f4b54bfe', '64ec3dc9dfa14b6a00d9ed4d', '64ec44bd531914683f1a0d2a', '64ed117e2308a042e50e1e9e', '64edc1c1da86d242ddd791dc', '64edce380fea5b4bd8e9d07b', '64edfc5ab93abb1407070292', '64ee0767ef32a83bd953ff02', '64ef8e040e83c20954eb5531', '64f0100c80f7e6493dd56daf', '64f0118efb915d7bc05af124', '64f025da85fd3c8ea2f0fa27', '64f02611de531ca0647eefc7', '64f089a1db3faccfc533cbff', '64f0a249c20ad57ee6a9eec5', '64f15b358df7bf90b33af4f7', '64f1e5d7b2ba78bf0012e62d', '64f5f515df8dec4fe7930467', '64f73f46321b7d7c90f934e7', '64f79f48ec7b2336cb40ac23', '64f7a3c467f1f2d0c555b2c2', '64f81f8ff52fad5db16bb439', '64f82a5669ac77d49b65d8b2', '64f83168126d0c8e4560898e', '64f889ae3e6e9ac2beff5aa5', '64f8a28ed26db043a3a49b8f', '64fb890d07b957bce429e843', '64ff75d91a3779386ce0db89', '64ff8659fbda343d073836d8', '6500ed487d6a53dbc05e65ab', '6501e97dcc36048f47bf5c0e', '65026d267b3e9726760b8412', '6504c05901f08e674ef3b2b5', '6507ed06369072388e83e0e1', '6508bf2c091a0e27ac8c8a6b', '65098c02e496bac50d69ef6b', '650b06d796cd8766b8c78159', '650c117ced2ba5bf9e98cc55', '650c22ef421b73bfc5084a2f', '650c275b63ae6e8908bb044c', '650ded880cdcdb63f3e6d6a4', '65136f65f7240bd2ba4b325c', '6513714fdda63d45ca28b566', '6515d344918d6a84587478f5', '651c46b48ecc0d81fbfbb6a3', '651c87dbbfb3feb9d7e61095', '651e61624fb51a94d2cae0f7', '651e7759f5cd95dc5864eb42', '6527a287ffc43739464c180e', '6528bbc1b8729ebb21b05fbc', '6528d192add1f3a298fa6119', '6528e2ca1ac9af81cbccf1e7', '652c37538751db7140a0116c', '652cb1748f360b525511693f', '652e35b4659a1cac6a6cc7a5', '652fd1753a38a5356e13ba8d', '6531fa57b7edb6c002b800e5', '6533ee8c0ad188c0b6b16395', '653c917b9971656a61f105c7', '653cc626e73ed009f8e016e6', '653ddf2064c48ed4c86a8c6f', '653e3e8fac6db1dc159ac405', '653f178696291a18cc6873b0', '653f198a561b9efde661055c', '653f1b744f82ff189926035a', '653f1b77c1090397a32b6979', '653f1b78e5e7e24debcdd89b', '6540a12316771e26acbd45cc', '654285889d7ef29454af1bcb', '65428665499c00a9c4ec327b', '6545eee0b03d278bf2908089', '6545eef78aa1b2795f7863fa', '6547e7a90067c4adbb6d30f2', '654a3c97592fed67f6160951', '654df70ad03ff5af1348e7b3', '655316e23e57c80646bc966d', '65531730ef5c8c1e54481db5', '65531750ac364c3eeb77f790', '655341c1ef0c27b71aa66e9b', '6553be2c130e91294f47b7eb', '65546d9c21a527f30d0c93d7', '6556309dcc4f09e80664bcf4', '655824a1bad92dd5af110f35', '655840df3324e68a3568d167', '65588f35ae13d4be8544c84a', '655d884c9a33197e30b76ec1', '655fcc95fbf727ea7ca1d4fd', '6560d56ef9efa9ca598b0798', '6560e5815d27c3cdb1eb79f4', '6560e58625cb4f8af860f5b6', '65690ce1ae12070f7646d522', '656aafd0e93efa420f74123c', '656d9daf15ed534a0fdb7b6a', '656dc747752db42a206cc595', '656df3583acf01e754ceb0b0', '656ef7c06db516dee1063af2', '656ef8a1b5f595a5ed95747e', '656f099bf01a1b4552ccc1dc', '656f208d1c8d89f5f9163b17', '656f30c505e8b346f16b150f', '65707ab6e66cbcb43bd4f250', '65707c2a382a678a5e555538', '65707ce8de15507d25a1b32c', '65708b6882f70086411c547f', '65708b8fbdd57cf6a24a1e17', '65708ba10e79097bb5108b8c', '65708c81e7887638b9516617', '65708cb730ef54f76f707779', '65708cbaff7263ec6e06baa7', '65708cbc25cf0a3e497cffb4', '65708cee75cb524822443805', '65708cf911f044ba6f739580', '65708d02227a55ebcdec684b', '65708d3bff7263ec6e06baa9', '65708d3fec7eeee20ce02403', '65708dace1108a2df26f5c5e', '65708e34d64c3320402789ef', '65708e3b91fece390b1a096f', '65708e71b750327c5b8a72ab', '65708e83c28406ffed5c2e76', '65708f49f703fa2e5a1feae5', '65708fa9137ebcc0549ac8d0', '65708faaee010f81d3f9b3b2', '65708facb61096b6096093c1', '65708fb3e1b74572dd4b540a', '65708fbf9f1cbb0353d22e50', '65708fcf38f1bd5e40046c7a', '65708ff435400450988945f1', '65708ff8e1cd2e25819001c6', '65709030eef4846c00f17bae', '657090fe5e8c659c7e5cb094', '65709116c5f6d5f3aff5332f', '65709118190b5ec86b9caf62', '65709120ed2b0db3696f67ac', '657091236e0eec6ed07a100c', '65709127d115350fd0249f37', '65709134e73d6efb17198811', '6570914fe73d6efb17198812', '65709158fffe8e43cf5395ff', '6570917294346eb331d6f0e3', '657091d54d2eacfef981b927', '657094d16c1c591be50d6abb', '65709a49ed44fea53e9aeec5', '65709ca13f7a17df14af4912', '65709d6711dda25336749dda', '65709d698f91ad4dbf340ffb', '65709dd6926a5676de0e2a19', '65709ded7d8a5ce8dba3444a', '65709e14974bdb5d6dbda091', '65709e1db42d958b88f07ca8', '65709e245df26a29e78d740a', '65709e5d4c59f8ac3f86f615', '65709e736e1768898768814f', '65709e8b31eda9b13196277a', '65709ea58a4b251d0f7aac7b', '65709ebd65cdc059b8e373ef', '65709ed8415c89746a234d89', '65709eee2f74978bd15d60a9', '65709f0bbdd32cb4b343a12f', '65709ffcf3ffe737f8cb8dfd', '6570a756ee3c8ce2314e235a', '6570a97ed5b9c8108b7f9298', '6570bb36fd543359ab65ef91', '65718fe32b1d1c2b34416c4a', '6571f47272b64f9db7f3a254', '6572250f298d2a69b238cf72', '6572ecbfe8aef75cf0c77f79', '6573042a8a814b3833b2de12', '6576bca4bf04e7d8197d32a3', '6576bceffed76b8fd0b5e665', '65794c2111e3cf7b64379879', '65796ba7f1bdd15feb72dae8', '657a2c924ea0e3e9e95e9433', '657ad1e4a481230daba5c061', '657c604ec955bcd4f65f5781', '6580d422fb57aab8e21c1f39', '658176d93bed642cb715cd05', '65819c782f730303fbf978f6', '65819d57ad84212d34b216da', '65819d8b1d340924fb83e7b0', '6581d83bfd115be1f92d75a9', '6581d83e20634ac0d58ceca9', '6582dab353de943633b9ec35', '6582dab4e8a8d2a8197527f2', '658957c2bba4df9e9c7e2306', '658957d634894cae706e6177', '658aa57b6951aa256ac91991', '658c2881de6bf210213b16b3', '658f04cbfad6edb57a71a5b9', '659007ce9ee62457a6519b3c', '6590451e97969a31c5d5b7b0', '65926d953adb7c379a0aebc3', '65951ce29c954d984da5c961', '659c1be7320df4b756f2adb2', '659c6b890244c991aaef7f90', '659cd6d97519b21156635bdc', '659d5f9099ecb2a5a3c38b84', '65a119bd5035be71f1530a30', '65a11e3746e6adfeb24af445', '65a343fabb20abd246d5d9ff', '65a41889dfc3c004ef9c6ec8', '65a418b5f2b19073ab89c895', '65a44313bdf0d08977925d7c', '65a4702bda23cca5d43142ac', '65a4f9f85b5bd21036e9c8a4', '65a5a51fc84ada06ec6306e3', '65a607039b6f96da6d6acedd', '65a7ad36bec071156cc9a8c7', '65a8c078d096fab75ef94460', '65a8da47c4229bb3c5f863f6', '65a933dc5b451cc43bf8ee06', '65a961cd7f7bdd4535be76cb', '65b0d73758504849f3f41888', '65b239b451841d46725b938a', '65b284094143ec863fb3d218', '65b28549b64e1f9b1bf218bc', '65b35cbb1133c905d53076ee', '65ba13fbd931bd814c43af88', '65ba186c2ed71b3ab4d788c4', '65ba6e96bf39f49c3acff7fa', '65bf9b680a31915bed66fe9b', '65bf9b6b83552213615b08b6', '65c09e6935e446f36ee67d16', '65c0fd55c5d3c780a06aa227', '65c2068e125ab9c038a7f362', '65c257c4178a74335a9eb520', '65c4593f9ed8f1cc0798dd12', '65c45d85181421d82767f286', '65c4647794683023fab54394', '65c4680f310d3c84523c2498', '65c5e50dda752af9eab50933', '65c6090b8d4b149e72aa3f31', '65c7b86fa120d19bbc88f367', '65c91f2b7c03b480379ae4d1', '65c994c8b145925072b6583a', '65cab679e6ff8544ecf11962', '65cb4768b06f4da2fba5959b', '65cb476d0566c2d07e474df5', '65cb476d935dd560b4a3e938', '65cb4772c3d3ad1f7accc98a', '65cb4c8edd9af58ad3bd3568', '65cbbbae38ec92abff597e47', '65cbca0c00fdf5e10fb48df9', '65cc8e8c35b12698deac3015', '65cccf011d1e0c1dfe83404c', '65ccf39352648e8458ed369d', '65cd1f7ec0b162bf4f3a24cc', '65cdb5b826d8dd5eba1cf8c6', '65cdd03530de996755c30f3a', '65cdd04241f862569a73a536', '65ce54e2c2845b199bb10a25', '65cf4b438c3b886f23ead1b9', '65d065deda59dc1fc76e1cef', '65d46aad8585653d4eaf5c61', '65d4a0a7bc75db1f978c56ec', '65d583f6501aa4a193cb2adc', '65d586d7eefa5f0192048f09', '65d616b1eefa7aa972db5504', '65d7704b1bc8d4fc2d15f8be', '65d8ff6840d8158ba18ba9ae', '65da2e10a6bc9c007c0c0e32', '65dc77cdd2ffe8938fa7081d', '65dcd7c0e41a000110f95a29', '65dd60403573b9d3e35d9d25', '65dd6093a886672fb45693d9', '65ddb8559a7fb902ff7b6b63', '65ddd4a87f6360ddef51226b', '65ddef7ecafd3674a3c69e98', '65df437e63d56224a22e7050', '65e188eefc338622653f8804', '65e1c7a8436b3ef37c0ccfc6', '65e47aa9fa69e853c978b0c8', '65e47ba21c8abaa91d183ad9', '65e6a5f33d884544fa40856b', '65e6c31bde674f8c8c5ba04d', '65e6f2c8f044bef9fde9041f', '65e6f37cb409d5e35f1af808', '65e75ec1fb99e7e5df7414d7', '65e86194b8b205680605ef3a', '65e98f23da70e605c92ed98f', '65ee93336bc9f9c8188a376a', '65ee96a04bc644589ec3298f', '65eec345cb766ad558df43ff', '65eef971d4ef2c0660635c90', '65ef25e8bf9e270fae148298', '65ef3d025e2449674e623ae7', '65ef4e1568d2b25a7844748f', '65f05941b40e4689441ec9d2', '65f090766d1e1dff502f640e', '65f17f66772585fd1fd38929', '65f1c06e8876e33699dbf7f4', '65f2a8b8e7b04d73b32c17cd', '65f2a922146da15f711f3e8c', '65f2a949ac47d1a435813c52', '65f2a9652fa171566433aaa7', '65f2aa5d5904399d8676ada1', '65f2aa8e06971228fee2d2d8', '65f2eb395e34c23db0028a97', '65f422be7fd82b8a6cef5ac9', '65f57c83b19e7f2132659632', '65f8136ba71c4fcd01964aaa', '65f818642d29b84d812a6654', '65f836ae206a0f1efc242600', '65f8375d2828d9f414c9baf3', '65f965e78c86851dff663913', '65f9801e719106c8fa1f4d06', '65f9a49a566f081a518fa465', '65f9b66f8a9de3739701158b', '65f9dd36a80750817368ea51', '65f9e6ef3361e138aa1fb531', '65f9e7f68f58a4a0e3c7fffa', '65fa5c186e2e4496d0d4101f', '65fa7b4d570a3add83c348e8', '65fa7bd30543961d8060e122', '65fa7e64483fe8cd429b5cf4', '65fac01e5952f7f0c71f43bf', '65fb461800291b6741775dea', '65fc6576a64ed102b5b18790', '65fd63087c04fd9dee1b45de', '66014cf07d433df8f506faef', '6601788cff0e704171219cdd', '6601ba32b14ccd23a46ccd0b', '6604069850dc834391ee2eeb', '66040698fad13b9878da2eaa', '6608297f107807a45452733a', '6608aaf7ca0e965e593ed1d4', '660b176a98b0c92ba5a962bc', '660bfaad3b226529fae81af9', '660e22c6ac6105fec7bcc159', '6615b94cd046482fb0a36736', '66164fcde697b27ff6f19825', '6616a8c8db96521f89e83a6c', '661b0e4b314db9befbc127e3', '661b0edd4de157a23c431364', '661c4bd8bff27492dc1e380a', '661d8c3e4aadbc315f1eab28', '661d97bab11ed8d2c64cdfd6', '66209fd0ca1d09ea82c71f67', '6620b36d87b9ee5c7dbb6860', '6620f0659fd0f3ab5526ce0d', '662172953b276eb42fde850c', '6622555fde94b57cb4cb468f', '66246ff49ed29ea9bb2bf122', '6626e5ae070d276ca5e1d626', '6627a4166a9ad399ac429caf', '6627d32a080d57fb9b8bce5f', '6628e16fb713bafd507e5364', '662ab40d1f6e2e9c38b65355', '662edc10a9158fb6cbe169ee', '662f2e629dec6a7b033ee246', '662fda252781e2c16172ecca', '66312f9ab880e28122921794', '663d2869e0f3a42bbddc42ff', '664159d753d614e3b46d46c2', '6641b2af1018105ca9e05f71', '6641de0f085ac4fc0c55aec4', '6641ee545cefcbbdf2b35cd2', '6646149c91ffc734760e4e89', '66463a8d4b207e87a5ca4d8f', '6646f915a52131411dcda3d9', '664701e89371ec4fd04a1afa', '66470d32f3d41765c5c976f9', '6647908c09468f42bc1249f1', '66480cb1cc174fd804e0cef9', '664ae0bba7216fa4c9e46276', '664cc3c1938ed0b9abf026a2', '664fa66213d6bb0091d6da0b', '665182d791bfc08412ec2c0a', '66584b1059a41e0b81c98c98', '665a1fc3ee450e5f92ad6322', '665b42f777a1cbf2d8b03f6e', '6662038a011ff863c55d507d', '6662203cee6d1ee59eca9ba9', '6662d151b9773d2765dd4f59', '6662d21f6eb29fe72eef78a7', '6662e21ee7ef8ef2d6e40ae2', '66671a01946385446a23e240', '6668b85065eec626e4766a38', '6669b3872f841e97f437f7bc', '666a6ae5bb437ef87caedb43', '666ac558d08da6cfb3ba135b', '666c5069942e22586678e846', '666cb2c03bfab5a7876ee623', '666ef70ef24edc2334eaa205', '66729e8ca578b61e7fe82e17', '6672a50222375966a0ccd9d3', '6672c3147b1c9093e7b1f4af', '6673d86998fd3559adf8d655', '66746fdf1f78721bb353ba7a', '667505ab398e1f558e33f3b1', '66758dedab66e77443c4785b', '6675ae9acdf332298d4477f6', '6677e2ee9ffb8273f40f278a', '6679271d75f66267fe0e5c98', '66793d4fc20f3888ca20fe66', '6679760b1535ffeced9cfea8', '667a6171388b456ebca12bdc', '667e67d37bd6cbcb85853618', '6682abf4fd4b3b7f10a778a3', '6682ac0355982625657327e7', '6683fa870cbc5bd54df046fb', '6687f19eff15590621dd48ba', '668c9a150fa545d3147b575a', '668ce0e438ae6e904191a8d4', '668ce10552f7bbd9fe8e4313', '668cfe88e4408eff40ca5eec', '668e5353bb297021f2c812f9', '668e6bd1b7048a19975989d9', '66911f5e1df065e410143bab', '66949e2f9d02c3634af3689c', '6694a8b5c6e74d011d365eaf', '6694c387cdb8ddd8c1baad16', '66960694f735e2d25dcea7fa', '6696401c81cae41e046a6a6e', '66964336d4e619f1923fbe7f', '6696439162b61a376502d01c', '669694b25d4fa94af57e271b', '6696951cef4fc8ab6f7207a9', '669764c71040503ba11cefc3', '669c0d4c2e7c99bbcf906d65', '669f55ed91ac62173bdd8fd4', '669f562a5d60ba548b103ba4', '669f882c5109175fe742fe58', '66a0533b93086bdaf78cdcc5', '66a0b505f6ffa19e6978de60', '66a0ced5fa22f82c2fe5d827', '66a1221a4c2d00084ba3bcba', '66a65d4f674ea56d9d35c93b', '66a6653d3bcaac5485a826d7', '66a74deb8ae7dc0919fab86a', '66a7553fd2fda0bc326ad0de', '66a765979f4aef2a326ae999', '66a76d40be5a62aab8cf8c56', '66a88e6cbc9c988f2e974516', '66a910ebfaae800237ade759', '66a9dc385e9a61bdfee6a4d2', '66aa153b000fe60e6fb7e7df', '66aa334b7fc91f0b45a50743', '66aa9be0d89cea811f363544', '66ad032749f2c0b96b5db569', '66b014f647ffb47ce39c71ff', '66b1ebf1bb045b0a9ac380d8', '66b4f3ac391f5bf2a9698e64', '66b5071f2aa63cb23cb6441b', '66b7ad525082bf9ab42ce6cf', '66bc5ead78716b92e43c1f2a', '66bedb276b3f5084115c1dbc', '66bf67dab92d1f0b87fd0417', '66bf8a2b7b8312e6727e18bd', '66c235b05007103d3e3e7038', '66c25619381b2a7849a5f366', '66c345fdd97c3a26a18d9441', '66c45c9502fac52d01335ad3', '66c4a24e4891a611deb36702', '66c4b26951718ab675296d47', '66c5ddea9ce78f3a2401d6cb', '66c76a410bca940a8cb84f91', '66ca0b8a0f04023037784b85', '66ce46378b65e6999b05e29c', '66d0a3b39d413763be480fd1', '66d1a22a590c82565cf3676e', '66d1cbf1ba6bd4a6eaecb0a2', '66d204f1d658869764c07d47', '66d26f755e2f49a3ced7b14e', '66d5e47820f3cbebfa24a298', '66d6af13616de9ef06fa4dd5', '66d6c64b00e2d196ee33182f', '66d76a56e630fe7d3076ffc8', '66d7b3f068ec9e834870e27f', '66d83373d1caed89028bc70b', '66d875a3c5b05d0e8dc29ffb', '66dae136fe855042399d3e28', '66dae150ec54aac7f03461e6', '66deab9db2c98c27aa4b5576', '66e113761932791d673f4379', '66e14fe8e606bc88f19cf19b', '66e1784dc458381c4f5a1ef8', '66e179d7520251e162f342b6', '66e1a584c1f2be27f7d11527', '66e21f753ef692bf60ff54d7', '66e32107ddfe23ab0f47195f', '66e3e60aa174e545398979fb', '66e42fc62878928200520cb6', '66e834ca12da67c3ffc6831f', '66eb32068c2650ba5ca663dc', '66ebb199bf80b3e8560779db', '66ebda7ebbe759bb12cebd4a', '66ebe08841abbc50444fbcae', '66ec1d053abb8d066f52e871', '66ec83942b81e4ceb340a5ae', '66ecf80a1f8cb3abb7b874c6', '66ed3d4f6c14a7f20638ec81', '66ed5a9c197554aebcf74460', '66eeef55e2c1ec3a4fa60ef4', '66f100d791f9f9f6ab7b4f24', '66f126dd559ebfb24ccbd49f', '66f1280003d54c1cb29cec18', '66f2aa473dedb93f51b568cf', '66f31b9a0551ca166c872292', '66f3d4dff860d18695313fd6', '66f704e78d659ab692fbb616', '66f79d146f77699c76532842', '66faab0f798418fcb1d64f2b', '66fc4dc786e2c423e5ff8e70', '66fd391fe0318d98f8af0eb4', '66fe40a79474f0b0180b8ea6', '66ff35047c2d98c0d9b6d383', '66ffbebaaa259cf7b7dfe08d', '66ffcd4576abe61ebb5d6005', '67001accebe7ac7738d5053c', '6701a76eac930cce841d04f3', '6704c6be60820192bdf044b5', '670539745f57431bb4c657ba', '670554e431658456dcf6c26b', '6706b52d4f2c0f49ac7cece9', '670ffa1c505fcc114556837f', '6710059101b736e38b9cd2b0', '6714286a840c6e486069bef1', '67154dc5e5c8ff8d7546948b', '67169021f448c48c2786c3ef', '67176428e4871c2726288178', '671773ca6756ce1c876f2a07', '671a90af1a2cfe22b737b2cf', '671d5c3645fc8a18ad6a6248', '671f88cacb0b22d6e4c180f7', '671fb2564a37133fb0a88d97', '6721dd99d50e56e96ec6bdbe', '6722577e757627411ce76bd2', '6726b02397b2b6316fe25c5a', '672d552ee02b6697ba5e6474', '672e171a3896d93dab235558', '672e466a2f3520bbbfe6c9a5', '672f88f9190a6de9940c4e58', '6731136d422eca964d89a255', '6731fb12d3f39a471c1c6c67', '6733161f299ee6721c92568e', '6734c7ed9f5c0b30bce7ad42', '6734d8cd5182e70e037f271c', '67380eff4b589d9ae37a68bd', '6738f1e426ca184cec873450', '673c582be52f89fb7bcd8a59', '673c82cbbf082b90bbae2d49', '673c92da4d5e24f128eeb05e', '673c92e0aff8460480c966a0', '673c93cf09f906b5cb824a56', '673c944f09bdd4428998a5af', '673ec88967c80a1f2767bb3a', '67440c6c49e3680d57b29290', '674ab0533a99927e9b6b10b8', '674f33517d92691127e4b3bd', '674fa7e00e63c8a1f701fb53', '67508be45f644e4ae2a878d2', '6751b06f651a46b4891c747f', '6751b336d5c878b910c2f5ad', '6751b35da1118d97c16b9fd0', '6751b35e1d33bb3161cd2696', '6751b5815195293815ed8187', '675366f6752767d13bcff5c2', '6757280e8556585092e41874', '675ab84acc5b6d2e4b83a1e6', '675c09cf3bde03946b0c024d', '676041a7afcf16fd2894ad64', '6762beefe124b068a2a44f8e', '6763151578f3ce5dfbb14bde', '676652f6a7ad2f6c63bac5d6', '676652f7393d00df56cffdc0', '676652f78bbff8f3a7dcb4c3', '676652f7a55e1301c9233e6c', '676652f7b1080740c101e9e9', '6769a86ee0b9c79ebd4a6614', '6769dd2ac0f748c372512eef', '676af0f2663cdd2f25ed9e80', '67729edc25a16cc679b32b06', '677cd54761ae1299ecfe27aa', '677cd56c6fd3f2632e1b4411', '677e0f2b70102dbe9485da97', '678638141cbf938dbca1090d', '6788b4daa6fb8eeb5e2eccc3', '6788b50e7be5db04f049def4', '6788c5a5e14be820846090a9', '678f0dbdbc59dd2ea5656dcf', '6790b7ec6082fcd92eb0d93e', '6791038138d8832a699f084e', '6791f6def3d34704ec47561b', '67972e91ec7476bda7b214bd', '6797ba7c2feca94bc6c4ac1a', '6798a366aede6b105b258201', '679a3beebc879c3c3d9bbb8c', '679a41b1f84094e260c7364c', '67a06a13cefb52d72853cb73', '67a190facf12f98043d32b8a', '67a1da548e9df794c0484262', '67a219f7c06d372f86335d64', '67a24d4f4282073f325c0970', '67a3a023230240bb77a11434', '67a4cc99fcd39d24100ca29a', '67a52f204b29036fb39890a2', '67a91100f303686a27cc8a1e', '67a99faf633b5eae5a909266', '67aa6fdc9d11e3813a040447', '67aad6551764d380d0f060a8', '67ad20d5eebfab02de73ce33', '67ad4358e7823ee5091f4215', '67ad879da2b978d686ffd8c5', '67ada51ddb9ac69ecc3309b6', '67ada5610727b81262a34ee0', '67adaaacf2cecff0ff090c57', '67adbc7a8e2a82411cc8f21e', '67adffd6ce78220116cd907e', '67ae7c9e50bcf1570e859af8', '67ae7cf68d1fbbbb9f07e6ea', '67ae7d1bca1eeaa7a019baaa', '67aeff1201885e8fbb5e90c2', '67b8778e5ac4d2d6400ac839', '67b8c8c79c31a847649a026f', '67ba1f9cb65c4e646bd20f12', '67ba25b2da44abaa51df5497', '67ba26921050c137f83c3b8d', '67ba28fe2d815b044c734397', '67ba2beb8262c1be708cbba1', '67ba2e8b3e8576d4967000de', '67ba2f64b573ebab9d4f3409', '67bbf7ea148b4aeb55d955be', '67bca3be361fcda6441b562e', '67bee4e567d977e4a07d1fe0', '67beee0328f327df37f068e7', '67bf76d3cd9e39241c181126', '67bf76d64a1012a8dc2f8273', '67bf821001fe07eff29d02ff', '67c0bcfa5d5c50c904670867', '67c1c2815eebab2620b4ad4a', '67c54116a493b51c1a8422ac', '67c678cb6b4722c6ec6eb1af', '67c6bb5aa601e91b1314ff44', '67c80377889901edb281da72', '67d2b583341bfc2f89e2c844', '67db1577532e9094852895d7', '67db2bceaeb33fde1496fef2', '67dd406e35e3344749124f94', '67e1271ac9064e29484ec84f', '67e1c4092151bb306c3b85ef', '67e1c9e582110e45c6db12dc', '67e2845001f26d14a7a31ec4', '67e3af5245bfaeab49f8efe7', '67e46065a414650e2ec93230', '67e4c1dd7b36e6d82df4d612', '67e51b5b16ae46dbc64ee75b', '67e59d30fc2fe9b7ddaded28', '67e6cbf320876638e1b066ff', '67ea97fab81001ece19f54a1', '67ea97ff5aacbdc125ff5175', '67eb87f8beff23dfe3508000', '67eca9b9739aeabdab40759b', '67f4036d091adaa2882cae21', '67f4f48d37821bbe86d273e4', '67f5ecdbeb1e17428cea2fbb', '67fcd35e323b628bf4b30fb8', '67fcf02e1f38f0bd9523281a', '67fcf21d5c109d1d5c2b2a3d', '67fcf25308f363e7380a28b8', '67fcf3654a803c595d6ca4c6', '67fcf409e0525d335590d3f5', '67fcf4551457439372556369', '67fcf498495cb49f7043ddb0', '67fcf49909e5638f66dbf5e9', '67fcf4fed446efdfa04749cd', '67fcf539f4281bc94584140a', '67fcf580989d7e29fd5f472a', '67fcf5b3fe865986f36a2005', '67fcf5ee65aaa8ffc42fa627', '67fcf63186b38b5acb36e1e4', '67fcf65e597ec61559e01409', '67fcf78856748d33ccbeed22', '67fcf7ab89af972acbd27663', '67fcf7c3492cb125d16a39b9', '67fcf7fd9ab8548a25eff013', '67fcf8301ab1c1c942d23beb', '67fcf885bc1017bbb1f756f3', '67fcf8cc378c7e4aa641d7a1', '67fcf90549323491ba5fb916', '67fcfa3c7577544896a499a5', '67fcfb17c6fe2b509369c54f', '67fcfb53dee3c81445fb633a', '67fcfcb7450202c7abbcbab2', '67fcfce93e9fbcfea945593a', '67fcfd2253214839d3b31b1e', '67fcfd559948cc790c09ef30', '67fcfd83c85519d305b4f935', '67fcfdba449621e3ee719f41', '67fcfe000c9d0ea18d25eb76', '67fcfe667e235bb7444855af', '67fcfe9924874b24e6985f4e', '67fcfeeb7f900a8125c6f990', '67fd000d5f987184223898b0', '67fd00449c11540620717226', '67fd008ad722483de4613192', '67fd01d24742963526d93ccc', '67fd02842ee4d0ff69a0b851', '67fd06c310f2f0e2a84d5704', '67fe9eaac7555ece2e4d5fa6', '680034fc84efc0751b3bc07d', '6801c04caaf185db4d81e3f5', '6805c789f7969ee1180d42dd', '6807a105799b0b04432a5753', '6807e1c45dd52410facbed8f', '68084aaf4453be8e744d52e4', '680851e102d72c44ae0158bb']
    # }

    import json

    # Read evidence from JSON file
    with open('../query_input.json', 'r') as f:
        evidence = json.load(f)

    
    encoded_evidence = process_evidence(evidence, df_mapping)
    
    print(" LOADING SIAMESE FEATURE ENCODER MODEL...")
    print(" Loaded FeatureEncoder model")
    
    # Query!
    results = query_siamese_with_features(model, apt_embeddings, encoded_evidence, apt_ids, profiles, apt_name_mapping)
    print_results_table(results, encoded_evidence)
    analyze_feature_matching(results, encoded_evidence, profiles)

if __name__ == "__main__":
    main()
