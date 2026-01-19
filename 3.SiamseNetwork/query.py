# siamese_query.py - XGBoost-style output with FeatureEncoder
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ---------- FeatureEncoder (unchanged) ----------
class FeatureEncoder(nn.Module):
    def __init__(self, num_techniques, num_software, num_cves, embed_dim=128):
        super().__init__()
        embed_per_feature = embed_dim // 3
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cves+1, embed_per_feature)
        total_embed_size = embed_per_feature * 3
        self.fc1 = nn.Linear(total_embed_size, 64)
        self.fc2 = nn.Linear(64, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech, soft, cve = features[:, 0], features[:, 1], features[:, 2]
        t_emb = self.tech_embed(tech.long() + 1)
        s_emb = self.soft_embed(soft.long() + 1)
        c_emb = self.cve_embed(cve.long() + 1)
        x = torch.cat([t_emb, s_emb, c_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

# ---------- Load everything ----------
def load_model_and_data():
    with open("apt_siamese_feature_model.pkl", 'rb') as f:
        data = pickle.load(f)
    config = data['model_config']
    model = FeatureEncoder(config['num_techniques'], config['num_software'], config['num_cves'], config['embedding_dim'])
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
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'total_features': len(apt_data)
        }
    
    # APT name mapping
    apt_name_mapping = dict(zip(df_mapping['APTnameEncoded'], df_mapping['APTnameOrigin'].fillna(df_mapping['APTnameEncoded'])))
    
    print(f"✅ Loaded profiles for {len(profiles)} APTs")
    return df_mapping, profiles, apt_name_mapping

# ---------- Evidence processing (XGBoost-style) ----------
def process_evidence(evidence, df_mapping):
    print("📊 CONVERSION RESULTS:")
    
    column_pairs = {
        'techniques_str': ('techniques_Origin', 'techniques_Encoded', 'techniques'),
        'software_str': ('software_Origin', 'software_Encoded', 'software'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded', 'cve')
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
        cve = np.random.choice(encoded_evidence.get('cve', [-1]))
        evidence_vectors.append([float(tech), float(soft), float(cve)])
    
    evidence_vectors = torch.tensor(evidence_vectors, dtype=torch.float32)
    evidence_emb = model(evidence_vectors).mean(dim=0)
    similarities = F.cosine_similarity(evidence_emb.unsqueeze(0), apt_embeddings, dim=1).detach().cpu().numpy()
    
    # Feature matching (XGBoost-style)
    results = []
    for idx, apt_id in enumerate(apt_ids):
        profile = profiles.get(apt_id, {})
        tech_matches = len(set(encoded_evidence.get('techniques', [])) & profile.get('techniques', set()))
        soft_matches = len(set(encoded_evidence.get('software', [])) & profile.get('software', set()))
        cve_matches = len(set(encoded_evidence.get('cve', [])) & profile.get('cve', set()))
        total_matches = tech_matches + soft_matches + cve_matches
        
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile_size = f"{len(profile.get('techniques',[])):.0f}T/{len(profile.get('software',[])):.0f}S/{len(profile.get('cve',[])):.0f}C"
        
        results.append({
            'apt_id': apt_id,
            'apt_name': apt_name,
            'similarity': similarities[idx],
            'tech_matches': tech_matches,
            'soft_matches': soft_matches, 
            'cve_matches': cve_matches,
            'total_matches': total_matches,
            'profile_size': profile_size,
            'profile_total': len(profile.get('techniques',[])) + len(profile.get('software',[])) + len(profile.get('cve',[]))
        })
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# ---------- XGBoost-style output functions ----------
def print_results_table(results, encoded_evidence):
    print("\n📋 QUERY EVIDENCE:")
    print(f"   Techniques: {len(encoded_evidence.get('techniques', []))} items")
    print(f"   Software: {len(encoded_evidence.get('software', []))} items")
    print(f"   CVEs: {len(encoded_evidence.get('cve', []))} items")
    print()
    print("🎯 SIAMESE NETWORK SIMILARITY RANKINGS")
    print("🏆 Features: FeatureEncoder embeddings + Feature overlap")
    print("📊 SHOWING TOP 50 APTs")
    print("=" * 140)
    
    header = f"{'Rank':<4} {'APT Name':<35} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CVE':<8} {'Matches':<8} {'Profile'}"
    print(header)
    print("-" * 140)
    
    for i, result in enumerate(results[:50], 1):
        tech_str = f"{result['tech_matches']}/{len(encoded_evidence.get('techniques', []))}"
        soft_str = f"{result['soft_matches']}/{len(encoded_evidence.get('software', []))}"
        cve_str = f"{result['cve_matches']}/{len(encoded_evidence.get('cve', []))}"
        star = "★" if result['profile_total'] > 100 else ""
        
        row = (f"{i:<4} {result['apt_name']:<35} {result['similarity']:.4f}    "
               f"{tech_str:<7} {soft_str:<7} {cve_str:<7} "
               f"{result['total_matches']:<8} {result['profile_size']} {star}")
        print(row)
    
    print("-" * 140)

def analyze_feature_matching(results, encoded_evidence, profiles):
    print("\n" + "="*80)
    print("🔍 FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    all_feature_scores = []
    tech_list = encoded_evidence.get('techniques', [])
    soft_list = encoded_evidence.get('software', [])
    cve_list = encoded_evidence.get('cve', [])
    
    for result in results:
        apt_id = result['apt_id']
        profile = profiles.get(apt_id, {})
        
        tech_matches = len(set(tech_list) & profile.get('techniques', set()))
        soft_matches = len(set(soft_list) & profile.get('software', set()))
        cve_matches = len(set(cve_list) & profile.get('cve', set()))
        
        # Harmonic mean scores
        def harmonic(a, b):
            return 2*a*b/(a+b) if a+b > 0 else 0
        
        tech_score = harmonic(tech_matches/max(1,len(tech_list)), tech_matches/max(1,len(profile.get('techniques',[]))))
        soft_score = harmonic(soft_matches/max(1,len(soft_list)), soft_matches/max(1,len(profile.get('software',[]))))
        cve_score = harmonic(cve_matches/max(1,len(cve_list)), cve_matches/max(1,len(profile.get('cve',[]))))
        
        all_feature_scores.append({
            'apt_name': result['apt_name'],
            'tech_score': tech_score,
            'soft_score': soft_score,
            'cve_score': cve_score
        })
    
    # Print top 5 by category (XGBoost format)
    categories = [('TTP', 'tech_score'), ('Software', 'soft_score'), ('CVE', 'cve_score')]
    for cat_name, score_key in categories:
        print(f"\n{cat_name} TOP 5 APTs BY {cat_name} MATCHING")
        print("-" * 85)
        header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
        print(header)
        print("-" * 85)
        
        top5 = sorted(all_feature_scores, key=lambda x: x[score_key], reverse=True)[:5]
        for i, score in enumerate(top5, 1):
            apt_name = score['apt_name'][:29]
            print(f"{i:<4} {apt_name:<30} {score[score_key]:.4f}    ...")
    
    print("\n✅ SIAMESE NETWORK QUERY COMPLETE!")
    print(f"🏆 Most similar: {results[0]['apt_name']} (score: {results[0]['similarity']:.4f})")
    print(f"💩 Least similar: {results[-1]['apt_name']} (score: {results[-1]['similarity']:.4f})")
    print("📊 Using Siamese FeatureEncoder + feature matching")

# ---------- MAIN ----------
def main():
    print("🚀 RUNNING SIAMESE FEATURE ENCODER APT SIMILARITY QUERY")
    print("="*70)
    print("🎯 FEATURES: FeatureEncoder embeddings + Feature overlap")
    print("="*70)
    
    # Load everything
    model, data, apt_embeddings, apt_ids, apt_to_idx = load_model_and_data()
    df_mapping, profiles, apt_name_mapping = load_mapping_and_profiles()
    
    # Your FULL evidence
    evidence = {
    'techniques_str': ['T1001.002', 'T1003.002', 'T1003.004', 'T1003.006', 'T1005', 'T1016.001', 'T1018', 'T1021.001', 
                      'T1021.002', 'T1021.006', 'T1021.007', 'T1027.001', 'T1027.002', 'T1027.003', 'T1027.006', 
                      'T1036.004', 'T1036.005', 'T1037', 'T1037.004', 'T1047', 'T1048.002', 'T1053.005', 'T1057', 
                      'T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1059.009', 'T1068', 'T1069.002', 
                      'T1070.004', 'T1070.006', 'T1070.008', 'T1071.001', 'T1074.002', 'T1078', 'T1078.002', 
                      'T1078.003', 'T1078.004', 'T1082', 'T1083', 'T1087.002', 'T1087.004', 'T1090.001', 'T1090.002', 
                      'T1090.003', 'T1090.004', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.005', 'T1102.002', 
                      'T1105', 'T1110.001', 'T1110.003', 'T1114.002', 'T1133', 'T1136.003', 'T1140', 'T1190', 
                      'T1195.002', 'T1199', 'T1203', 'T1204.001', 'T1204.002', 'T1213', 'T1213.003', 'T1218.005', 
                      'T1218.011', 'T1482', 'T1484.002', 'T1505.003', 'T1528', 'T1539', 'T1546.003', 'T1546.008', 
                      'T1547.001', 'T1548.002', 'T1550.001', 'T1550.003', 'T1550.004', 'T1552.004', 'T1553.002', 
                      'T1553.005', 'T1555.003', 'T1556.007', 'T1558.003', 'T1560.001', 'T1562.001', 'T1562.002', 
                      'T1562.004', 'T1562.008', 'T1566.001', 'T1566.002', 'T1566.003', 'T1568', 'T1573', 'T1583.001', 
                      'T1583.006', 'T1584.001', 'T1585.001', 'T1586.002', 'T1586.003', 'T1587.001', 'T1587.003', 
                      'T1588.002', 'T1589.001', 'T1595.002', 'T1606.001', 'T1606.002', 'T1621', 'T1649', 'T1651', 
                      'T1665'],
    'software_str': ['S0037', 'S0559', 'S0633', 'S0002', 'S0046', 'S0050', 'S0154', 'S0357', 'S0677', 'S0048', 
                    'S0512', 'S0514', 'S0515', 'S0634', 'S0661', 'S0597', 'S0039', 'S0521', 'S0552', 'S0684', 
                    'S0029', 'S0053', 'S0513', 'S0565', 'S0588', 'S0139', 'S0511', 'S0518', 'S0560', 'S0562', 
                    'S0589', 'S1084', 'S0516', 'S0049', 'S0057', 'S0150', 'S0195', 'S0051', 'S0052', 'S0054', 
                    'S0635', 'S0636', 'S0682', 'S0096', 'S0183', 'S0175', 'S0637', 'S0100'],
    'CVE_ID': ['CVE-2021-44228', 'ewhoiehe']
    }
    
    encoded_evidence = process_evidence(evidence, df_mapping)
    
    print("🔍 LOADING SIAMESE FEATURE ENCODER MODEL...")
    print("✅ Loaded FeatureEncoder model")
    
    # Query!
    results = query_siamese_with_features(model, apt_embeddings, encoded_evidence, apt_ids, profiles, apt_name_mapping)
    
    print_results_table(results, encoded_evidence)
    analyze_feature_matching(results, encoded_evidence, profiles)

if __name__ == "__main__":
    main()

