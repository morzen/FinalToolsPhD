import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict

class APTQueryModel(nn.Module):
    def __init__(self, num_techniques, num_software, num_cve, hidden_dim=128):
        super().__init__()
        self.technique_embedding = nn.Embedding(num_techniques, hidden_dim, padding_idx=0)
        self.software_embedding = nn.Embedding(num_software, hidden_dim, padding_idx=0)
        self.cve_embedding = nn.Embedding(num_cve, hidden_dim, padding_idx=0)
        self.feature_combiner = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, techniques, software, cves):
        tech_emb = self.technique_embedding(techniques).mean(dim=1)
        soft_emb = self.software_embedding(software).mean(dim=1)
        cve_emb = self.cve_embedding(cves).mean(dim=1)
        combined = torch.cat([tech_emb, soft_emb, cve_emb], dim=1)
        return self.feature_combiner(combined)

def confidence_scoring(similarities, top_k=5):
    if len(similarities) < top_k:
        top_k = len(similarities)
    top_similarities = similarities.topk(top_k).values
    if top_similarities.std() > 0:
        confidence = (top_similarities[0] - top_similarities[1:].mean()) / top_similarities.std()
    else:
        confidence = torch.tensor(0.0)
    return torch.sigmoid(confidence).item()

def normalized_similarity(query_embedding, apt_embeddings):
    query_norm = F.normalize(query_embedding, p=2, dim=1)
    apt_norm = F.normalize(apt_embeddings, p=2, dim=1)
    return torch.matmul(query_norm, apt_norm.t()).squeeze(0)

def convert_evidence_to_encoded(evidence, mapping_file="../BaseData/comparison_Encoded_df.csv"):
    print("🔍 LOADING MAPPING DATA...")
    df_mapping = pd.read_csv(mapping_file)
    
    technique_mapping = {}
    software_mapping = {} 
    cve_mapping = {}
    
    for _, row in df_mapping.iterrows():
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            technique_mapping[row['techniques_Origin']] = int(row['techniques_Encoded'])
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            software_mapping[row['software_Origin']] = int(row['software_Encoded'])
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            cve_mapping[row['CVE_ID_Origin']] = int(row['CVE_ID_Encoded'])
    
    print(f"✅ Mappings loaded: {len(technique_mapping)} techniques, {len(software_mapping)} software, {len(cve_mapping)} CVEs")
    
    encoded_evidence = {'techniques': [], 'software': [], 'cves': []}
    
    for technique in evidence.get('techniques', []):
        if technique in technique_mapping:
            encoded_evidence['techniques'].append(technique_mapping[technique])
        else:
            print(f"⚠️  Warning: Technique '{technique}' not found in mapping")
    
    for software in evidence.get('software', []):
        if software in software_mapping:
            encoded_evidence['software'].append(software_mapping[software])
        else:
            print(f"⚠️  Warning: Software '{software}' not found in mapping")
    
    for cve in evidence.get('cves', []):
        if cve in cve_mapping:
            encoded_evidence['cves'].append(cve_mapping[cve])
        else:
            print(f"⚠️  Warning: CVE '{cve}' not found in mapping")
    
    print(f"\n📊 CONVERSION RESULTS:")
    print(f"   Input techniques: {len(evidence.get('techniques', []))} → Encoded: {len(encoded_evidence['techniques'])}")
    print(f"   Input software: {len(evidence.get('software', []))} → Encoded: {len(encoded_evidence['software'])}")
    print(f"   Input CVEs: {len(evidence.get('cves', []))} → Encoded: {len(encoded_evidence['cves'])}")
    
    return encoded_evidence

def safe_model_config(config):
    compatible_keys = ['num_techniques', 'num_software', 'num_cve', 'hidden_dim']
    clean_config = {}
    for k in compatible_keys:
        v = config.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if k == 'num_techniques': v = 548
            elif k == 'num_software': v = 541
            elif k == 'num_cve': v = 7938
            elif k == 'hidden_dim': v = 128
        clean_config[k] = v
    print(f"📏 Using model config: {clean_config}")
    return clean_config

def harmonic_mean(a, b):
    if a + b == 0: return 0.0
    return 2 * a * b / (a + b)

def compute_feature_scores(evidence, profiles, apt_name_mapping):
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    cves = evidence.get('cves', [])
    
    feature_scores = []
    for apt_id, profile in profiles.items():
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        
        tech_profile = profile.get('techniques', [])
        soft_profile = profile.get('software', [])
        cve_profile = profile.get('cve', [])
        
        tech_matches = len(set(techniques) & set(tech_profile))
        soft_matches = len(set(software) & set(soft_profile))
        cve_matches = len(set(cves) & set(cve_profile))
        
        tech_evid_cov = tech_matches / max(1, len(techniques))
        soft_evid_cov = soft_matches / max(1, len(software))
        cve_evid_cov = cve_matches / max(1, len(cves))
        
        tech_prof_cov = tech_matches / max(1, len(tech_profile))
        soft_prof_cov = soft_matches / max(1, len(soft_profile))
        cve_prof_cov = cve_matches / max(1, len(cve_profile))
        
        feature_scores.append({
            'apt_id': apt_id,
            'apt_name': apt_name,
            'tech_score': harmonic_mean(tech_evid_cov, tech_prof_cov),
            'soft_score': harmonic_mean(soft_evid_cov, soft_prof_cov),
            'cve_score': harmonic_mean(cve_evid_cov, cve_prof_cov),
            'tech_matches': tech_matches,
            'soft_matches': soft_matches,
            'cve_matches': cve_matches,
            'tech_evid_cov': tech_evid_cov,
            'soft_evid_cov': soft_evid_cov,
            'cve_evid_cov': cve_evid_cov,
            'tech_prof_cov': tech_prof_cov,
            'soft_prof_cov': soft_prof_cov,
            'cve_prof_cov': cve_prof_cov
        })
    return feature_scores

def print_top5_by_category(feature_scores, evidence):
    print("\n" + "="*80)
    print("🔍 FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    # Sort by each category
    ttp_sorted = sorted(feature_scores, key=lambda x: x['tech_score'], reverse=True)[:5]
    soft_sorted = sorted(feature_scores, key=lambda x: x['soft_score'], reverse=True)[:5]
    cve_sorted = sorted(feature_scores, key=lambda x: x['cve_score'], reverse=True)[:5]
    
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    cves = evidence.get('cves', [])
    
    print("\n🎯 TOP 5 APTs BY TTP MATCHING")
    print("-" * 85)
    print(f"{'Rank':<4} {'APT Name':<28} {'Score':<10} {'Evidence':<16} {'Profile':<12} {'Harmonic':<8}")
    print("-" * 85)
    for i, score in enumerate(ttp_sorted, 1):
        evid = f"{score['tech_matches']}/{len(techniques)} ({score['tech_evid_cov']:.3f})"
        prof = f"{score['tech_matches']}/{score['tech_matches']+1} ({score['tech_prof_cov']:.3f})"
        print(f"{i:<4} {score['apt_name']:<28} {score['tech_score']:<10.4f} {evid:<16} {prof:<12} {score['tech_score']:<8.4f}")
    
    print("\n🛠️  TOP 5 APTs BY SOFTWARE MATCHING")
    print("-" * 85)
    print(f"{'Rank':<4} {'APT Name':<28} {'Score':<10} {'Evidence':<16} {'Profile':<12} {'Harmonic':<8}")
    print("-" * 85)
    for i, score in enumerate(soft_sorted, 1):
        evid = f"{score['soft_matches']}/{len(software)} ({score['soft_evid_cov']:.3f})"
        prof = f"{score['soft_matches']}/{score['soft_matches']+1} ({score['soft_prof_cov']:.3f})"
        print(f"{i:<4} {score['apt_name']:<28} {score['soft_score']:<10.4f} {evid:<16} {prof:<12} {score['soft_score']:<8.4f}")
    
    print("\n💥 TOP 5 APTs BY CVE MATCHING")
    print("-" * 85)
    print(f"{'Rank':<4} {'APT Name':<28} {'Score':<10} {'Evidence':<16} {'Profile':<12} {'Harmonic':<8}")
    print("-" * 85)
    for i, score in enumerate(cve_sorted, 1):
        evid = f"{score['cve_matches']}/{len(cves)} ({score['cve_evid_cov']:.3f})"
        prof = f"{score['cve_matches']}/{score['cve_matches']+1} ({score['cve_prof_cov']:.3f})"
        print(f"{i:<4} {score['apt_name']:<28} {score['cve_score']:<10.4f} {evid:<16} {prof:<12} {score['cve_score']:<8.4f}")

def print_feature_summary_and_false_flags(feature_scores):
    ttp_scores = [s['tech_score'] for s in feature_scores]
    soft_scores = [s['soft_score'] for s in feature_scores]
    cve_scores = [s['cve_score'] for s in feature_scores]
    
    print("\n📊 FEATURE MATCHING SUMMARY")
    print("-" * 50)
    print(f"Average TTP matching score: {np.mean(ttp_scores):.4f}")
    print(f"Average Software matching score: {np.mean(soft_scores):.4f}")
    print(f"Average CVE matching score: {np.mean(cve_scores):.4f}")
    print("Most distinctive feature: TTPs (Behavioral patterns)")
    
    print("\n🔎 FALSE FLAG DETECTION ANALYSIS")
    print("-" * 50)
    print("⚠️  POTENTIAL FALSE FLAG INDICATORS:")
    for score in feature_scores[:5]:  # Check top candidates
        variance = np.std([score['tech_score'], score['soft_score'], score['cve_score']])
        if variance > 0.2:
            print(f"   • {score['apt_name']}: Variance={variance:.4f}")
            print(f"     TTPs: {score['tech_score']:.4f}, SW: {score['soft_score']:.4f}, CVEs: {score['cve_score']:.4f}")

def query_apt_similarity_enhanced(evidence):
    print("🔍 LOADING ENHANCED TRAINED MODEL...")
    model_file = "trained_apt_gnn_model.pkl"
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found!")
        return [], 0.0, {}, {}
    
    try:
        with open(model_file, 'rb') as f:
            saved_data = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return [], 0.0, {}, {}
    
    model_config = safe_model_config(saved_data.get("model_config", {}))
    model = APTQueryModel(**model_config)

    if "model_state_dict" in saved_data:
        state_dict = saved_data["model_state_dict"]
        filtered_state = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state, strict=False)
        print("✅ Model state loaded (extra keys ignored)")
    model.eval()
    
    feature_mappings = saved_data.get("feature_mappings")
    if feature_mappings:
        technique_to_idx, software_to_idx, cve_to_idx = feature_mappings
    else:
        technique_to_idx = software_to_idx = cve_to_idx = {}
    
    apt_embeddings = saved_data.get("apt_embeddings")
    apt_ids = saved_data.get("apt_ids", [])
    profiles = saved_data.get("profiles", {})
    apt_name_mapping = saved_data.get("apt_name_mapping", {})
    
    feature_weights = saved_data.get("feature_weights", {'techniques': 10.0, 'software': 3.0, 'cve': 0.1})
    
    techniques = evidence.get("techniques", [])
    software = evidence.get("software", [])
    cves = evidence.get("cves", [])
    
    tech_indices = [technique_to_idx.get(t, 0) for t in techniques] if techniques else [0]
    soft_indices = [software_to_idx.get(s, 0) for s in software] if software else [0]
    cve_indices = [cve_to_idx.get(c, 0) for c in cves] if cves else [0]
    
    tech_tensor = torch.tensor(tech_indices, dtype=torch.long)
    soft_tensor = torch.tensor(soft_indices, dtype=torch.long)
    cve_tensor = torch.tensor(cve_indices, dtype=torch.long)
    
    with torch.no_grad():
        query_embedding = model(tech_tensor.unsqueeze(0), soft_tensor.unsqueeze(0), cve_tensor.unsqueeze(0))
    
    similarities = normalized_similarity(query_embedding, apt_embeddings)
    confidence = confidence_scoring(similarities, top_k=5)
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    
    results = []
    for i, (sim, idx) in enumerate(zip(sorted_similarities, sorted_indices)):
        apt_id = apt_ids[idx.item()]
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        profile = profiles.get(apt_id, {"techniques": [], "software": [], "cve": []})
        
        tech_matches = len(set(techniques) & set(profile.get("techniques", [])))
        soft_matches = len(set(software) & set(profile.get("software", [])))
        cve_matches = len(set(cves) & set(profile.get("cve", [])))
        
        weighted_match = (tech_matches * feature_weights["techniques"] +
                         soft_matches * feature_weights["software"] +
                         cve_matches * feature_weights["cve"])
        
        results.append({
            "rank": i + 1,
            "apt_name": apt_name,
            "similarity": sim.item(),
            "tech_match": f"{tech_matches}/{len(techniques)}",
            "soft_match": f"{soft_matches}/{len(software)}",
            "cve_match": f"{cve_matches}/{len(cves)}",
            "weighted_match": weighted_match,
            "profile_size": f"{len(profile.get('techniques', []))}T/{len(profile.get('software', []))}S/{len(profile.get('cve', []))}C",
            "apt_id": apt_id
        })
    
    feature_scores = compute_feature_scores(evidence, profiles, apt_name_mapping)
    return results, confidence, profiles, apt_name_mapping, feature_scores

def print_enhanced_results_table(results, evidence, confidence):
    print("\n" + "="*80)
    print("🔍 FEATURE MATCHING CLASSIFICATION - ALL APTs")
    print("="*80)
    
    print("\n📋 QUERY EVIDENCE:")
    print(f"   Techniques: {len(evidence['techniques'])} - {evidence['techniques']}")
    print(f"   Software: {len(evidence['software'])} - {evidence['software']}")
    print(f"   CVEs: {len(evidence['cves'])} - {evidence['cves']}")
    
    print(f"\n🎯 ENHANCED GNN COSINE SIMILARITY RANKINGS (Confidence: {confidence:.2%})")
    print("=" * 120)
    header = f"{'Rank':<4} {'APT Name':<28} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CVE':<8} {'Weighted':<10} {'Profile'}"
    print(header)
    print("-" * 120)
    
    for result in results:
        apt_name = result['apt_name'][:25] + "..." if len(result['apt_name']) > 25 else result['apt_name']
        row = (f"{result['rank']:<4} {apt_name:<28} {result['similarity']:<10.4f} "
               f"{result['tech_match']:<8} {result['soft_match']:<8} "
               f"{result['cve_match']:<8} {result['weighted_match']:<10.1f} {result['profile_size']}")
        print(row)
    print("-" * 120)

if __name__ == "__main__":
    evidence_humanreadable = {
        "techniques": ["T1001.002", "T1003.002"],
        "software": ["S0002", "S0029"],
        "cves": ["CVE-2021-44228"]
    }
    
    evidence = convert_evidence_to_encoded(evidence_humanreadable)
    
    print("🚀 RUNNING ENHANCED GNN APT SIMILARITY QUERY")
    print("=" * 70)
    
    results, confidence, profiles, apt_name_mapping, feature_scores = query_apt_similarity_enhanced(evidence)
    
    print_enhanced_results_table(results, evidence, confidence)
    print_top5_by_category(feature_scores, evidence)
    print_feature_summary_and_false_flags(feature_scores)
    
    print("\n✅ SIAMESE NETWORK QUERY COMPLETE!")
    if results:
        best = results[0]
        worst = results[-1]
        print(f"🏆 Most similar: {best['apt_name']} (score: {best['similarity']:.4f})")
        print(f"💩 Least similar: {worst['apt_name']} (score: {worst['similarity']:.4f})")
    print("📊 Using GNN with weighted feature overlap")