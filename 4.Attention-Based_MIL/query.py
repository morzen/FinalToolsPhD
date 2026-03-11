import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import islice
import random

# Define the model class here to avoid pickling issues
class OptimizedAttentionMILSimilarity(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature embedding layers
        self.technique_embed = nn.Embedding(10000, hidden_dim)
        self.software_embed = nn.Embedding(10000, hidden_dim)
        self.capec_embed = nn.Embedding(10000, hidden_dim)
        self.cwe_embed = nn.Embedding(10000, hidden_dim)
        self.cve_embed = nn.Embedding(10000, hidden_dim)
        self.pulse_embed = nn.Embedding(10000, hidden_dim)
        
        # Attention networks
        self.attention_nets = nn.ModuleDict({
            'technique': self._build_attention(),
            'software': self._build_attention(),
            'capec': self._build_attention(),
            'cwe': self._build_attention(),
            'cve': self._build_attention(),
            'pulse': self._build_attention()
        })
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 6 * 2 + 6, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 6),
            nn.Sigmoid()
        )
        
        # Initialize weights with higher technique importance
        self.feature_weights = nn.Parameter(torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4, 0.3]))
        self.perfect_match_boost = 2.0  # Direct score multiplier for perfect matches
        
        self.init_weights()

    def _build_attention(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

    def init_weights(self):
        """Initialize weights for embeddings and linear layers"""
        for embed in [self.technique_embed, self.software_embed, self.cve_embed]:
            nn.init.xavier_uniform_(embed.weight)
        
        for net in self.attention_nets.values():
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        for layer in self.similarity_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _calculate_match_ratios(self, apt_dict, evidence_dict):
        ratios = []
        perfect_matches = []
        feature_types = ['technique', 'software', 'capec', 'cwe', 'cve', 'pulse']
        
        for feat_type in feature_types:
            dict_key = feat_type
            apt_features = set(apt_dict.get(dict_key, []))
            evidence_features = set(evidence_dict.get(dict_key, []))
            
            if len(evidence_features) == 0:
                ratios.append(0.01)
                perfect_matches.append(0)
            else:
                coverage = len(apt_features & evidence_features) / len(evidence_features)
                is_perfect = 1 if coverage == 1.0 else 0
                perfect_matches.append(is_perfect)
                
                # Linear coverage with small minimum
                ratios.append(max(coverage, 0.01))
                
        return torch.tensor(ratios), torch.tensor(perfect_matches)

    def forward(self, x1_dict, x2_dict):
        match_ratios, perfect_matches = self._calculate_match_ratios(x1_dict, x2_dict)
        emb1 = self._forward_single(x1_dict, match_ratios)
        emb2 = self._forward_single(x2_dict, match_ratios)
        
        combined = torch.cat([emb1, emb2, match_ratios.unsqueeze(0)], dim=-1)
        feature_similarities = self.similarity_net(combined)
        
        # Use softmax weights
        weights = F.softmax(self.feature_weights, dim=0)

        # Final score - NO PERFECT MATCH BOOST
        similarity = (feature_similarities * weights).sum(dim=-1, keepdim=True)
        return torch.sigmoid(similarity * 10)

    def _forward_single(self, x_dict, match_ratios):
        features = []
        feature_types = ['technique', 'software', 'capec', 'cwe', 'cve', 'pulse']
        
        for i, feat_type in enumerate(feature_types):
            dict_key = feat_type
            feat = x_dict.get(dict_key, [])
            
            if len(feat) > 0:
                embedded = getattr(self, f'{feat_type}_embed')(
                    torch.tensor(feat, dtype=torch.long))
                
                # Enhanced attention with sigmoid gate
                attn_weights = F.softmax(
                    self.attention_nets[feat_type](embedded), dim=0)
                attended = (embedded * attn_weights).sum(dim=0, keepdim=True)
                
                # Weight by both match ratio and log count
                weight = torch.log1p(torch.tensor(len(feat))) * (match_ratios[i] + 0.1)
                features.append(attended * weight)
            else:
                features.append(torch.zeros(1, self.hidden_dim))
                
        return torch.cat(features, dim=-1)

def analyze_feature_matching_classification(encoded_evidence, results, apt_profiles, apt_name_dict):
    """ANALYZE FEATURE MATCHING BY CATEGORY - SEPARATE TABLES FOR TTP, SOFTWARE, CVE"""
    print("\n" + "="*80)
    print(" FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    # Prepare feature matching breakdown
    technique = encoded_evidence.get('technique', [])
    software = encoded_evidence.get('software', [])
    capec = encoded_evidence.get('capec', [])
    cwe = encoded_evidence.get('cwe', [])
    cve = encoded_evidence.get('cve', [])
    pulse = encoded_evidence.get('pulse', [])
    
    # Create a mapping from APT name to profile key
    apt_name_to_key = {}
    for result in results:
        apt_name = result['apt_name']
        # Find the apt profile key that matches this APT name
        for apt_key, profile in apt_profiles.items():
            if apt_name_dict[profile['encoded_id']] == apt_name:
                apt_name_to_key[apt_name] = apt_key
                break
    
    # Calculate feature matching scores for all APTs
    all_feature_scores = []
    
    for result in results:
        apt_name = result['apt_name']
        apt_key = apt_name_to_key.get(apt_name)
        
        if apt_key and apt_key in apt_profiles:
            profile = apt_profiles[apt_key]
            
            # Calculate individual feature matches
            tech_matches = len(set(technique) & set(profile.get('technique', [])))
            soft_matches = len(set(software) & set(profile.get('software', [])))
            capec_matches = len(set(capec) & set(profile.get('capec', [])))
            cwe_matches = len(set(cwe) & set(profile.get('cwe', [])))
            cve_matches = len(set(cve) & set(profile.get('cve', [])))
            pulse_matches = len(set(pulse) & set(profile.get('pulse', [])))
            
            # Calculate normalized scores (0-1)
            # Evidence coverage: how much of the evidence is covered by the APT profile
            tech_evidence_coverage = tech_matches / max(1, len(technique))
            soft_evidence_coverage = soft_matches / max(1, len(software))
            capec_evidence_coverage = capec_matches / max(1, len(capec))
            cwe_evidence_coverage = cwe_matches / max(1, len(cwe))
            cve_evidence_coverage = cve_matches / max(1, len(cve))
            pulse_evidence_coverage = pulse_matches / max(1, len(pulse))
            
            # Profile coverage: how much of the APT profile matches the evidence
            tech_profile_coverage = tech_matches / max(1, len(profile.get('technique', [])))
            soft_profile_coverage = soft_matches / max(1, len(profile.get('software', [])))
            capec_profile_coverage = capec_matches / max(1, len(profile.get('capec', [])))
            cwe_profile_coverage = cwe_matches / max(1, len(profile.get('cwe', [])))
            cve_profile_coverage = cve_matches / max(1, len(profile.get('cve', [])))
            pulse_profile_coverage = pulse_matches / max(1, len(profile.get('pulse', [])))
            
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
                'apt_key': apt_key,
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
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by TTP score
    ttp_sorted = sorted(all_feature_scores, key=lambda x: x['tech_score'], reverse=True)[:5]
    for i, score in enumerate(ttp_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_technique = len(profile.get('technique', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['tech_score']:.4f}    " \
              f"{score['tech_matches']:>2}/{len(technique):<2} ({score['tech_evidence_coverage']:.3f})  " \
              f"{score['tech_matches']:>2}/{total_technique:<3} ({score['tech_profile_coverage']:.3f})  " \
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
    
    # TABLE 3: TOP 5 CAPEC MATCHING
    print("\n TOP 5 APTs BY CAPEC MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CAPEC score
    capec_sorted = sorted(all_feature_scores, key=lambda x: x['capec_score'], reverse=True)[:5]
    for i, score in enumerate(capec_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_capec = len(profile.get('capec', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['capec_score']:.4f}    " \
              f"{score['capec_matches']:>2}/{len(capec):<2} ({score['capec_evidence_coverage']:.3f})  " \
              f"{score['capec_matches']:>2}/{total_capec:<3} ({score['capec_profile_coverage']:.3f})  " \
              f"{score['capec_score']:.4f}"
        print(row)

    # TABLE 4: TOP 5 CWE MATCHING
    print("\n TOP 5 APTs BY CWE MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CWE score
    cwe_sorted = sorted(all_feature_scores, key=lambda x: x['cwe_score'], reverse=True)[:5]
    for i, score in enumerate(cwe_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_cwe = len(profile.get('cwe', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['cwe_score']:.4f}    " \
              f"{score['cwe_matches']:>2}/{len(cwe):<2} ({score['cwe_evidence_coverage']:.3f})  " \
              f"{score['cwe_matches']:>2}/{total_cwe:<3} ({score['cwe_profile_coverage']:.3f})  " \
              f"{score['cwe_score']:.4f}"
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

    # TABLE 6: TOP 5 pulse MATCHING
    print("\n TOP 5 APTs BY pulse MATCHING")
    print("-" * 85)
    header = f"{'rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by pulse score
    pulse_sorted = sorted(all_feature_scores, key=lambda x: x['pulse_score'], reverse=True)[:5]
    for i, score in enumerate(pulse_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_pulse = len(profile.get('pulse', []))
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
        distinctive = "Pulse"
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
        scores = [score['tech_score'], score['soft_score'], score['capec_score'], score['cwe_score'], score['cve_score'], score['pulse_score']]
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
                'pulse_score': score['pulse_score'],
            })
    
    if inconsistent_groups:
        print("POTENTIAL FALSE FLAG INDICATORS:")
        for group in inconsistent_groups[:3]:
            print(f"   • {group['apt_name']}: Variance={group['variance']:.4f}")
            print(f"     TTPs: {group['tech_score']:.4f}, SW: {group['soft_score']:.4f}, CAPEC: {group['capec_score']:.4f}, cwe: {group['cwe_score']:.4f}, cve: {group['cve_score']:.4f}, Pulse: {group['pulse_score']:.4f}")
    else:
        print("No strong false flag indicators detected")
    
    return all_feature_scores

def print_mil_results_table(results, encoded_evidence, apt_profiles, apt_name_dict):
    """Print results table for MIL model"""
    print("\n QUERY EVIDENCE:")
    print(f"   technique: {len(encoded_evidence.get('technique', []))} items")
    print(f"   Software: {len(encoded_evidence.get('software', []))} items")
    print(f"   CAPEC: {len(encoded_evidence.get('capec', []))} items")
    print(f"   cwe: {len(encoded_evidence.get('cwe', []))} items")
    print(f"   cve: {len(encoded_evidence.get('cve', []))} items")
    print(f"   Pulse: {len(encoded_evidence.get('pulse', []))} items")
    print()
    print(f" ATTENTION MIL SIMILARITY RANKINGS")
    print(f" Features: Attention-based multi-instance learning")
    print("=" * 160)
    
    # Header
    header = f"{'rank':<4} {'APT Name':<35} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CAPEC':<8} {'CWE':<8}{'CVE':<8}{'Pulse':<8}{'Matches':<8} {'Profile'}"
    print(header)
    print("-" * 160)
    
    for result in results:  # Show ALL results
        # Truncate long APT names
        apt_name = result['apt_name']
        if len(apt_name) > 34:
            apt_name = apt_name[:31] + "..."
        
        # Get profile information
        total_matches = result['technique'] + result['software'] + result['capec'] + result['cwe'] + result['cve'] + result['pulse']
        
        # Find profile for this APT
        profile_info = ""
        for apt_key, profile in apt_profiles.items():
            if apt_name_dict[profile['encoded_id']] == result['apt_name']:
                tech_count = len(profile.get('technique', []))
                soft_count = len(profile.get('software', []))
                capec_count = len(profile.get('capec', []))
                cwe_count = len(profile.get('cwe', []))
                cve_count = len(profile.get('cve', []))
                pulse_count = len(profile.get('pulse', []))
                profile_info = f"{tech_count}T/{soft_count}S/{capec_count}Ca/{cwe_count}Cw/{cve_count}Cv/{pulse_count}P"
                break
        
        row = (f"{result['rank']:<4} {apt_name:<35} {result['Score']:.4f}    "
               f"{result['technique']}/{len(encoded_evidence.get('technique', [])):<7} "
               f"{result['software']}/{len(encoded_evidence.get('software', [])):<7} "
               f"{result['capec']}/{len(encoded_evidence.get('capec', [])):<7} "
               f"{result['cwe']}/{len(encoded_evidence.get('cwe', [])):<7} "
               f"{result['cve']}/{len(encoded_evidence.get('cve', [])):<7} "
               f"{result['pulse']}/{len(encoded_evidence.get('pulse', [])):<7} "
               f"{total_matches:<8} {profile_info}")
        print(row)
    
    print("-" * 160)

# Load the complete model and data from a single file
def load_complete_model(status):
    import pickle
    import torch
    
    if status == "hybrid":
        # Try the fixed version first, then original
        fixed_path = '../4.Attention-Based_MIL/6feat_apt_similarity_model_fixed.pth'
        original_path = '../4.Attention-Based_MIL/6feat_apt_similarity_model_complete.pth'
        
        # Check which file exists
        import os
        if os.path.exists(fixed_path):
            model_path = fixed_path
            print(f"Loading model from: {model_path} (FIXED VERSION)")
        else:
            model_path = original_path
            print(f"Loading model from: {model_path} (ORIGINAL)")
    else:
        model_path = '6feat_apt_similarity_model_complete.pth'
        print(f"Loading model from: {model_path}")
    
    # SIMPLE LOADING - NO BULLSHIT
    try:
        # Method 1: Try pickle first
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        print("✓ Loaded with pickle")
    except Exception as e:
        print(f"Pickle failed: {e}")
        
        # Method 2: Try torch.load as fallback
        try:
            save_data = torch.load(model_path, weights_only=False, map_location='cpu')
            print("✓ Loaded with torch.load")
        except Exception as e2:
            print(f"Torch.load failed: {e2}")
            raise
    
    # Initialize the model
    model = OptimizedAttentionMILSimilarity()
    
    # Check what we loaded
    if isinstance(save_data, dict) and 'model_state_dict' in save_data:
        # It's a dict with state_dict
        model.load_state_dict(save_data['model_state_dict'])
        apt_profiles = save_data['apt_profiles']
        apt_name_dict = save_data['apt_name_dict']
        df_mapping = save_data['df_mapping']
        apt_ids = save_data['apt_ids']
    elif isinstance(save_data, OptimizedAttentionMILSimilarity):
        # It's already a model object
        model = save_data
        # Extract other data from somewhere else or use defaults
        apt_profiles = {}
        apt_name_dict = {}
        df_mapping = pd.DataFrame()
        apt_ids = []
    else:
        # Try to extract from the object
        model.load_state_dict(save_data)
        apt_profiles = {}
        apt_name_dict = {}
        df_mapping = pd.DataFrame()
        apt_ids = []
    
    model.eval()
    
    return model, apt_profiles, apt_name_dict, df_mapping, apt_ids

# Evidence processing functions
def sanitize_evidence(evidence, df_mapping):
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques': ('techniques_Origin', 'techniques_Encoded'),
        'software': ('software_Origin', 'software_Encoded'),
        'capec': ('CAPEC_ID_Origin', 'CAPEC_ID_Encoded'),
        'cwe': ('CWE_ID_Origin', 'CWE_ID_Encoded'),
        'cve': ('CVE_ID_Origin', 'CVE_ID_Encoded'),
        'pulse': ('Pulse_ID_Origin', 'Pulse_ID_Encoded')
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
    encoded_evidence = {}
    
    column_mapping = {
        'techniques': ('techniques_Encoded', 'technique'),
        'software': ('software_Encoded', 'software'),
        'capec': ('CAPEC_ID_Encoded', 'capec'),
        'cwe': ('CWE_ID_Encoded', 'cwe'),
        'cve': ('CVE_ID_Encoded', 'cve'),
        'pulse': ('Pulse_ID_Encoded', 'pulse'),
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

def calculate_similarities(model, apt_profiles, evidence_features, apt_ids):
    """Calculate similarity scores for all APTs using the Attention MIL model"""
    similarities = {}
    with torch.no_grad():
        for apt_name in apt_ids:
            apt_features = {
                'technique': apt_profiles[apt_name]['technique'],
                'software': apt_profiles[apt_name]['software'],
                'capec': apt_profiles[apt_name]['capec'],
                'cwe': apt_profiles[apt_name]['cwe'],
                'cve': apt_profiles[apt_name]['cve'],
                'pulse': apt_profiles[apt_name]['pulse']
            }
            
            similarity = model(apt_features, evidence_features)
            similarities[apt_name] = similarity.item()
    
    return similarities

def process_custom_evidence(evidence, model, apt_profiles, apt_name_dict, df_mapping, apt_ids):
    """Process custom evidence and return ranked results"""
    print("Processing evidence...")
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
    
    print("Evidence processed:")
    print({k: len(v) for k, v in encoded_evidence.items()})
    
    # Prepare evidence features
    evidence_features = {
        'technique': encoded_evidence.get('techniques', []),
        'software': encoded_evidence.get('software', []),
        'capec': encoded_evidence.get('capec', []),
        'cwe': encoded_evidence.get('cwe', []),
        'cve': encoded_evidence.get('cve', []),
        'pulse': encoded_evidence.get('pulse', [])
    }
    
    # Calculate similarities
    print("Calculating similarities...")
    similarities = calculate_similarities(model, apt_profiles, evidence_features, apt_ids)
    
    # Create results
    full_results = []
    sorted_apts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (apt_name, score) in enumerate(sorted_apts, 1):
        apt_data = apt_profiles[apt_name]
        
        common_features = {
            'technique': len(set(encoded_evidence.get('technique', [])) & set(apt_data['technique'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'capec': len(set(encoded_evidence.get('capec', [])) & set(apt_data['capec'])),
            'cwe': len(set(encoded_evidence.get('cwe', [])) & set(apt_data['cwe'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve'])),
            'pulse': len(set(encoded_evidence.get('pulse', [])) & set(apt_data['pulse']))
        }
        
        full_results.append({
            'rank': rank,
            'apt_name': apt_name_dict[apt_data['encoded_id']],
            'Score': score,
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })
    
    return full_results, encoded_evidence

def hybrid_query(evidence):
    model, apt_profiles, apt_name_dict, df_mapping, apt_ids = load_complete_model("hybrid")
    results, encoded_evidence = process_custom_evidence(evidence, model, apt_profiles, apt_name_dict, df_mapping, apt_ids)
    
    return results

def main():
    # Load everything
    model, apt_profiles, apt_name_dict, df_mapping, apt_ids = load_complete_model("nonhybrid")
    
    print("Model loaded successfully!")
    print(f"Number of APTs: {len(apt_ids)}")
    
    # Define your evidence here (APT29 example - replace with your own evidence)
    # evidence = {
    #      'techniques': ['T1001.002', 'T1003.002'],
    #      'software': ['S0002', 'S0029', 'S0037'],
    #      'capec': ['CAPEC-127', 'CAPEC-150'],
    #      'cwe': ['CWE-1021', 'CWE-113'],
    #      'cve': ['CVE-2012-0433', 'CVE-2013-10007'],
    #      'pulse': ['5c88e255d5fbbe1cb7c54534', '5cb741fb29d3ac01c1ac01ea']
    #  }
    
    import json

    # Read evidence from JSON file
    with open('../query_input.json', 'r') as f:
        evidence = json.load(f)
    
    # Process custom evidence and get results
    results, encoded_evidence = process_custom_evidence(evidence, model, apt_profiles, apt_name_dict, df_mapping, apt_ids)
    
    # PRINT MIL RESULTS TABLE (ALL 181 APTs)
    print_mil_results_table(results, encoded_evidence, apt_profiles, apt_name_dict)
    
    # PERFORM FEATURE MATCHING CLASSIFICATION ANALYSIS
    feature_scores = analyze_feature_matching_classification(
        encoded_evidence, 
        results, 
        apt_profiles, 
        apt_name_dict
    )
    
    print(f"\n ATTENTION MIL QUERY COMPLETE!")
    print(f" Most similar: {results[0]['apt_name']} (score: {results[0]['Score']:.4f})")
    print(f" Least similar: {results[-1]['apt_name']} (score: {results[-1]['Score']:.4f})")
    print(f" Total APTs ranked: {len(results)}")
    print(f" Using Attention MIL with perfect match boosting")

if __name__ == "__main__":
    main()
