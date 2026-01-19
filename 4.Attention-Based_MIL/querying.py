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
        self.cve_embed = nn.Embedding(10000, hidden_dim)
        
        # Attention networks
        self.attention_nets = nn.ModuleDict({
            'technique': self._build_attention(),
            'software': self._build_attention(),
            'cve': self._build_attention()
        })
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2 + 3, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 3),
            nn.Sigmoid()
        )
        
        # Initialize weights with higher technique importance
        self.feature_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.1]))
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
        feature_types = ['technique', 'software', 'cve']
        
        for feat_type in feature_types:
            dict_key = feat_type + 's' if feat_type != 'software' else feat_type
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
        
        # Apply perfect match boost directly to scores
        boosted_similarities = feature_similarities * (1 + perfect_matches * (self.perfect_match_boost - 1))
        
        # Use softmax weights
        weights = F.softmax(self.feature_weights, dim=0)
        
        # Final score
        similarity = (boosted_similarities * weights).sum(dim=-1, keepdim=True)
        return torch.sigmoid(similarity * 10)

    def _forward_single(self, x_dict, match_ratios):
        features = []
        feature_types = ['technique', 'software', 'cve']
        
        for i, feat_type in enumerate(feature_types):
            dict_key = feat_type + 's' if feat_type != 'software' else feat_type
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
    print("🔍 FEATURE MATCHING CLASSIFICATION - TOP 5 BY CATEGORY")
    print("="*80)
    
    # Prepare feature matching breakdown
    techniques = encoded_evidence.get('techniques', [])
    software = encoded_evidence.get('software', [])
    cves = encoded_evidence.get('cve', [])
    
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
            cve_matches = len(set(cves) & set(profile.get('cve', [])))
            
            # Calculate normalized scores (0-1)
            # Evidence coverage: how much of the evidence is covered by the APT profile
            tech_evidence_coverage = tech_matches / max(1, len(techniques))
            soft_evidence_coverage = soft_matches / max(1, len(software))
            cve_evidence_coverage = cve_matches / max(1, len(cves))
            
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
    print("\n🎯 TOP 5 APTs BY TTP MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
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
    print("\n🛠️  TOP 5 APTs BY SOFTWARE MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
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
    
    # TABLE 3: TOP 5 CVE MATCHING
    print("\n💥 TOP 5 APTs BY CVE MATCHING")
    print("-" * 85)
    header = f"{'Rank':<4} {'APT Name':<30} {'Score':<10} {'Evidence':<15} {'Profile':<15} {'Harmonic':<10}"
    print(header)
    print("-" * 85)
    
    # Sort by CVE score
    cve_sorted = sorted(all_feature_scores, key=lambda x: x['cve_score'], reverse=True)[:5]
    for i, score in enumerate(cve_sorted, 1):
        apt_key = score['apt_key']
        profile = apt_profiles[apt_key]
        total_cves = len(profile.get('cve', []))
        row = f"{i:<4} {score['apt_name']:<30} {score['cve_score']:.4f}    " \
              f"{score['cve_matches']:>2}/{len(cves):<2} ({score['cve_evidence_coverage']:.3f})  " \
              f"{score['cve_matches']:>2}/{total_cves:<3} ({score['cve_profile_coverage']:.3f})  " \
              f"{score['cve_score']:.4f}"
        print(row)
    
    # SUMMARY STATISTICS
    print("\n📊 FEATURE MATCHING SUMMARY")
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
        distinctive = "TTPs (Behavioral patterns)"
    elif max_avg == avg_soft:
        distinctive = "Software (Tools and malware)"
    else:
        distinctive = "CVEs (Vulnerability exploitation)"
    
    print(f"Most distinctive feature: {distinctive}")
    
    # FALSE FLAG DETECTION
    print("\n🔎 FALSE FLAG DETECTION ANALYSIS")
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
        print("⚠️  POTENTIAL FALSE FLAG INDICATORS:")
        for group in inconsistent_groups[:3]:
            print(f"   • {group['apt_name']}: Variance={group['variance']:.4f}")
            print(f"     TTPs: {group['tech_score']:.4f}, SW: {group['soft_score']:.4f}, CVEs: {group['cve_score']:.4f}")
    else:
        print("✅ No strong false flag indicators detected")
    
    return all_feature_scores

def print_mil_results_table(results, encoded_evidence, apt_profiles, apt_name_dict):
    """Print results table for MIL model"""
    print("\n📋 QUERY EVIDENCE:")
    print(f"   Techniques: {len(encoded_evidence.get('techniques', []))} items")
    print(f"   Software: {len(encoded_evidence.get('software', []))} items")
    print(f"   CVEs: {len(encoded_evidence.get('cve', []))} items")
    print()
    print(f"🎯 ATTENTION MIL SIMILARITY RANKINGS")
    print(f"🏆 Features: Attention-based multi-instance learning")
    print("=" * 120)
    
    # Header
    header = f"{'Rank':<4} {'APT Name':<35} {'Similarity':<10} {'Tech':<8} {'Soft':<8} {'CVE':<8} {'Matches':<8} {'Profile'}"
    print(header)
    print("-" * 120)
    
    for result in results:  # Show ALL results
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
                profile_info = f"{tech_count}T/{soft_count}S/{cve_count}C"
                break
        
        row = (f"{result['Rank']:<4} {apt_name:<35} {result['Score']:.4f}    "
               f"{result['techniques']}/{len(encoded_evidence.get('techniques', [])):<7} "
               f"{result['software']}/{len(encoded_evidence.get('software', [])):<7} "
               f"{result['cve']}/{len(encoded_evidence.get('cve', [])):<7} "
               f"{total_matches:<8} {profile_info}")
        print(row)
    
    print("-" * 120)

# Load the complete model and data from a single file
def load_complete_model():
    # Add safe globals for PyTorch 2.6+ compatibility
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    torch.serialization.add_safe_globals([np.dtype])
    
    try:
        # First try with weights_only=True (safer)
        save_data = torch.load('apt_similarity_model_complete.pth', weights_only=True)
    except Exception as e:
        print(f"Error with weights_only=True: {e}")
        print("Trying with weights_only=False...")
        # If that fails, try with weights_only=False (less secure but works)
        save_data = torch.load('apt_similarity_model_complete.pth', weights_only=False)
    
    # Initialize the model
    model = OptimizedAttentionMILSimilarity()
    model.load_state_dict(save_data['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Extract other data
    apt_profiles = save_data['apt_profiles']
    apt_name_dict = save_data['apt_name_dict']
    df_mapping = save_data['df_mapping']
    apt_ids = save_data['apt_ids']
    
    return model, apt_profiles, apt_name_dict, df_mapping, apt_ids

# Evidence processing functions
def sanitize_evidence(evidence, df_mapping):
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Origin', 'techniques_Encoded'),
        'software_str': ('software_Origin', 'software_Encoded'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded')
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
        'techniques_str': ('techniques_Encoded', 'techniques'),
        'software_str': ('software_Encoded', 'software'),
        'CVE_ID': ('CVE_ID_Encoded', 'cve')
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
                'techniques': apt_profiles[apt_name]['techniques'],
                'software': apt_profiles[apt_name]['software'],
                'cve': apt_profiles[apt_name]['cve']
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
        'techniques': encoded_evidence.get('techniques', []),
        'software': encoded_evidence.get('software', []),
        'cve': encoded_evidence.get('cve', [])
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
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve']))
        }
        
        full_results.append({
            'Rank': rank,
            'APT': apt_name_dict[apt_data['encoded_id']],
            'Score': score,
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })
    
    return full_results, encoded_evidence

def main():
    # Load everything
    model, apt_profiles, apt_name_dict, df_mapping, apt_ids = load_complete_model()
    
    print("Model loaded successfully!")
    print(f"Number of APTs: {len(apt_ids)}")
    
    # Define your evidence here (APT29 example - replace with your own evidence)
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
        'CVE_ID': ['CVE-2021-44228', 'patrick', 'CVE-2012-0433', 'CVE-2013-10007', 'CVE-2016-2124', 'CVE-2016-8612',]
    }
    
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
    
    print(f"\n✅ ATTENTION MIL QUERY COMPLETE!")
    print(f"🏆 Most similar: {results[0]['APT']} (score: {results[0]['Score']:.4f})")
    print(f"💩 Least similar: {results[-1]['APT']} (score: {results[-1]['Score']:.4f})")
    print(f"📊 Total APTs ranked: {len(results)}")
    print(f"📊 Using Attention MIL with perfect match boosting")

if __name__ == "__main__":
    main()