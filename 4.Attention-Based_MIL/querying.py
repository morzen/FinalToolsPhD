import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
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
            nn.Linear(hidden_dim * 6 * 2 + 6, hidden_dim*4),  # +6 for match ratios
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 6),
            nn.Sigmoid()
        )
        
        # Initialize weights with higher technique importance
        self.feature_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.15, 0.1, 0.025, 0.025]))
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
        for embed in [self.technique_embed, self.software_embed, self.capec_embed, 
                     self.cwe_embed, self.cve_embed, self.pulse_embed]:
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
        feature_types = ['technique', 'software', 'capec', 'cwe', 'cve', 'pulse']
        
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
        'CAPEC_ID': ('CAPEC_ID_Origin', 'CAPEC_ID_Encoded'),
        'CWE_ID': ('CWE_ID_Origin', 'CWE_ID_Encoded'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded'),
        'Pulse_ID': ('Pulse_ID_Origin', 'Pulse_ID_Encoded')
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
        'CAPEC_ID': ('CAPEC_ID_Encoded', 'capec'),
        'CWE_ID': ('CWE_ID_Encoded', 'cwe'),
        'CVE_ID': ('CVE_ID_Encoded', 'cve'),
        'Pulse_ID': ('Pulse_ID_Encoded', 'pulse')
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
                'capec': apt_profiles[apt_name]['capec'],
                'cwe': apt_profiles[apt_name]['cwe'],
                'cve': apt_profiles[apt_name]['cve'],
                'pulse': apt_profiles[apt_name]['pulse']
            }
            
            similarity = model(apt_features, evidence_features)
            similarities[apt_name] = similarity.item()
    
    return similarities

def generate_cross_similarity_matrix(model, apt_profiles, apt_ids):
    """Generate cross-similarity matrix between all APT pairs"""
    num_apts = len(apt_ids)
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    print("Computing cross-similarity matrix for all APT pairs...")
    
    with torch.no_grad():
        for i in range(num_apts):
            apt1_name = apt_ids[i]
            apt1_features = {
                'techniques': apt_profiles[apt1_name]['techniques'],
                'software': apt_profiles[apt1_name]['software'],
                'capec': apt_profiles[apt1_name]['capec'],
                'cwe': apt_profiles[apt1_name]['cwe'],
                'cve': apt_profiles[apt1_name]['cve'],
                'pulse': apt_profiles[apt1_name]['pulse']
            }
            
            for j in range(i, num_apts):  # Only compute upper triangle to save computation
                apt2_name = apt_ids[j]
                apt2_features = {
                    'techniques': apt_profiles[apt2_name]['techniques'],
                    'software': apt_profiles[apt2_name]['software'],
                    'capec': apt_profiles[apt2_name]['capec'],
                    'cwe': apt_profiles[apt2_name]['cwe'],
                    'cve': apt_profiles[apt2_name]['cve'],
                    'pulse': apt_profiles[apt2_name]['pulse']
                }
                
                similarity = model(apt1_features, apt2_features)
                similarity_matrix[i, j] = similarity.item()
                similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetric matrix
    
    print("Cross-similarity matrix computation completed!")
    return similarity_matrix

def create_similarity_matrix(similarity_matrix, apt_name_dict, apt_profiles, apt_ids):
    """Create similarity matrix DataFrame with APT names"""
    # Get APT names in order
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=apt_names,
        columns=apt_names
    )
    
    return similarity_df

def plot_similarity_heatmap(similarity_df):
    """Plot heatmap of similarity matrix with complete square and red-green color scheme"""
    plt.figure(figsize=(20, 18))
    
    # Create custom red-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    
    # Plot complete heatmap (no mask)
    sns.heatmap(similarity_df, 
                mask=None,  # Remove the triangular mask
                cmap=cmap,
                center=0.5,  # Center the colormap at 0.5
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Similarity Score'})
    
    plt.title("APT Group Cross-Similarity Matrix (Attention MIL Model)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.show()

def evaluate_similarity_matrix(similarity_df, top_x_values):
    """Evaluate the similarity matrix for top X matches"""
    apt_names = similarity_df.index.tolist()
    similarity_matrix = similarity_df.values
    
    print("\nEvaluating similarity matrix for top X matches:")
    print("="*50)
    
    for topX in top_x_values:
        success = 0
        failure = 0
        
        for i, target_apt in enumerate(apt_names):
            # Get similarities for this APT (excluding self-similarity)
            similarities = similarity_matrix[i].copy()
            
            # Set self-similarity to a very low value to exclude it from top X
            self_similarity = similarities[i]
            similarities[i] = -999
            
            # Get indices of top X most similar APTs
            top_indices = np.argsort(similarities)[-topX:]
            top_similar_apts = [apt_names[idx] for idx in top_indices]
            
            # Check if target APT would be in top X when excluding itself
            # We need to compare against the actual self-similarity value
            if self_similarity >= np.sort(similarities)[-topX]:
                success += 1
            else:
                failure += 1
        
        print(f"Top {topX}: {success}/{success+failure} ({success/(success+failure)*100:.2f}%)")

def evaluate_profile_division(model, apt_profiles, apt_name_dict, df_mapping, apt_ids, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance with divided profile data"""
    resultdict = {}
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    
    print(f"\nEvaluating with divider: {divider} (using {100/divider:.1f}% of profile data)")
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        # Get the target APT's original data
        target_apt_id = next(apt for apt in apt_ids if apt_name_dict[apt_profiles[apt]['encoded_id']] == target_apt_name)
        target_data = apt_profiles[target_apt_id]
        
        # Create evidence with divided data for each feature type
        divided_evidence = {}
        
        for feature_type, evidence_key in [('techniques', 'techniques_str'),
                                         ('software', 'software_str'),
                                         ('capec', 'CAPEC_ID'),
                                         ('cwe', 'CWE_ID'),
                                         ('cve', 'CVE_ID'),
                                         ('pulse', 'Pulse_ID')]:
            
            if feature_type in target_data and target_data[feature_type]:
                feature_list = target_data[feature_type].copy()
                random.shuffle(feature_list)
                division_size = max(1, len(feature_list) // divider)  # At least 1 item
                divided_features = feature_list[:division_size]
                
                # Convert encoded features back to original format for sanitization
                if feature_type == 'techniques':
                    mapping = dict(zip(df_mapping['techniques_Encoded'], df_mapping['techniques_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'software':
                    mapping = dict(zip(df_mapping['software_Encoded'], df_mapping['software_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'capec':
                    mapping = dict(zip(df_mapping['CAPEC_ID_Encoded'], df_mapping['CAPEC_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'cwe':
                    mapping = dict(zip(df_mapping['CWE_ID_Encoded'], df_mapping['CWE_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'cve':
                    mapping = dict(zip(df_mapping['CVE_ID_Encoded'], df_mapping['CVE_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'pulse':
                    mapping = dict(zip(df_mapping['Pulse_ID_Encoded'], df_mapping['Pulse_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
        
        # Process the divided evidence
        clean_evidence = sanitize_evidence(divided_evidence, df_mapping)
        encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
        
        # Prepare evidence features
        evidence_features = {
            'techniques': encoded_evidence.get('techniques', []),
            'software': encoded_evidence.get('software', []),
            'capec': encoded_evidence.get('capec', []),
            'cwe': encoded_evidence.get('cwe', []),
            'cve': encoded_evidence.get('cve', []),
            'pulse': encoded_evidence.get('pulse', [])
        }
        
        # Calculate similarities with divided data
        weighted_similarities = calculate_similarities(model, apt_profiles, evidence_features, apt_ids)
        
        # Create ranked results (INCLUDE self this time!)
        ranked_results = {}
        for apt_name in apt_ids:
            apt_real_name = apt_name_dict[apt_profiles[apt_name]['encoded_id']]
            ranked_results[apt_real_name] = weighted_similarities[apt_name]
        
        resultdict[target_apt_name] = ranked_results
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SucessFailDict = {}
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
                SucessFailDict[target_apt] = "Pass"
            else:
                failure = failure + 1
                SucessFailDict[target_apt] = "Fail"
        
        print("\nSettings: TopX: " + str(topX) + " Divider: " + str(divider))
        print("Attention MIL Model for this set had a success/failure rate of:")
        print(str(success) + "/" + str(failure) + " Total: " + str(lengthResultDict))
        print("Success rate: " + str((success/lengthResultDict)*100) + "%")

def main():
    # Load everything
    model, apt_profiles, apt_name_dict, df_mapping, apt_ids = load_complete_model()
    
    print("Model loaded successfully!")
    print(f"Number of APTs: {len(apt_ids)}")
    
    # 1. Generate cross-similarity matrix for all APT pairs
    print("\n" + "="*80)
    print("GENERATING CROSS-SIMILARITY MATRIX")
    print("="*80)
    
    similarity_matrix = generate_cross_similarity_matrix(model, apt_profiles, apt_ids)
    similarity_df = create_similarity_matrix(similarity_matrix, apt_name_dict, apt_profiles, apt_ids)
    
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 2. Plot heatmap
    print("\nPlotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # 4. Process custom evidence (APT29 example)
    print("\n" + "="*80)
    print("PROCESSING CUSTOM EVIDENCE (APT29)")
    print("="*80)
    
    # Define your evidence here (APT29 example)
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
        'CAPEC_ID': ['CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-292', 'CAPEC-561', 'CAPEC-572', 'CAPEC-655', 
                    'CAPEC-35', 'CAPEC-504', 'CAPEC-543', 'CAPEC-616', 'CAPEC-564', 'CAPEC-573', 'CAPEC-560', 'CAPEC-70', 
                    'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-481', 'CAPEC-49', 
                    'CAPEC-565', 'CAPEC-555', 'CAPEC-186', 'CAPEC-443', 'CAPEC-523', 'CAPEC-669', 'CAPEC-673', 'CAPEC-678', 
                    'CAPEC-691', 'CAPEC-150', 'CAPEC-650', 'CAPEC-21', 'CAPEC-31', 'CAPEC-558', 'CAPEC-593', 'CAPEC-645', 
                    'CAPEC-60', 'CAPEC-474', 'CAPEC-485', 'CAPEC-639', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-509', 
                    'CAPEC-578', 'CAPEC-268', 'CAPEC-571', 'CAPEC-163', 'CAPEC-542', 'CAPEC-464'],
        'CWE_ID': ['CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 'CWE-1278', 'CWE-285', 'CWE-200', 
                'CWE-522', 'CWE-294', 'CWE-521', 'CWE-94', 'CWE-272', 'CWE-59', 'CWE-282', 'CWE-270', 'CWE-1021', 
                'CWE-284', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 
                'CWE-693', 'CWE-441', 'CWE-923', 'CWE-257', 'CWE-494', 'CWE-552', 'CWE-287', 'CWE-290', 'CWE-302', 
                'CWE-346', 'CWE-539', 'CWE-384', 'CWE-664', 'CWE-602', 'CWE-113', 'CWE-20', 'CWE-472', 'CWE-488', 
                'CWE-330', 'CWE-327', 'CWE-325', 'CWE-117', 'CWE-451', 'CWE-359'],
        'CVE_ID': ['CVE-2021-44228'],
        'Pulse_ID': ['']
    }
    
    # Process evidence
    print("Processing evidence...")
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
    
    print("Evidence processed:")
    print({k: len(v) for k, v in encoded_evidence.items()})
    
    # Prepare evidence features
    evidence_features = {
        'techniques': encoded_evidence.get('techniques', []),
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
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'capec': len(set(encoded_evidence.get('capec', [])) & set(apt_data['capec'])),
            'cwe': len(set(encoded_evidence.get('cwe', [])) & set(apt_data['cwe'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve'])),
            'pulse': len(set(encoded_evidence.get('pulse', [])) & set(apt_data['pulse']))
        }
        
        full_results.append({
            'Rank': rank,
            'APT': apt_name_dict[apt_data['encoded_id']],
            'Score': score,
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })
    
    # Display results
    ranked_apts = pd.DataFrame(full_results)
    
    print("\nComplete APT Ranking with Feature Matches:")
    print("="*110)
    print(f"{'Rank':<5} | {'APT':<30} | {'Encoded ID':<10} | {'Score':<8} | {'Techniques':<10} | {'Software':<8} | {'CAPEC':<6} | {'CWE':<4} | {'CVE':<4} | {'Pulse':<5}")
    print("-"*110)
    
    for _, row in ranked_apts.iterrows():
        apt_name = (row['APT'][:27] + '..') if len(row['APT']) > 29 else row['APT']
        print(f"{row['Rank']:<5} | {apt_name:<30} | {row['Encoded_ID']:<10} | {row['Score']:.4f} | {row['techniques']:<10} | {row['software']:<8} | {row['capec']:<6} | {row['cwe']:<4} | {row['cve']:<4} | {row['pulse']:<5}")


    # 3. Evaluate similarity matrix for top X matches
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX")
    print("="*80)
    
    top_x_values = [1, 5, 10, 20]
    evaluate_similarity_matrix(similarity_df, top_x_values)


    # 5. Evaluate with profile data division
    print("\n" + "="*80)
    print("PROFILE DATA DIVISION EVALUATION")
    print("="*80)
    
    evaluate_profile_division(model, apt_profiles, apt_name_dict, df_mapping, apt_ids, 
                             divider=2, top_x_values=[1, 5, 10, 20])

if __name__ == "__main__":
    main()