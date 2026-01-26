import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
            'pulse': self._build_attention(),
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
        for embed in [self.technique_embed, self.software_embed, self.capec_embed, self.cwe_embed, self.cve_embed, self.pulse_embed]:
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
    """Plot heatmap of similarity matrix with adaptive color scaling"""
    plt.figure(figsize=(20, 18))
    
    # Create custom red-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    
    # Analyze the data distribution
    data = similarity_df.values
    actual_min = data.min()
    actual_max = data.max()
    data_range = actual_max - actual_min
    
    print(f"Data statistics:")
    print(f"Min: {actual_min:.6f}")
    print(f"Max: {actual_max:.6f}") 
    print(f"Range: {data_range:.6f}")
    print(f"Mean: {data.mean():.6f}")
    print(f"Std: {data.std():.6f}")
    
    # Set adaptive limits based on data range
    if data_range < 0.01:  # Very small range - use tight limits
        vmin = actual_min
        vmax = actual_max
        print("Using tight color scale (range < 0.01)")
    elif data_range < 0.05:  # Small range - focus on upper 80%
        vmin = actual_max - (data_range * 0.8)
        vmax = actual_max
        print("Using focused color scale (upper 80% of range)")
    else:  # Larger range - use full range
        vmin = actual_min
        vmax = actual_max
        print("Using full color scale")
    
    print(f"Color scale limits: {vmin:.6f} to {vmax:.6f}")
    
    # Plot heatmap
    sns.heatmap(similarity_df, 
                mask=None,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=(vmin + vmax) / 2,
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Similarity Score'})
    
    plt.title(f"APT Group Cross-Similarity Matrix\nColor scale: {vmin:.4f} to {vmax:.4f}", 
              fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig('apt_MIL_similarity_matrix.png', dpi=300, bbox_inches='tight')
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
                                         ('software', 'software_str'), ('capec', 'CAPEC_ID'),
                                         ('cwe', 'CWE_ID'), ('cve', 'CVE_ID'), ('pulse', 'Pulse_ID'),]:
            
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
        from query import sanitize_evidence, translate_to_encoded
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
    
    # 3. Evaluate similarity matrix for top X matches
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX")
    print("="*80)
    
    top_x_values = [1, 5, 10, 20]
    evaluate_similarity_matrix(similarity_df, top_x_values)

    # 4. Evaluate with profile data division
    print("\n" + "="*80)
    print("PROFILE DATA DIVISION EVALUATION")
    print("="*80)
    
    evaluate_profile_division(model, apt_profiles, apt_name_dict, df_mapping, apt_ids, 
                             divider=2, top_x_values=[1, 5, 10, 20])

if __name__ == "__main__":
    main()