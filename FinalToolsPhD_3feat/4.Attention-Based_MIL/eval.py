import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from query import MILScorer

# Define the model class here to avoid pickling issues
class OptimizedAttentionMILSimilarity(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature embedding layers
        self.technique_embed = nn.Embedding(100000, hidden_dim)
        self.software_embed = nn.Embedding(100000, hidden_dim)
        self.cve_embed = nn.Embedding(100000, hidden_dim)
        
        # Attention networks
        self.attention_nets = nn.ModuleDict({
            'technique': self._build_attention(),
            'software': self._build_attention(),
            'cve': self._build_attention(),
        })
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2 + 6, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 3),
            nn.Sigmoid()
        )
        
        
        # Initialize weights with higher technique importance
        self.feature_weights = nn.Parameter(torch.tensor([6/6, 5/6, 2/6]))   # rank-sum starting values, then learned
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
            dict_key = feat_type
            apt_features = set(apt_dict.get(dict_key, []))
            evidence_features = set(evidence_dict.get(dict_key, []))
            
            if len(evidence_features) == 0:
                ratios.append(0.00)   # as in train.py / query.py
                perfect_matches.append(0)
            else:
                coverage = len(apt_features & evidence_features) / len(evidence_features)
                is_perfect = 1 if coverage == 1.0 else 0
                perfect_matches.append(is_perfect)
                
                # Linear coverage with small minimum
                ratios.append(max(coverage, 0.01))
                
        return torch.tensor(ratios), torch.tensor(perfect_matches)

    def forward(self, x1_dict, x2_dict):
        ratios_12, _ = self._calculate_match_ratios(x1_dict, x2_dict)
        ratios_21, _ = self._calculate_match_ratios(x2_dict, x1_dict)
        emb1 = self._forward_single(x1_dict, ratios_12)
        emb2 = self._forward_single(x2_dict, ratios_21)
        both_ratios = torch.cat([ratios_12, ratios_21])   # length 12
        combined = torch.cat([emb1, emb2, both_ratios.unsqueeze(0)], dim=-1)
        feature_similarities = self.similarity_net(combined)
        
        # Use softmax weights
        weights = F.softmax(self.feature_weights, dim=0)
        
        # Final score
        similarity = (feature_similarities * weights).sum(dim=-1, keepdim=True)
        return similarity

    def _forward_single(self, x_dict, match_ratios):
        features = []
        feature_types = ['technique', 'software', 'cve']
        
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


def load_complete_model():
    import pickle
    
    try:
        # Load with pickle (since the model was saved with pickle)
        with open('3feat_apt_similarity_model_complete.pth', 'rb') as f:
            save_data = pickle.load(f)
        
        print("✓ Model loaded successfully with pickle")
        
    except Exception as e:
        print(f"Pickle load failed: {e}")
        print("Trying with torch.load as fallback...")
        
        # Fallback to torch.load if it's actually a torch file
        import torch
        save_data = torch.load('3feat_apt_similarity_model_complete.pth', 
                              weights_only=False, 
                              map_location='cpu')
    
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
                'technique': apt_profiles[apt_name]['technique'],
                'software': apt_profiles[apt_name]['software'],
                'cve': apt_profiles[apt_name]['cve']
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
                'technique': apt_profiles[apt1_name]['technique'],
                'software': apt_profiles[apt1_name]['software'],
                'cve': apt_profiles[apt1_name]['cve']
            }
            
            for j in range(i, num_apts):  # Only compute upper triangle to save computation
                apt2_name = apt_ids[j]
                apt2_features = {
                    'technique': apt_profiles[apt2_name]['technique'],
                    'software': apt_profiles[apt2_name]['software'],
                    'cve': apt_profiles[apt2_name]['cve']
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
    
    
    
    # 3. Attribution evaluation with the shared protocol (full data, shuffled & divided by 2),
    #    using the same scorer as query.py / hybrid_query (model(APT profile, evidence))
    P.evaluate_attribution(MILScorer(), "MIL", results_dir="results", smoke=P.smoke_flag())

    # 4. Plot heatmap
    print("\nPlotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)


if __name__ == "__main__":
    main()