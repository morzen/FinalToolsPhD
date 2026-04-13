# ============================
# 1. IMPORTS
# ============================
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import pickle
import os
import random

# ============================
# 2. CONFIGURATION
# ============================
class AttentionMILConfig:
    """Configuration for Attention-based MIL"""
    #variable shared by all models:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "3feat_apt_similarity_model_complete.pth"

    #variables share by some models:
    MAPPING_PATH = "../BaseData/comparison_Encoded_df.csv"
    HIDDEN_DIM = 256
    FEATURE_WEIGHTS = {
        'technique': 0.8,
        'software': 0.7,
        'cve': 0.4
    }

    #variables unique to this model:
    EMBEDDING_VOCAB_SIZE = 10000  # For each feature type
        # Training parameters
    PERFECT_MATCH_BOOST = 2.0
    
    # New training parameters
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 3. DATA LOADING & PREPROCESSING
# ============================
class AttentionMILDataLoader:
    """Handles data loading and preprocessing for Attention MIL"""
    
    def __init__(self):
        self.df_encoded = None
        self.df_mapping = None
        self.apt_name_dict = {}
        
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        self.df_encoded = pd.read_csv(AttentionMILConfig.DATA_PATH)
        self.df_mapping = pd.read_csv(AttentionMILConfig.MAPPING_PATH)
        print(f"Encoded data: {len(self.df_encoded)} records")
        print(f"Mapping data: {len(self.df_mapping)} records")
        return self.df_encoded, self.df_mapping
    
    def create_name_mapping(self):
        """Create mapping from encoded IDs to original APT names"""
        apt_name_mapping = self.df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
        self.apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
        print(f"Created mapping for {len(self.apt_name_dict)} APTs")
        return self.apt_name_dict
    
    def create_apt_profiles(self):
        """Create comprehensive APT profiles with names"""
        print("\nCreating APT profiles...")
        
        profiles = {}
        feature_types = ['technique', 'software', 'cve']
        missing_apts = []  # Collect missing encoded IDs
        
        for _, row in self.df_encoded.iterrows():
            apt_encoded = row['APTname']
            
            # Check if mapping exists
            if apt_encoded not in self.apt_name_dict:
                missing_apts.append(apt_encoded)
                continue  # Skip this row for now; we'll raise error later
            
            apt_name = self.apt_name_dict[apt_encoded]
            
            if apt_name not in profiles:
                profiles[apt_name] = {
                    'encoded_id': apt_encoded,
                    'technique': set(),
                    'software': set(),
                    'cve': set()
                }
            
            # Add features if present
            self._add_feature(profiles[apt_name], 'technique', row['techniques_str'])
            self._add_feature(profiles[apt_name], 'software', row['software_str'])
            self._add_feature(profiles[apt_name], 'cve', row['CVE_ID'])
        
        # If any missing APTs were found, raise an error
        if missing_apts:
            print("\nERROR: The following encoded APT IDs are missing from the mapping dataframe:")
            for apt_id in missing_apts:
                print(f"  - {apt_id}")
            print("\nPlease ensure the mapping file 'comparison_Encoded_df.csv' contains all APT IDs present in 'EncodedData.csv'.")
            raise KeyError(f"Missing mapping for APT IDs: {missing_apts}")
        
        # Convert sets to lists and add counts
        for apt_name in profiles:
            profile = profiles[apt_name]
            for key in feature_types:
                profile[key] = list(profile[key])
                profile[f'{key}_count'] = len(profile[key])
        
        print(f"Created profiles for {len(profiles)} APTs")
        self._print_profile_statistics(profiles)
        
        return profiles 
    
    def _add_feature(self, profile, feature_name, value):
        """Add feature value if valid"""
        if value != -1:
            profile[feature_name].add(value)
    
    def _print_profile_statistics(self, profiles):
        """Print statistics about APT profiles"""
        print("\nProfile Statistics:")
        print("=" * 50)
        
        feature_types = ['technique', 'software', 'cve']
        stats = {ft: 0 for ft in feature_types}
        
        for profile in profiles.values():
            for ft in feature_types:
                stats[ft] += len(profile[ft])
        
        for ft in feature_types:
            print(f"  Total {ft}: {stats[ft]}")
        
        # Show sample profiles
        print("\nSample Profiles (first 3):")
        sample_apts = list(profiles.keys())[:3]
        for apt_name in sample_apts:
            profile = profiles[apt_name]
            print(f"\n  {apt_name}:")
            for ft in feature_types:
                count = len(profile[ft])
                sample = profile[ft][:3] if count > 0 else []
                print(f"    {ft}: {count} features, sample: {sample}")

# ============================
# 4. MODEL DEFINITION
# ============================
class OptimizedAttentionMILSimilarity(nn.Module):  # KEEP THE ORIGINAL NAME
    """Optimized Attention-based MIL model for APT similarity"""
    
    def __init__(self, hidden_dim=AttentionMILConfig.HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature embedding layers
        self.technique_embed = nn.Embedding(AttentionMILConfig.EMBEDDING_VOCAB_SIZE, hidden_dim)
        self.software_embed = nn.Embedding(AttentionMILConfig.EMBEDDING_VOCAB_SIZE, hidden_dim)
        self.cve_embed = nn.Embedding(AttentionMILConfig.EMBEDDING_VOCAB_SIZE, hidden_dim)
        
        # Attention networks
        self.attention_nets = nn.ModuleDict({
            'technique': self._build_attention(),
            'software': self._build_attention(),
            'cve': self._build_attention(),
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
        self.feature_weights = nn.Parameter(torch.tensor([0.8, 0.7, 0.4]))
        self.perfect_match_boost = 2.0  # Direct score multiplier for perfect matches
        
        self.init_weights()

    def _build_attention(self):
        """Build attention network for a feature type"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

    def init_weights(self):
        """Initialize weights for embeddings and linear layers"""
        print("\nInitializing model weights...")
        
        # Initialize embedding layers
        for embed in [self.technique_embed, self.software_embed, self.cve_embed]:
            nn.init.xavier_uniform_(embed.weight)
        
        # Initialize attention networks
        for net in self.attention_nets.values():
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Initialize similarity network
        for layer in self.similarity_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        print("Weights initialized successfully")

    def _calculate_match_ratios(self, apt_dict, evidence_dict):
        """Calculate match ratios between APT and evidence features"""
        ratios = []
        perfect_matches = []
        feature_types = ['technique', 'software', 'cve']
        
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
        """Forward pass for similarity calculation"""
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
        """Forward pass for a single APT profile"""
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

# ============================
# 5. MODEL INITIALIZATION
# ============================
class ModelInitializer:
    """Handles model initialization and setup"""
    
    @staticmethod
    def initialize_model():
        """Initialize the Attention MIL model"""
        print("\nInitializing Attention MIL Model...")
        print("=" * 50)
        
        model = OptimizedAttentionMILSimilarity()
        
        print(f"\nModel Architecture:")
        print(f"  Hidden dimension: {AttentionMILConfig.HIDDEN_DIM}")
        print(f"  Embedding vocabulary size: {AttentionMILConfig.EMBEDDING_VOCAB_SIZE}")
        print(f"  Feature weights: {AttentionMILConfig.FEATURE_WEIGHTS}")
        print(f"  Perfect match boost: {AttentionMILConfig.PERFECT_MATCH_BOOST}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter Count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model

# ============================
# 6. TRAINING FUNCTION
# ============================
class AttentionMILTrainer:
    """Simple training function for the MIL model"""
    
    @staticmethod
    def train_model(model, apt_profiles, epochs=AttentionMILConfig.EPOCHS, 
                learning_rate=AttentionMILConfig.LEARNING_RATE):
        """Train the model on APT similarity"""
        print("\n" + "=" * 60)
        print("TRAINING THE MODEL")
        print("=" * 60)
        
        # Move model to device
        device = torch.device(AttentionMILConfig.DEVICE)
        model = model.to(device)
        model.train()
        
        # Create simple training data
        apt_names = list(apt_profiles.keys())
        train_pairs = []
        train_labels = []
        
        # Create positive pairs (same APT)
        for apt_name in apt_names:
            train_pairs.append((apt_name, apt_name))
            train_labels.append(1.0)  # Same APT = high similarity
        
        # Create negative pairs (different APTs)
        for _ in range(len(apt_names)):
            apt1 = random.choice(apt_names)
            apt2 = random.choice(apt_names)
            if apt1 != apt2:
                train_pairs.append((apt1, apt2))
                train_labels.append(0.0)  # Different APT = low similarity
        
        # Convert to tensors
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
        
        # Setup training
        criterion = nn.BCELoss()  # Binary cross entropy for similarity score
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training on {len(train_pairs)} pairs")
        print(f"Using device: {device}")
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            # Process all pairs
            predictions = []
            for apt1_name, apt2_name in train_pairs:
                profile1 = apt_profiles[apt1_name]
                profile2 = apt_profiles[apt2_name]
                
                # Forward pass
                similarity_score = model(profile1, profile2)
                predictions.append(similarity_score)
            
            # Stack predictions
            predictions = torch.cat(predictions, dim=0)
            
            # FIX: Squeeze predictions to match labels shape
            predictions = predictions.squeeze()  # Remove the extra dimension
            
            # Calculate loss
            loss = criterion(predictions, train_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss = loss.item()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_pred = predictions.mean().item()
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss:.6f} | Avg Similarity: {avg_pred:.4f}")
        
        print(f"\nTraining completed!")
        print(f"Final loss: {total_loss:.6f}")
        
        return model

# ============================
# 7. MODEL SAVING
# ============================
# ============================
# 7. MODEL SAVING - FIXED VERSION
# ============================
class AttentionMILModelSaver:
    """Handles saving of Attention MIL model"""
    
    @staticmethod
    def save_model(model, apt_profiles, apt_name_dict, df_mapping, apt_ids, 
                   filename=AttentionMILConfig.MODEL_PATH):
        """Save complete model with all required data"""
        print("\nSaving model and data...")
        
        # DO NOT SAVE THE CLASS ITSELF - it causes pickling issues
        save_data = {
            'model_state_dict': model.state_dict(),
            'apt_profiles': apt_profiles,
            'apt_name_dict': apt_name_dict,
            'df_mapping': df_mapping,
            'apt_ids': apt_ids,
            # DO NOT SAVE THE CLASS: 'model_class': OptimizedAttentionMILSimilarity,  # REMOVE THIS LINE
            'config': {
                'hidden_dim': AttentionMILConfig.HIDDEN_DIM,
                'embedding_vocab_size': AttentionMILConfig.EMBEDDING_VOCAB_SIZE,
                'feature_weights': AttentionMILConfig.FEATURE_WEIGHTS,
                'perfect_match_boost': AttentionMILConfig.PERFECT_MATCH_BOOST
            }
        }
        
        # Save with pickle protocol 2 for compatibility
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f, protocol=2)
        
        print(f"Model saved to '{filename}'")
        print(f"  APT profiles: {len(apt_profiles)}")
        print(f"  APT IDs: {len(apt_ids)}")
        print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
        
        return save_data
    
    
# ============================
# 8. MAIN EXECUTION
# ============================
def main():
    """Execute the Attention MIL model creation pipeline"""
    print("ATTENTION-BASED MIL FOR APT SIMILARITY")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n1. DATA LOADING AND PREPROCESSING")
    data_loader = AttentionMILDataLoader()
    df_encoded, df_mapping = data_loader.load_data()
    apt_name_dict = data_loader.create_name_mapping()
    apt_profiles = data_loader.create_apt_profiles()
    apt_ids = list(apt_profiles.keys())
    
    # 2. Initialize model
    print("\n2. MODEL INITIALIZATION")
    model = ModelInitializer.initialize_model()
    
    # 3. Train model
    print("\n3. MODEL TRAINING")
    trained_model = AttentionMILTrainer.train_model(model, apt_profiles)
    
    # 4. Save model
    print("\n4. MODEL SAVING")
    save_data = AttentionMILModelSaver.save_model(
        trained_model, apt_profiles, apt_name_dict, df_mapping, apt_ids
    )
    
    print("\nTRAINING SUMMARY")
    print("=" * 60)
    print(f"• Model: Optimized Attention-based MIL")
    print(f"• Hidden dimension: {AttentionMILConfig.HIDDEN_DIM}")
    print(f"• Feature weights: {AttentionMILConfig.FEATURE_WEIGHTS}")
    print(f"• APT profiles created: {len(apt_profiles)}")
    print(f"• Training epochs: {AttentionMILConfig.EPOCHS}")
    print(f"• Learning rate: {AttentionMILConfig.LEARNING_RATE}")
    print(f"• Saved to: {AttentionMILConfig.MODEL_PATH}")
    print("\nModel is ready for similarity calculations!")
    
    return save_data

if __name__ == "__main__":
    saved_data = main()
