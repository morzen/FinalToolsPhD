import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class APTSiameseNetwork(nn.Module):
    def __init__(self, num_apts, embedding_dim=64):
        super().__init__()
        self.apt_embedding = nn.Embedding(num_apts, embedding_dim)
        
    def forward(self, apt_idx1, apt_idx2):
        # Get embeddings for both APTs
        emb1 = self.apt_embedding(apt_idx1)
        emb2 = self.apt_embedding(apt_idx2)
        return emb1, emb2
    
    def get_similarity(self, emb1, emb2):
        # Cosine similarity
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).squeeze()
    
    def get_all_embeddings(self):
        # Get embeddings for all APTs
        return self.apt_embedding.weight

def create_apt_profiles_with_names(df_encoded, name_mapping):
    profiles = {}
    
    for _, row in df_encoded.iterrows():
        apt_encoded = row['APTname']
        apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
        
        if apt_name not in profiles:
            profiles[apt_name] = {
                'encoded_id': apt_encoded,
                'techniques': set(),
                'software': set(),
                'capec': set(),
                'cwe': set(),
                'cve': set(),
                'pulse': set()
            }
        
        if row['techniques_str'] != -1:
            profiles[apt_name]['techniques'].add(row['techniques_str'])
        if row['software_str'] != -1:
            profiles[apt_name]['software'].add(row['software_str'])
        if row['CAPEC_ID'] != -1:
            profiles[apt_name]['capec'].add(row['CAPEC_ID'])
        if row['CWE_ID'] != -1:
            profiles[apt_name]['cwe'].add(row['CWE_ID'])
        if row['CVE_ID'] != -1:
            profiles[apt_name]['cve'].add(row['CVE_ID'])
        if row['Pulse_ID'] != -1:
            profiles[apt_name]['pulse'].add(row['Pulse_ID'])
    
    return profiles

def create_similarity_dataset(apt_profiles, apt_ids, max_pairs=1000):
    """Create training pairs with similarity labels"""
    pairs = []
    labels = []
    
    # Calculate Jaccard similarity for ground truth
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    # Create feature sets for each APT
    apt_feature_sets = {}
    for apt_id in apt_ids:
        profile = apt_profiles[apt_id]
        all_features = set()
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            all_features.update(profile[key])
        apt_feature_sets[apt_id] = all_features
    
    n_apts = len(apt_ids)
    
    # Create diverse pairs
    for i in range(n_apts):
        for j in range(i + 1, n_apts):
            if len(pairs) >= max_pairs:
                break
                
            apt1 = apt_ids[i]
            apt2 = apt_ids[j]
            
            sim = jaccard_similarity(apt_feature_sets[apt1], apt_feature_sets[apt2])
            pairs.append((i, j))
            labels.append(sim)
    
    return pairs, labels

def train_siamese_model(model, pairs, labels, num_epochs=50, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    print(f"Starting training with {len(pairs)} pairs, {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle the data
        indices = torch.randperm(len(pairs))
        shuffled_pairs = [pairs[i] for i in indices]
        shuffled_labels = labels_tensor[indices]
        
        # Train in batches
        for batch_start in range(0, len(shuffled_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled_pairs))
            batch_pairs = shuffled_pairs[batch_start:batch_end]
            batch_labels = shuffled_labels[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            batch_loss = 0
            for (idx1, idx2), label in zip(batch_pairs, batch_labels):
                # Convert to tensors
                idx1_tensor = torch.tensor(idx1, dtype=torch.long)
                idx2_tensor = torch.tensor(idx2, dtype=torch.long)
                
                # Get embeddings and similarity
                emb1, emb2 = model(idx1_tensor, idx2_tensor)
                pred_similarity = model.get_similarity(emb1, emb2)
                
                # Calculate loss
                loss = criterion(pred_similarity, label)
                batch_loss += loss
            
            # Average loss over the batch
            if len(batch_pairs) > 0:
                batch_loss = batch_loss / len(batch_pairs)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def save_complete_model(model, apt_embeddings, apt_profiles, apt_name_dict, apt_ids, apt_to_idx, df_mapping, save_path="./DataForSiamese/siamese_complete_model.pkl"):
    """Save everything in one pickle file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare all data for saving
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_apts': len(apt_ids),
            'embedding_dim': model.apt_embedding.embedding_dim
        },
        'apt_embeddings': apt_embeddings,
        'apt_profiles': apt_profiles,
        'apt_name_dict': apt_name_dict,
        'apt_ids': apt_ids,
        'apt_to_idx': apt_to_idx,
        'df_mapping': df_mapping
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Complete model saved to {save_path}")

def load_complete_model(load_path):
    """Load everything from one pickle file"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create model instance
    model = APTSiameseNetwork(data['model_config']['num_apts'], 
                             data['model_config']['embedding_dim'])
    
    # Load state dict
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    return model, data

def main():
    pwd = "./DataForSiamese/"
    
    # Load data
    print("Loading data...")
    df_encoded = pd.read_csv(pwd + 'data_for_Siamese.csv')
    df_mapping = pd.read_csv(pwd + 'comparison_Encodedtest_df.csv')
    
    # Create APT name mapping
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
    
    # Create APT profiles
    print("Creating APT profiles...")
    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)
    
    # Convert sets to lists
    for apt_name in apt_profiles:
        profile = apt_profiles[apt_name]
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            profile[key] = list(profile[key])
    
    # Get APT IDs
    apt_ids = list(apt_profiles.keys())
    print(f"Found {len(apt_ids)} APT groups")
    
    # Create mapping from APT name to index
    apt_to_idx = {apt: i for i, apt in enumerate(apt_ids)}
    
    # Create similarity dataset for training
    print("Creating training pairs...")
    pairs, labels = create_similarity_dataset(apt_profiles, apt_ids, max_pairs=2000)
    
    print(f"Created {len(pairs)} training pairs")
    
    # Create and train Siamese model
    print("Initializing model...")
    model = APTSiameseNetwork(len(apt_ids))
    
    print("Starting training...")
    train_siamese_model(model, pairs, labels, num_epochs=50, batch_size=32)
    
    # Get all embeddings for inference (detach to remove gradients)
    print("Generating final embeddings...")
    model.eval()
    with torch.no_grad():
        all_embeddings = model.get_all_embeddings().detach()
    
    # Save everything in one file
    save_path = os.path.join(pwd, 'siamese_complete_model.pkl')
    save_complete_model(model, all_embeddings, apt_profiles, apt_name_dict, apt_ids, apt_to_idx, df_mapping, save_path)
    
    print("Siamese model trained and saved successfully!")
    print(f"Number of APTs: {len(apt_ids)}")
    print(f"Number of training pairs: {len(pairs)}")

def test_similarity():
    """Test the trained model by loading it from the single file"""
    pwd = "./DataForSiamese/"
    model_path = os.path.join(pwd, 'siamese_complete_model.pkl')
    
    # Load the complete model
    model, data = load_complete_model(model_path)
    
    # Test similarity between first two APTs
    apt_names = list(data['apt_to_idx'].keys())
    if len(apt_names) >= 2:
        apt1_name = apt_names[0]
        apt2_name = apt_names[1]
        
        idx1 = data['apt_to_idx'][apt1_name]
        idx2 = data['apt_to_idx'][apt2_name]
        
        with torch.no_grad():
            emb1, emb2 = model(torch.tensor(idx1), torch.tensor(idx2))
            similarity = model.get_similarity(emb1, emb2).item()
            
        print(f"Similarity between {apt1_name} and {apt2_name}: {similarity:.4f}")
    
    # Get similarity matrix for all APTs
    embeddings = data['apt_embeddings'].numpy()
    similarity_matrix = cosine_similarity(embeddings)
    
    print("Similarity matrix shape:", similarity_matrix.shape)
    
    # Show some example similarities
    print("\nTop 5 most similar APT pairs:")
    n_apts = len(data['apt_ids'])
    similarities = []
    for i in range(n_apts):
        for j in range(i + 1, n_apts):
            similarities.append((i, j, similarity_matrix[i, j]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, sim in similarities[:5]:
        apt1 = data['apt_ids'][i]
        apt2 = data['apt_ids'][j]
        print(f"{apt1} - {apt2}: {sim:.4f}")
    
    return similarity_matrix

if __name__ == "__main__":
    main()
    # After training, test the model by loading from the single file
    print("\nTesting loaded model:")
    similarity_matrix = test_similarity()