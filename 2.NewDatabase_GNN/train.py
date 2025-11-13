import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import defaultdict
import pickle
import numpy as np
import os

###################################################################
#data pre processing and profile building for the GNN

def load_training_data(pwd="../BaseData/"):
    """Load only the training data (EncodedData.csv)"""
    print("Loading training data...")
    
    # Load encoded data only
    df_encoded = pd.read_csv(pwd + 'EncodedData.csv')
    print()
    
    print(f"Training data: {len(df_encoded)} records")
    print("\nSample training data:")
    print(df_encoded.head(10))
    
    return df_encoded

def create_apt_profiles_simple(df_encoded):
    """Create APT profiles directly from encoded data"""
    print("\n📊 CREATING APT PROFILES FROM ENCODED DATA")
    print("=" * 50)
    
    # Get unique APT IDs
    unique_apts = sorted(df_encoded['APTname'].unique())
    print(f"Found {len(unique_apts)} unique APTs")
    
    # Build APT profiles
    profiles = {}
    for apt_id in unique_apts:
        # Get all records for this APT
        apt_records = df_encoded[df_encoded['APTname'] == apt_id]
        
        profiles[apt_id] = {
            'encoded_id': apt_id,
            'techniques': set(),
            'software': set(), 
            'capec': set(),
            'cwe': set(),
            'cve': set(),
            'pulse': set()
        }
        
        # Collect all features for this APT (skip -1 values)
        for _, row in apt_records.iterrows():
            # Techniques (skip -1)
            if row['techniques_str'] != -1:
                profiles[apt_id]['techniques'].add(row['techniques_str'])
            
            # Software (skip -1)
            if row['software_str'] != -1:
                profiles[apt_id]['software'].add(row['software_str'])

            if row['CAPEC_ID'] != -1:
                profiles[apt_id]['capec'].add(row['CAPEC_ID'])
            
            if row['CWE_ID'] != -1:
                profiles[apt_id]['cwe'].add(row['CWE_ID'])
            
            # CVE (skip -1)
            if row['CVE_ID'] != -1:
                profiles[apt_id]['cve'].add(row['CVE_ID'])
            
            if row['Pulse_ID'] != -1:
                profiles[apt_id]['pulse'].add(row['Pulse_ID'])
    
    # Print profile statistics
    print(f"\n📈 PROFILE STATISTICS:")
    total_tech = sum(len(p['techniques']) for p in profiles.values())
    total_soft = sum(len(p['software']) for p in profiles.values())
    total_capec = sum(len(p['capec']) for p in profiles.values())
    total_cwe = sum(len(p['cwe']) for p in profiles.values())
    total_cve = sum(len(p['cve']) for p in profiles.values())
    total_pulse = sum(len(p['pulse']) for p in profiles.values())
    
    print(f"Total APTs: {len(profiles)}")
    print(f"Total techniques: {total_tech}")
    print(f"Total software: {total_soft}")
    print(f"Total capec: {total_capec}")
    print(f"Total cwe: {total_cwe}")
    print(f"Total CVEs: {total_cve}")
    print(f"Total pulse: {total_pulse}")
    
    # Show first 3 profiles as examples
    print(f"\n🧩 SAMPLE PROFILES (first 3 APTs):")
    for i, (apt_id, profile) in enumerate(list(profiles.items())[:3]):
        print(f"  APT {apt_id}:")
        print(f"    Techniques: {len(profile['techniques'])} - sample: {list(profile['techniques'])[:3]}")
        print(f"    Software: {len(profile['software'])} - sample: {list(profile['software'])[:3]}")
        print(f"    Capec: {len(profile['capec'])} - sample: {list(profile['capec'])[:3]}")
        print(f"    CWE: {len(profile['cwe'])} - sample: {list(profile['cwe'])[:3]}")
        print(f"    CVEs: {len(profile['cve'])} - sample: {list(profile['cve'])[:3]}")
        print(f"    Pulse: {len(profile['pulse'])} - sample: {list(profile['pulse'])[:3]}")
    
    return profiles, unique_apts

#####################################################################
# creation of GNN and training function
class APTGraphModel(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwe, num_cve, num_pulse, hidden_dim=128):
        super().__init__()
        # Feature embeddings for all 6 feature types
        self.technique_embedding = nn.Embedding(num_techniques + 1, hidden_dim, padding_idx=0)
        self.software_embedding = nn.Embedding(num_software + 1, hidden_dim, padding_idx=0)
        self.capec_embedding = nn.Embedding(num_capec + 1, hidden_dim, padding_idx=0)
        self.cwe_embedding = nn.Embedding(num_cwe + 1, hidden_dim, padding_idx=0)
        self.cve_embedding = nn.Embedding(num_cve + 1, hidden_dim, padding_idx=0)
        self.pulse_embedding = nn.Embedding(num_pulse + 1, hidden_dim, padding_idx=0)
        
        # Combine all 6 features
        self.feature_combiner = nn.Linear(hidden_dim * 6, hidden_dim)
        
    def forward(self, techniques, software, capec, cwe, cves, pulse):
        # Embed each feature type
        tech_emb = self.technique_embedding(techniques).mean(dim=1)
        soft_emb = self.software_embedding(software).mean(dim=1)
        capec_emb = self.capec_embedding(capec).mean(dim=1)
        cwe_emb = self.cwe_embedding(cwe).mean(dim=1)
        cve_emb = self.cve_embedding(cves).mean(dim=1)
        pulse_emb = self.pulse_embedding(pulse).mean(dim=1)
        
        # Combine all features
        combined = torch.cat([tech_emb, soft_emb, capec_emb, cwe_emb, cve_emb, pulse_emb], dim=1)
        query_embedding = self.feature_combiner(combined)
        
        return query_embedding

def create_feature_mappings_from_profiles(profiles):
    """Create feature mappings directly from profiles for all 6 feature types"""
    print("\n🔧 CREATING FEATURE MAPPINGS FROM PROFILES")
    
    # Collect all unique features from profiles for all 6 types
    all_techniques = set()
    all_software = set()
    all_capec = set()
    all_cwe = set()
    all_cve = set()
    all_pulse = set()
    
    for profile in profiles.values():
        all_techniques.update(profile['techniques'])
        all_software.update(profile['software'])
        all_capec.update(profile['capec'])
        all_cwe.update(profile['cwe'])
        all_cve.update(profile['cve'])
        all_pulse.update(profile['pulse'])
    
    # Create mappings with 1-based indexing (0 reserved for padding)
    technique_to_idx = {tech: i+1 for i, tech in enumerate(all_techniques)}
    software_to_idx = {soft: i+1 for i, soft in enumerate(all_software)}
    capec_to_idx = {capec: i+1 for i, capec in enumerate(all_capec)}
    cwe_to_idx = {cwe: i+1 for i, cwe in enumerate(all_cwe)}
    cve_to_idx = {cve: i+1 for i, cve in enumerate(all_cve)}
    pulse_to_idx = {pulse: i+1 for i, pulse in enumerate(all_pulse)}
    
    print(f"Feature mappings created:")
    print(f"  Techniques: {len(technique_to_idx)}")
    print(f"  Software: {len(software_to_idx)}")
    print(f"  CAPEC: {len(capec_to_idx)}")
    print(f"  CWE: {len(cwe_to_idx)}")
    print(f"  CVEs: {len(cve_to_idx)}")
    print(f"  Pulse: {len(pulse_to_idx)}")
    
    return technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, cve_to_idx, pulse_to_idx


def proper_supervised_training(query_model, profiles, apt_ids, technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, 
                            cve_to_idx, pulse_to_idx, epochs=200):
    """Train the model to learn similarities between APTs using all 6 features"""
    optimizer = torch.optim.Adam(query_model.parameters(), lr=0.001)
    
    print(f"🚀 SUPERVISED TRAINING FOR {epochs} EPOCHS...")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for i, apt_id in enumerate(apt_ids):
            # Get current APT's features for all 6 types
            techniques_i = list(profiles[apt_id]['techniques'])
            software_i = list(profiles[apt_id]['software'])
            capec_i = list(profiles[apt_id]['capec'])
            cwe_i = list(profiles[apt_id]['cwe'])
            cves_i = list(profiles[apt_id]['cve'])
            pulse_i = list(profiles[apt_id]['pulse'])
            
            # Convert to indices
            tech_indices_i = [technique_to_idx.get(t, 0) for t in techniques_i] if techniques_i else [0]
            soft_indices_i = [software_to_idx.get(s, 0) for s in software_i] if software_i else [0]
            capec_indices_i = [capec_to_idx.get(c, 0) for c in capec_i] if capec_i else [0]
            cwe_indices_i = [cwe_to_idx.get(c, 0) for c in cwe_i] if cwe_i else [0]
            cve_indices_i = [cve_to_idx.get(c, 0) for c in cves_i] if cves_i else [0]
            pulse_indices_i = [pulse_to_idx.get(p, 0) for p in pulse_i] if pulse_i else [0]
            
            # Convert to tensors
            tech_tensor_i = torch.tensor(tech_indices_i, dtype=torch.long)
            soft_tensor_i = torch.tensor(soft_indices_i, dtype=torch.long)
            capec_tensor_i = torch.tensor(capec_indices_i, dtype=torch.long)
            cwe_tensor_i = torch.tensor(cwe_indices_i, dtype=torch.long)
            cve_tensor_i = torch.tensor(cve_indices_i, dtype=torch.long)
            pulse_tensor_i = torch.tensor(pulse_indices_i, dtype=torch.long)
            
            # Compute embedding
            apt_embedding_i = query_model(
                tech_tensor_i.unsqueeze(0), 
                soft_tensor_i.unsqueeze(0),
                capec_tensor_i.unsqueeze(0),
                cwe_tensor_i.unsqueeze(0),
                cve_tensor_i.unsqueeze(0),
                pulse_tensor_i.unsqueeze(0)
            )
            
            # For each other APT, compute similarity
            batch_loss = 0
            comparison_count = 0
            
            for j, other_apt_id in enumerate(apt_ids):
                if i == j:
                    continue
                
                # Get other APT's features
                techniques_j = list(profiles[other_apt_id]['techniques'])
                software_j = list(profiles[other_apt_id]['software'])
                capec_j = list(profiles[other_apt_id]['capec'])
                cwe_j = list(profiles[other_apt_id]['cwe'])
                cves_j = list(profiles[other_apt_id]['cve'])
                pulse_j = list(profiles[other_apt_id]['pulse'])
                
                # Convert to indices
                tech_indices_j = [technique_to_idx.get(t, 0) for t in techniques_j] if techniques_j else [0]
                soft_indices_j = [software_to_idx.get(s, 0) for s in software_j] if software_j else [0]
                capec_indices_j = [capec_to_idx.get(c, 0) for c in capec_j] if capec_j else [0]
                cwe_indices_j = [cwe_to_idx.get(c, 0) for c in cwe_j] if cwe_j else [0]
                cve_indices_j = [cve_to_idx.get(c, 0) for c in cves_j] if cves_j else [0]
                pulse_indices_j = [pulse_to_idx.get(p, 0) for p in pulse_j] if pulse_j else [0]
                
                # Convert to tensors
                tech_tensor_j = torch.tensor(tech_indices_j, dtype=torch.long)
                soft_tensor_j = torch.tensor(soft_indices_j, dtype=torch.long)
                capec_tensor_j = torch.tensor(capec_indices_j, dtype=torch.long)
                cwe_tensor_j = torch.tensor(cwe_indices_j, dtype=torch.long)
                cve_tensor_j = torch.tensor(cve_indices_j, dtype=torch.long)
                pulse_tensor_j = torch.tensor(pulse_indices_j, dtype=torch.long)
                
                # Compute other embedding
                apt_embedding_j = query_model(
                    tech_tensor_j.unsqueeze(0), 
                    soft_tensor_j.unsqueeze(0),
                    capec_tensor_j.unsqueeze(0),
                    cwe_tensor_j.unsqueeze(0),
                    cve_tensor_j.unsqueeze(0),
                    pulse_tensor_j.unsqueeze(0)
                )
                
                # Calculate predicted similarity (dot product + sigmoid)
                predicted_sim = torch.sigmoid(torch.matmul(apt_embedding_i, apt_embedding_j.t()).squeeze())
                
                # Calculate ground truth similarity with weights for all 6 features
                profile_i = profiles[apt_id]
                profile_j = profiles[other_apt_id]
                
                # Feature weights
                TECHNIQUE_WEIGHT = 5.0
                SOFTWARE_WEIGHT = 3.0 
                CAPEC_WEIGHT = 1.0
                CWE_WEIGHT = 1.0 
                CVE_WEIGHT = 0.5
                PULSE_WEIGHT = 0.2
                
                # Calculate overlaps
                tech_overlap = len(profile_i['techniques'] & profile_j['techniques'])
                soft_overlap = len(profile_i['software'] & profile_j['software'])
                capec_overlap = len(profile_i['capec'] & profile_j['capec'])
                cwe_overlap = len(profile_i['cwe'] & profile_j['cwe'])
                cve_overlap = len(profile_i['cve'] & profile_j['cve'])
                pulse_overlap = len(profile_i['pulse'] & profile_j['pulse'])
                
                ground_truth_sim = (tech_overlap * TECHNIQUE_WEIGHT + 
                                   soft_overlap * SOFTWARE_WEIGHT +
                                   capec_overlap * CAPEC_WEIGHT +
                                   cwe_overlap * CWE_WEIGHT +
                                   cve_overlap * CVE_WEIGHT +
                                   pulse_overlap * PULSE_WEIGHT)
                
                # Normalize by maximum possible
                max_i = (len(profile_i['techniques']) * TECHNIQUE_WEIGHT +
                        len(profile_i['software']) * SOFTWARE_WEIGHT +
                        len(profile_i['capec']) * CAPEC_WEIGHT +
                        len(profile_i['cwe']) * CWE_WEIGHT +
                        len(profile_i['cve']) * CVE_WEIGHT +
                        len(profile_i['pulse']) * PULSE_WEIGHT)
                
                max_j = (len(profile_j['techniques']) * TECHNIQUE_WEIGHT +
                        len(profile_j['software']) * SOFTWARE_WEIGHT +
                        len(profile_j['capec']) * CAPEC_WEIGHT +
                        len(profile_j['cwe']) * CWE_WEIGHT +
                        len(profile_j['cve']) * CVE_WEIGHT +
                        len(profile_j['pulse']) * PULSE_WEIGHT)
                
                max_possible = min(max_i, max_j)
                if max_possible > 0:
                    ground_truth_sim = ground_truth_sim / max_possible
                else:
                    ground_truth_sim = 0.0
                
                # Loss for this pair
                pair_loss = F.mse_loss(predicted_sim, torch.tensor(ground_truth_sim, dtype=torch.float32))
                batch_loss += pair_loss
                comparison_count += 1
            
            if comparison_count > 0:
                # Average loss over all comparisons for this APT
                avg_batch_loss = batch_loss / comparison_count
                
                optimizer.zero_grad()
                avg_batch_loss.backward()
                optimizer.step()
                
                total_loss += avg_batch_loss.item()
                batch_count += 1
        
        if epoch % 20 == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}')
    
    # Generate final embeddings after training
    query_model.eval()
    with torch.no_grad():
        apt_embeddings = []
        for apt_id in apt_ids:
            techniques = list(profiles[apt_id]['techniques'])
            software = list(profiles[apt_id]['software'])
            capec = list(profiles[apt_id]['capec'])
            cwe = list(profiles[apt_id]['cwe'])
            cves = list(profiles[apt_id]['cve'])
            pulse = list(profiles[apt_id]['pulse'])
            
            # Convert to indices
            tech_indices = [technique_to_idx.get(t, 0) for t in techniques] if techniques else [0]
            soft_indices = [software_to_idx.get(s, 0) for s in software] if software else [0]
            capec_indices = [capec_to_idx.get(c, 0) for c in capec] if capec else [0]
            cwe_indices = [cwe_to_idx.get(c, 0) for c in cwe] if cwe else [0]
            cve_indices = [cve_to_idx.get(c, 0) for c in cves] if cves else [0]
            pulse_indices = [pulse_to_idx.get(p, 0) for p in pulse] if pulse else [0]
            
            # Convert to tensors
            tech_tensor = torch.tensor(tech_indices, dtype=torch.long)
            soft_tensor = torch.tensor(soft_indices, dtype=torch.long)
            capec_tensor = torch.tensor(capec_indices, dtype=torch.long)
            cwe_tensor = torch.tensor(cwe_indices, dtype=torch.long)
            cve_tensor = torch.tensor(cve_indices, dtype=torch.long)
            pulse_tensor = torch.tensor(pulse_indices, dtype=torch.long)
            
            embedding = query_model(
                tech_tensor.unsqueeze(0), 
                soft_tensor.unsqueeze(0),
                capec_tensor.unsqueeze(0),
                cwe_tensor.unsqueeze(0), 
                cve_tensor.unsqueeze(0),
                pulse_tensor.unsqueeze(0)
            )
            apt_embeddings.append(embedding.squeeze())
        
        apt_embeddings = torch.stack(apt_embeddings)
    
    return apt_embeddings

#######
#debuging

def debug_query_model_outputs(model, apt_embeddings, profiles, apt_ids):
    """Debug what the query model is producing"""
    print(f"\n🐛 QUERY MODEL DEBUG INFO")
    print("=" * 50)
    
    print(f"APT embeddings shape: {apt_embeddings.shape}")
    print(f"Embedding stats - Mean: {apt_embeddings.mean():.4f}, Std: {apt_embeddings.std():.4f}")
    print(f"Min: {apt_embeddings.min():.4f}, Max: {apt_embeddings.max():.4f}")
    
    # Check similarity matrix
    similarity = torch.matmul(apt_embeddings, apt_embeddings.t())
    print(f"\nSimilarity matrix stats:")
    print(f"Self-similarity mean: {torch.diag(similarity).mean():.4f}")
    print(f"Cross-similarity mean: {similarity[~torch.eye(len(apt_ids), dtype=bool)].mean():.4f}")

#############################################################################
#model saving
def save_trained_model(model, apt_embeddings, profiles, apt_ids, feature_mappings):
    """Save the trained model for future queries"""
    print(f"\n💾 SAVING TRAINED MODEL")
    print("=" * 50)
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'apt_embeddings': apt_embeddings,
        'profiles': profiles,
        'apt_ids': apt_ids,
        'feature_mappings': feature_mappings,
        'model_config': {
            'num_techniques': model.technique_embedding.num_embeddings,
            'num_software': model.software_embedding.num_embeddings,
            'num_capec': model.capec_embedding.num_embeddings,
            'num_cwe': model.cwe_embedding.num_embeddings,
            'num_cve': model.cve_embedding.num_embeddings,
            'num_pulse': model.pulse_embedding.num_embeddings,
            'hidden_dim': 128
        }
    }
    
    with open('trained_apt_model.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("✅ Model saved as 'trained_apt_model.pkl'")
    print(f"Saved: {len(apt_ids)} APT embeddings, {len(profiles)} profiles")
    print(f"Feature mappings: {len(feature_mappings[0])} techniques, {len(feature_mappings[1])} software, {len(feature_mappings[2])} CAPEC, {len(feature_mappings[3])} CWE, {len(feature_mappings[4])} CVEs, {len(feature_mappings[5])} Pulse")

#############################################################################
#############################################################################
#main
#############################################################################
#############################################################################
def main():
    print("Data import and preprocessing and profile creation for GNN")
    print("=" * 60)

    # Step 1: Load only training data
    df_encoded = load_training_data()
    
    # Step 2: Create APT profiles
    profiles, apt_ids = create_apt_profiles_simple(df_encoded)
    
    print(f"\n✅ READY FOR MODEL CREATION!")
    print(f"We have {len(apt_ids)} APT nodes for the model")

    # Step 3: Create proper feature mappings from profiles for all 6 features
    technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, cve_to_idx, pulse_to_idx = create_feature_mappings_from_profiles(profiles)
    feature_mappings = (technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, cve_to_idx, pulse_to_idx)

    ##################################
    # Create and train the query model with all 6 features
    model = APTGraphModel(
        num_techniques=len(technique_to_idx),
        num_software=len(software_to_idx), 
        num_capec=len(capec_to_idx),
        num_cwe=len(cwe_to_idx),
        num_cve=len(cve_to_idx),
        num_pulse=len(pulse_to_idx)
    )
    
    # Train the query model to create APT embeddings from features
    print("\n🧠 TRAINING QUERY MODEL...")
    apt_embeddings = proper_supervised_training(
        model, profiles, apt_ids, 
        technique_to_idx, software_to_idx, capec_to_idx,
        cwe_to_idx, cve_to_idx, pulse_to_idx, 
        epochs=200
    )
    print(f"\n✅ TRAINING COMPLETE!")

    # Debug the new model
    debug_query_model_outputs(model, apt_embeddings, profiles, apt_ids)
    
    ##################################
    # Save the model
    save_trained_model(model, apt_embeddings, profiles, apt_ids, feature_mappings) 

if __name__ == "__main__":
    main()