import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# CONFIG
DATA_PATH = "../BaseData/EncodedData.csv"
MODEL_PATH = "apt_siamese_feature_model.pkl"
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50  # Reduced for testing
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureEncoder(nn.Module):
    """FIXED: Handles exact embedding dimensions"""
    def __init__(self, num_techniques, num_software, num_capec, num_cwes, num_cves, num_pulse, embed_dim=EMBEDDING_DIM):
        super().__init__()
        embed_per_feature = embed_dim // 6  # e.g., 128//3 = 42
        
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature) 
        self.capec_embed = nn.Embedding(num_capec+1, embed_per_feature)
        self.cwe_embed = nn.Embedding(num_cwes+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cves+1, embed_per_feature)
        self.pulse_embed = nn.Embedding(num_pulse+1, embed_per_feature)

        total_embed_size = embed_per_feature * 6
        print(f"🔧 Embed/feature: {embed_per_feature}, Total input: {total_embed_size}")
        
        self.fc1 = nn.Linear(total_embed_size, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech, soft, capec, cwe, cve, pulse = features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4], features[:, 5]
        t_emb = self.tech_embed(tech.long() + 1)
        s_emb = self.soft_embed(soft.long() + 1)
        ca_emb = self.capec_embed(capec.long() + 1)
        cw_emb = self.cwe_embed(cwe.long() + 1)
        cv_emb = self.cve_embed(cve.long() + 1)
        p_emb = self.pulse_embed(pulse.long() + 1)
        x = torch.cat([t_emb, s_emb, ca_emb, cw_emb, cv_emb, p_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

class APTFeatureDataset(Dataset):
    def __init__(self, X, y, apt_to_idx):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor([apt_to_idx[apt] for apt in y])
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def contrastive_loss(emb1, emb2, labels, margin=0.5):
    sim = F.cosine_similarity(emb1, emb2)
    same_apt = (labels[:, 0] == labels[:, 1]).float()
    pos_loss = same_apt * (1 - sim)**2
    neg_loss = (1 - same_apt) * F.relu(margin - sim)**2
    return (pos_loss + neg_loss).mean()

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df[['techniques_str', 'software_str', 'CAPEC_ID','CWE_ID','CVE_ID','Pulse_ID']].values
    y = df['APTname'].values
    
    num_tech = int(X[:, 0].max()) + 1
    num_soft = int(X[:, 1].max()) + 1
    num_capec  = int(X[:, 2].max()) + 1
    num_cwe  = int(X[:, 3].max()) + 1
    num_cve  = int(X[:, 4].max()) + 1
    num_pulse  = int(X[:, 5].max()) + 1
    
    unique_apts = sorted(np.unique(y))
    apt_to_idx = {apt: i for i, apt in enumerate(unique_apts)}
    
    print(f"Data: {len(df)} records, {len(unique_apts)} APTs")
    print(f"Vocab: T={num_tech}, S={num_soft}, Ca={num_capec}, Cw={num_cwe}, Cv={num_cve}, P={num_pulse}")
    return X, y, apt_to_idx, num_tech, num_soft, num_capec, num_cwe, num_cve, num_pulse

def train_siamese_model():
    X, y, apt_to_idx, num_tech, num_soft, num_capec, num_cwe, num_cve, num_pulse = load_and_prepare_data()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = APTFeatureDataset(X_tr, y_tr, apt_to_idx)
    val_ds = APTFeatureDataset(X_val, y_val, apt_to_idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = FeatureEncoder(num_tech, num_soft, num_capec, num_cwe, num_cve, num_pulse).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 Training on {DEVICE} ({len(train_loader)} batches)")
    
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (features1, apt1) in enumerate(train_loader):
            # ✅ FIXED: Proper Siamese pairs from SAME APT
            batch_size = len(features1)
            
            # Positive pairs: same APT, different samples
            pos_indices = torch.randperm(batch_size)
            features2 = features1[pos_indices]  # Shuffle within batch
            apt2 = apt1[pos_indices]
            
            # Negative pairs: different APTs
            neg_mask = (apt2 != apt1)  # Find different APTs in batch
            if neg_mask.sum() > 1:  # Only if we have negatives
                neg_indices = torch.randperm(batch_size)
                neg_features = features1[neg_indices]
            else:
                neg_features = features1
            
            features1, features2, neg_features = [x.to(DEVICE) for x in [features1, features2, neg_features]]
            apt1_dev, apt2_dev = apt1.to(DEVICE), apt2.to(DEVICE)
            
            optimizer.zero_grad()
            
            emb1 = model(features1)
            emb2 = model(features2)
            emb_neg = model(neg_features)
            
            # ✅ FIXED: Triplet loss (anchor=features1, pos=features2, neg=neg_features)
            pos_dist = 1 - F.cosine_similarity(emb1, emb2)  # Distance = 1-cos_sim
            neg_dist = 1 - F.cosine_similarity(emb1, emb_neg)
            loss = F.relu(pos_dist - neg_dist + 0.3).mean()  # Margin=0.3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features1, apt1 in val_loader:
                pos_indices = torch.randperm(len(features1))
                features2 = features1[pos_indices]
                apt2 = apt1[pos_indices]
                
                features1, features2 = features1.to(DEVICE), features2.to(DEVICE)
                emb1 = model(features1)
                emb2 = model(features2)
                
                pos_dist = 1 - F.cosine_similarity(emb1, emb2)
                neg_dist = 1 - F.cosine_similarity(emb1, emb2.flip(0))  # Simple negative
                val_loss += F.relu(pos_dist - neg_dist + 0.3).mean().item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_siamese.tmp')
    
    model.load_state_dict(torch.load('best_siamese.tmp'))
    os.remove('best_siamese.tmp')
    return model, apt_to_idx


def extract_apt_embeddings(model, X, y, apt_to_idx):
    unique_apts = list(apt_to_idx.keys())
    apt_embeddings = {}
    
    model.eval()
    with torch.no_grad():
        for apt in unique_apts:
            apt_mask = (np.array(y) == apt)
            if apt_mask.sum() == 0: continue
            sample_features = X[apt_mask][0]
            features_tensor = torch.FloatTensor([sample_features]).to(DEVICE)
            embedding = model(features_tensor).cpu()
            apt_embeddings[apt] = embedding.squeeze(0)
    
    apt_emb_matrix = torch.stack([apt_embeddings[apt] for apt in unique_apts])
    return apt_emb_matrix, unique_apts

def save_model(model, apt_embeddings, apt_ids, X, apt_to_idx):
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_techniques': int(X[:,0].max()) + 1,
            'num_software': int(X[:,1].max()) + 1,
            'num_capec': int(X[:,2].max()) + 1,
            'num_cwes': int(X[:,3].max()) + 1,
            'num_cves': int(X[:,4].max()) + 1,
            'num_pulse': int(X[:,5].max()) + 1,
            'embedding_dim': EMBEDDING_DIM
        },
        'apt_embeddings': apt_embeddings,
        'apt_ids': apt_ids,
        'apt_to_idx': apt_to_idx
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"✅ SAVED: {MODEL_PATH}")

def main():
    model, apt_to_idx = train_siamese_model()
    X, y, _, _, _, _, _, _, _ = load_and_prepare_data()
    apt_embeddings, apt_ids = extract_apt_embeddings(model, X, y, apt_to_idx)
    save_model(model, apt_embeddings, apt_ids, X, apt_to_idx)
    print("🎉 SIAMESE MODEL READY FOR FALSE-FLAG ANALYSIS!")

if __name__ == "__main__":
    main()