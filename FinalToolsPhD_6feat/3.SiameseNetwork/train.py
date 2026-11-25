# ============================
# 1. IMPORTS (same as before)
# ============================
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

# ============================
# 2. CONFIGURATION (add SAMPLES_PER_APT)
# ============================
class SiameseConfig:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "6feat_apt_siamese_feature_model.pkl"
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42
    EPOCHS = 200
    NUM_FEATURES = 6
    EMBEDDING_DIM = 256
    TEMP_MODEL_PATH = "best_siamese.tmp"
    BATCH_SIZE = 256          # not used in APT-level training, kept for compatibility
    DROPOUT_RATE = 0.3
    MARGIN = 0.7
    GRADIENT_CLIP = 1.0
    TEST_SIZE = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLES_PER_APT = 30       # number of rows to sample per subset during training

# ============================
# 3. DATA LOADING (unchanged)
# ============================
class SiameseDataLoader:
    @staticmethod
    def load_raw_data(data_path=SiameseConfig.DATA_PATH):
        df = pd.read_csv(data_path)
        return df
    
    @staticmethod
    def prepare_features(df):
        feature_columns = ['techniques_str', 'software_str', 'CAPEC_ID', 
                          'CWE_ID', 'CVE_ID', 'Pulse_ID']
        X = df[feature_columns].values
        y = df['APTname'].values
        return X, y
    
    @staticmethod
    def get_vocabulary_sizes(X):
        vocab_sizes = {}
        feature_names = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
        for i, name in enumerate(feature_names):
            vocab_sizes[name] = int(X[:, i].max()) + 1
        return vocab_sizes
    
    @staticmethod
    def create_apt_mapping(y):
        unique_apts = sorted(np.unique(y))
        apt_to_idx = {apt: i for i, apt in enumerate(unique_apts)}
        return apt_to_idx, unique_apts
    
    @staticmethod
    def load_and_prepare():
        df = SiameseDataLoader.load_raw_data()
        X, y = SiameseDataLoader.prepare_features(df)
        vocab_sizes = SiameseDataLoader.get_vocabulary_sizes(X)
        apt_to_idx, unique_apts = SiameseDataLoader.create_apt_mapping(y)
        print(f"Data Statistics:")
        print(f"  Records: {len(df)}")
        print(f"  Unique APTs: {len(unique_apts)}")
        print(f"  Vocabulary sizes: {vocab_sizes}")
        return X, y, apt_to_idx, vocab_sizes, unique_apts

# ============================
# 4. NEW DATASET – returns all rows for a given APT
# ============================
class APTAggregateDataset(Dataset):
    """Returns all rows for a single APT when indexed by APT index."""
    def __init__(self, X, y, apt_to_idx):
        """
        apt_to_idx: mapping from APT id to index (0..N-1) for the APTs in this split.
        """
        self.apt_to_idx = apt_to_idx
        self.apt_ids = sorted(apt_to_idx.keys())
        # Group rows by APT id
        self.apt_rows = {}
        for apt_id in self.apt_ids:
            mask = (y == apt_id)
            self.apt_rows[apt_id] = X[mask]
    
    def __len__(self):
        return len(self.apt_ids)
    
    def __getitem__(self, idx):
        apt_id = self.apt_ids[idx]
        rows = self.apt_rows[apt_id]
        return torch.FloatTensor(rows), apt_id

# ============================
# 5. MODEL DEFINITION (unchanged)
# ============================
class SiameseFeatureEncoder(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=SiameseConfig.EMBEDDING_DIM):
        super().__init__()
        embed_per_feature = embed_dim // SiameseConfig.NUM_FEATURES
        self.tech_embed = nn.Embedding(vocab_sizes['techniques'] + 1, embed_per_feature)
        self.soft_embed = nn.Embedding(vocab_sizes['software'] + 1, embed_per_feature)
        self.capec_embed = nn.Embedding(vocab_sizes['capec'] + 1, embed_per_feature)
        self.cwe_embed = nn.Embedding(vocab_sizes['cwe'] + 1, embed_per_feature)
        self.cve_embed = nn.Embedding(vocab_sizes['cve'] + 1, embed_per_feature)
        self.pulse_embed = nn.Embedding(vocab_sizes['pulse'] + 1, embed_per_feature)
        total_embed_size = embed_per_feature * SiameseConfig.NUM_FEATURES
        self.fc1 = nn.Linear(total_embed_size, SiameseConfig.HIDDEN_DIM)
        self.fc2 = nn.Linear(SiameseConfig.HIDDEN_DIM, embed_dim)
        self.dropout = nn.Dropout(SiameseConfig.DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech = features[:, 0].long() + 1
        soft = features[:, 1].long() + 1
        capec = features[:, 2].long() + 1
        cwe = features[:, 3].long() + 1
        cve = features[:, 4].long() + 1
        pulse = features[:, 5].long() + 1
        t_emb = self.tech_embed(tech)
        s_emb = self.soft_embed(soft)
        ca_emb = self.capec_embed(capec)
        cw_emb = self.cwe_embed(cwe)
        cv_emb = self.cve_embed(cve)
        p_emb = self.pulse_embed(pulse)
        x = torch.cat([t_emb, s_emb, ca_emb, cw_emb, cv_emb, p_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

# ============================
# 6. LOSS FUNCTIONS (unchanged)
# ============================
class SiameseLoss:
    @staticmethod
    def triplet_loss(anchor, positive, negative, margin=SiameseConfig.MARGIN):
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

# ============================
# 7. NEW TRAINER with APT‑level averaging
# ============================
class SiameseTrainer:
    def __init__(self, model, device=SiameseConfig.DEVICE):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=SiameseConfig.LEARNING_RATE)
        self.best_val_loss = float('inf')
    
    def sample_subset(self, rows, n_samples=SiameseConfig.SAMPLES_PER_APT):
        """Randomly sample n_samples rows from the given rows (with replacement if needed)."""
        n_rows = len(rows)
        if n_rows >= n_samples:
            indices = np.random.choice(n_rows, n_samples, replace=False)
        else:
            indices = np.random.choice(n_rows, n_samples, replace=True)
        return rows[indices]
    
    def compute_apt_embedding(self, rows):
        """Compute embedding for a subset of rows by averaging their embeddings."""
        tensor = torch.FloatTensor(rows).to(self.device)
        embeddings = self.model(tensor)          # (n_samples, embed_dim)
        return embeddings.mean(dim=0)             # averaged embedding
    
    def train_epoch(self, train_dataset, epoch):
        self.model.train()
        total_loss = 0
        apt_ids = train_dataset.apt_ids
        num_apts = len(apt_ids)
        
        for anchor_idx, anchor_apt in enumerate(apt_ids):
            anchor_rows = train_dataset.apt_rows[anchor_apt]
            
            # Sample two different subsets for anchor and positive
            anchor_subset = self.sample_subset(anchor_rows)
            pos_subset = self.sample_subset(anchor_rows)
            
            # Choose a negative APT
            neg_idx = anchor_idx
            while neg_idx == anchor_idx:
                neg_idx = np.random.randint(num_apts)
            neg_apt = apt_ids[neg_idx]
            neg_rows = train_dataset.apt_rows[neg_apt]
            neg_subset = self.sample_subset(neg_rows)
            
            # Compute averaged embeddings
            anchor_emb = self.compute_apt_embedding(anchor_subset)
            pos_emb = self.compute_apt_embedding(pos_subset)
            neg_emb = self.compute_apt_embedding(neg_subset)
            
            # Triplet loss
            loss = SiameseLoss.triplet_loss(
                anchor_emb.unsqueeze(0),
                pos_emb.unsqueeze(0),
                neg_emb.unsqueeze(0)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), SiameseConfig.GRADIENT_CLIP)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if anchor_idx % 20 == 0:
                print(f"  Epoch {epoch+1}, APT {anchor_idx}/{num_apts}, Loss: {loss.item():.4f}")
        
        return total_loss / num_apts
    
    def validate(self, val_dataset):
        self.model.eval()
        total_loss = 0
        apt_ids = val_dataset.apt_ids
        num_apts = len(apt_ids)
        
        with torch.no_grad():
            for anchor_idx, anchor_apt in enumerate(apt_ids):
                anchor_rows = val_dataset.apt_rows[anchor_apt]
                anchor_subset = self.sample_subset(anchor_rows)
                pos_subset = self.sample_subset(anchor_rows)
                
                neg_idx = anchor_idx
                while neg_idx == anchor_idx:
                    neg_idx = np.random.randint(num_apts)
                neg_apt = apt_ids[neg_idx]
                neg_rows = val_dataset.apt_rows[neg_apt]
                neg_subset = self.sample_subset(neg_rows)
                
                anchor_emb = self.compute_apt_embedding(anchor_subset)
                pos_emb = self.compute_apt_embedding(pos_subset)
                neg_emb = self.compute_apt_embedding(neg_subset)
                
                loss = SiameseLoss.triplet_loss(
                    anchor_emb.unsqueeze(0),
                    pos_emb.unsqueeze(0),
                    neg_emb.unsqueeze(0)
                )
                total_loss += loss.item()
        
        return total_loss / num_apts
    
    def train(self, train_dataset, val_dataset, epochs=SiameseConfig.EPOCHS):
        print(f"\nTraining Siamese Network with APT‑level averaging on {self.device}")
        print(f"Training APTs: {len(train_dataset)}")
        print(f"Validation APTs: {len(val_dataset)}")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataset, epoch)
            val_loss = self.validate(val_dataset)
            
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), SiameseConfig.TEMP_MODEL_PATH)
                print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        self.model.load_state_dict(torch.load(SiameseConfig.TEMP_MODEL_PATH))
        os.remove(SiameseConfig.TEMP_MODEL_PATH)
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.4f}")
        return self.model

# ============================
# 8. NEW EMBEDDING EXTRACTOR – averages all rows per APT
# ============================
class EmbeddingExtractor:
    def __init__(self, model, device=SiameseConfig.DEVICE):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_apt_embeddings_avg(self, X, y, apt_to_idx):
        """Extract embeddings for each APT by averaging over all rows."""
        unique_apts = list(apt_to_idx.keys())
        apt_embeddings = []
        
        with torch.no_grad():
            for apt in unique_apts:
                mask = (y == apt)
                apt_rows = X[mask]
                if len(apt_rows) == 0:
                    # fallback (should not happen)
                    apt_rows = X[mask][:1]
                rows_tensor = torch.FloatTensor(apt_rows).to(self.device)
                embs = self.model(rows_tensor)          # (num_rows, embed_dim)
                avg_emb = embs.mean(dim=0).cpu()        # averaged embedding
                apt_embeddings.append(avg_emb)
        
        apt_emb_matrix = torch.stack(apt_embeddings)
        return apt_emb_matrix, unique_apts

# ============================
# 9. MODEL SAVING (unchanged)
# ============================
class ModelSaver:
    @staticmethod
    def save_model(model, apt_embeddings, apt_ids, X, apt_to_idx, 
                   filename=SiameseConfig.MODEL_PATH):
        vocab_sizes = SiameseDataLoader.get_vocabulary_sizes(X)
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_techniques': vocab_sizes['techniques'],
                'num_software': vocab_sizes['software'],
                'num_capec': vocab_sizes['capec'],
                'num_cwe': vocab_sizes['cwe'],
                'num_cve': vocab_sizes['cve'],
                'num_pulse': vocab_sizes['pulse'],
                'embedding_dim': SiameseConfig.EMBEDDING_DIM,
                'hidden_dim': SiameseConfig.HIDDEN_DIM,
                'dropout_rate': SiameseConfig.DROPOUT_RATE
            },
            'apt_embeddings': apt_embeddings,
            'apt_ids': apt_ids,
            'apt_to_idx': apt_to_idx,
            'training_config': {
                'batch_size': SiameseConfig.BATCH_SIZE,
                'epochs': SiameseConfig.EPOCHS,
                'learning_rate': SiameseConfig.LEARNING_RATE,
                'margin': SiameseConfig.MARGIN,
                'samples_per_apt': SiameseConfig.SAMPLES_PER_APT
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nModel saved to '{filename}'")
        print(f"  APT embeddings: {apt_embeddings.shape}")
        print(f"  Number of APTs: {len(apt_ids)}")
        return save_data

# ============================
# 10. MAIN EXECUTION
# ============================
def main():
    print("SIAMESE NETWORK FOR APT SIMILARITY (APT‑LEVEL AVERAGING)")
    print("=" * 60)
    
    # 1. Load data
    X, y, apt_to_idx, vocab_sizes, unique_apts = SiameseDataLoader.load_and_prepare()
    
    # 2. Split APTs into train and validation
    apt_ids_list = list(apt_to_idx.keys())
    train_apts, val_apts = train_test_split(
        apt_ids_list, test_size=SiameseConfig.TEST_SIZE, random_state=SiameseConfig.RANDOM_STATE
    )
    
    # Create mappings for the split
    train_apt_to_idx = {apt: i for i, apt in enumerate(train_apts)}
    val_apt_to_idx = {apt: i for i, apt in enumerate(val_apts)}
    
    # 3. Create datasets (each returns all rows for an APT)
    train_dataset = APTAggregateDataset(X, y, train_apt_to_idx)
    val_dataset = APTAggregateDataset(X, y, val_apt_to_idx)
    
    print(f"\nTrain APTs: {len(train_dataset)}")
    print(f"Val APTs: {len(val_dataset)}")
    
    # 4. Initialize model
    model = SiameseFeatureEncoder(vocab_sizes).to(SiameseConfig.DEVICE)
    
    # 5. Train
    trainer = SiameseTrainer(model)
    trained_model = trainer.train(train_dataset, val_dataset, epochs=SiameseConfig.EPOCHS)
    
    # 6. Extract averaged embeddings for all APTs (using all rows)
    extractor = EmbeddingExtractor(trained_model)
    apt_embeddings, apt_ids = extractor.extract_apt_embeddings_avg(X, y, apt_to_idx)
    
    # 7. Save model with averaged embeddings
    ModelSaver.save_model(trained_model, apt_embeddings, apt_ids, X, apt_to_idx)
    
    print("\nTRAINING SUMMARY")
    print("=" * 60)
    print(f"• Model: Siamese Feature Encoder with APT‑level averaging")
    print(f"• Embedding dimension: {SiameseConfig.EMBEDDING_DIM}")
    print(f"• Training epochs: {SiameseConfig.EPOCHS}")
    print(f"• Samples per subset: {SiameseConfig.SAMPLES_PER_APT}")
    print(f"• Unique APTs: {len(unique_apts)}")
    print(f"• Saved to: {SiameseConfig.MODEL_PATH}")

if __name__ == "__main__":
    main()