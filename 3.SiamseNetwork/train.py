import time
import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        print("\n" + "="*50)
        print("RESOURCE USAGE REPORT")
        print("="*50)
        
        # Time
        print(f"Execution Time: {end_time - self.start_time:.2f} seconds")
        
        # Memory
        print(f"Memory Usage: {end_memory - self.start_memory:.2f} MB")
        print(f"Current Memory: {end_memory:.2f} MB")
        
        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # System-wide CPU
        sys_cpu = psutil.cpu_percent(interval=0.1)
        print(f"System CPU: {sys_cpu}%")
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU {gpu.id} ({gpu.name}):")
                print(f"  Utilization: {gpu.load*100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        except:
            print("\nGPU: Not available or not detected")

# Usage
monitor = ResourceMonitor()
monitor.start()



# ============================
# 1. IMPORTS
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
# 2. CONFIGURATION
# ============================
class SiameseConfig:
    """Configuration for Siamese Network"""
#variable shared by all models:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "apt_siamese_feature_model.pkl"


#variables share by some models:
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42
    EPOCHS = 50


#variables unique to this model:
    NUM_FEATURES = 6  # techniques, software, capec, cwe, cve, pulse
    
    EMBEDDING_DIM = 128
    TEMP_MODEL_PATH = "best_siamese.tmp"
    BATCH_SIZE = 256
    DROPOUT_RATE = 0.3
    MARGIN = 0.3  # Triplet loss margin
    GRADIENT_CLIP = 1.0
    TEST_SIZE = 0.2

    GRADIENT_CLIP = 1.0
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# 3. DATA LOADING & PREPROCESSING
# ============================
class SiameseDataLoader:
    """Handles data loading and preparation for Siamese Network"""
    
    @staticmethod
    def load_raw_data(data_path=SiameseConfig.DATA_PATH):
        """Load and prepare raw data"""
        print("Loading encoded data...")
        df = pd.read_csv(data_path)
        return df
    
    @staticmethod
    def prepare_features(df):
        """Extract features and target from dataframe"""
        feature_columns = ['techniques_str', 'software_str', 'CAPEC_ID', 
                          'CWE_ID', 'CVE_ID', 'Pulse_ID']
        X = df[feature_columns].values
        y = df['APTname'].values
        return X, y
    
    @staticmethod
    def get_vocabulary_sizes(X):
        """Calculate vocabulary sizes for each feature type"""
        vocab_sizes = {}
        feature_names = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
        
        for i, name in enumerate(feature_names):
            vocab_sizes[name] = int(X[:, i].max()) + 1
        
        return vocab_sizes
    
    @staticmethod
    def create_apt_mapping(y):
        """Create mapping from APT IDs to indices"""
        unique_apts = sorted(np.unique(y))
        apt_to_idx = {apt: i for i, apt in enumerate(unique_apts)}
        return apt_to_idx, unique_apts
    
    @staticmethod
    def load_and_prepare():
        """Complete data loading and preparation pipeline"""
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
# 4. DATASET DEFINITION
# ============================
class APTFeatureDataset(Dataset):
    """Dataset for APT features"""
    
    def __init__(self, X, y, apt_to_idx):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor([apt_to_idx[apt] for apt in y])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================
# 5. MODEL DEFINITION
# ============================
class SiameseFeatureEncoder(nn.Module):
    """Siamese network feature encoder"""
    
    def __init__(self, vocab_sizes, embed_dim=SiameseConfig.EMBEDDING_DIM):
        super().__init__()
        
        # Calculate embedding dimensions per feature
        embed_per_feature = embed_dim // SiameseConfig.NUM_FEATURES
        
        # Embedding layers for each feature type
        self.tech_embed = nn.Embedding(vocab_sizes['techniques'] + 1, embed_per_feature)
        self.soft_embed = nn.Embedding(vocab_sizes['software'] + 1, embed_per_feature)
        self.capec_embed = nn.Embedding(vocab_sizes['capec'] + 1, embed_per_feature)
        self.cwe_embed = nn.Embedding(vocab_sizes['cwe'] + 1, embed_per_feature)
        self.cve_embed = nn.Embedding(vocab_sizes['cve'] + 1, embed_per_feature)
        self.pulse_embed = nn.Embedding(vocab_sizes['pulse'] + 1, embed_per_feature)
        
        # Calculate total embedding size
        total_embed_size = embed_per_feature * SiameseConfig.NUM_FEATURES
        
        print(f"\nModel Architecture:")
        print(f"  Embedding per feature: {embed_per_feature}")
        print(f"  Total embedding size: {total_embed_size}")
        print(f"  Hidden dimension: {SiameseConfig.HIDDEN_DIM}")
        print(f"  Output dimension: {embed_dim}")
        
        # Fully connected layers
        self.fc1 = nn.Linear(total_embed_size, SiameseConfig.HIDDEN_DIM)
        self.fc2 = nn.Linear(SiameseConfig.HIDDEN_DIM, embed_dim)
        self.dropout = nn.Dropout(SiameseConfig.DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        """Forward pass through the encoder"""
        # Extract features
        tech = features[:, 0].long() + 1
        soft = features[:, 1].long() + 1
        capec = features[:, 2].long() + 1
        cwe = features[:, 3].long() + 1
        cve = features[:, 4].long() + 1
        pulse = features[:, 5].long() + 1
        
        # Get embeddings
        t_emb = self.tech_embed(tech)
        s_emb = self.soft_embed(soft)
        ca_emb = self.capec_embed(capec)
        cw_emb = self.cwe_embed(cwe)
        cv_emb = self.cve_embed(cve)
        p_emb = self.pulse_embed(pulse)
        
        # Concatenate embeddings
        x = torch.cat([t_emb, s_emb, ca_emb, cw_emb, cv_emb, p_emb], dim=-1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        
        # Normalize for cosine similarity
        return F.normalize(x, p=2, dim=-1)

# ============================
# 6. LOSS FUNCTIONS
# ============================
class SiameseLoss:
    """Loss functions for Siamese network"""
    
    @staticmethod
    def triplet_loss(anchor, positive, negative, margin=SiameseConfig.MARGIN):
        """Calculate triplet loss"""
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss
    
    @staticmethod
    def contrastive_loss(emb1, emb2, labels, margin=0.5):
        """Calculate contrastive loss"""
        sim = F.cosine_similarity(emb1, emb2)
        same_apt = (labels[:, 0] == labels[:, 1]).float()
        pos_loss = same_apt * (1 - sim)**2
        neg_loss = (1 - same_apt) * F.relu(margin - sim)**2
        return (pos_loss + neg_loss).mean()

# ============================
# 7. TRAINING FUNCTION
# ============================
class SiameseTrainer:
    """Handles Siamese network training"""
    
    def __init__(self, model, device=SiameseConfig.DEVICE):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=SiameseConfig.LEARNING_RATE)
        self.best_val_loss = float('inf')
    
    def create_triplet_batch(self, features, apt_labels):
        """Create triplet batches for training"""
        batch_size = len(features)
        
        # Positive pairs: same APT, different samples
        pos_indices = torch.randperm(batch_size)
        positive_features = features[pos_indices]
        positive_labels = apt_labels[pos_indices]
        
        # Negative pairs: different APTs
        neg_mask = (positive_labels != apt_labels)
        if neg_mask.sum() > 1:
            neg_indices = torch.randperm(batch_size)
            negative_features = features[neg_indices]
        else:
            negative_features = features
        
        return positive_features, positive_labels, negative_features
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        
        for batch_idx, (features, apt_labels) in enumerate(train_loader):
            # Create triplet batch
            pos_features, pos_labels, neg_features = self.create_triplet_batch(features, apt_labels)
            
            # Move to device
            anchor = features.to(self.device)
            positive = pos_features.to(self.device)
            negative = neg_features.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            # Calculate loss
            loss = SiameseLoss.triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), SiameseConfig.GRADIENT_CLIP)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return train_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, apt_labels in val_loader:
                # Create simple validation pairs
                pos_indices = torch.randperm(len(features))
                positive_features = features[pos_indices]
                positive_labels = apt_labels[pos_indices]
                
                # Move to device
                anchor = features.to(self.device)
                positive = positive_features.to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(positive.flip(0))  # Simple negative
                
                # Calculate loss
                loss = SiameseLoss.triplet_loss(anchor_emb, positive_emb, negative_emb)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=SiameseConfig.EPOCHS):
        """Complete training pipeline"""
        print(f"\nTraining Siamese Network on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Training
            avg_train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            avg_val_loss = self.validate(val_loader)
            
            # Log epoch results
            print(f"Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), SiameseConfig.TEMP_MODEL_PATH)
                print(f"  ✓ Saved best model (Val Loss: {avg_val_loss:.4f})")
        
        # Load best model
        self.model.load_state_dict(torch.load(SiameseConfig.TEMP_MODEL_PATH))
        os.remove(SiameseConfig.TEMP_MODEL_PATH)
        
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.4f}")
        return self.model

# ============================
# 8. EMBEDDING EXTRACTION
# ============================
class EmbeddingExtractor:
    """Extracts APT embeddings from trained model"""
    
    def __init__(self, model, device=SiameseConfig.DEVICE):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_apt_embeddings(self, X, y, apt_to_idx):
        """Extract embeddings for each APT"""
        unique_apts = list(apt_to_idx.keys())
        apt_embeddings = {}
        
        with torch.no_grad():
            for apt in unique_apts:
                apt_mask = (np.array(y) == apt)
                if apt_mask.sum() == 0:
                    continue
                
                # Use first sample of each APT
                sample_features = X[apt_mask][0]
                features_tensor = torch.FloatTensor([sample_features]).to(self.device)
                embedding = self.model(features_tensor).cpu()
                apt_embeddings[apt] = embedding.squeeze(0)
        
        # Create embedding matrix
        apt_emb_matrix = torch.stack([apt_embeddings[apt] for apt in unique_apts])
        return apt_emb_matrix, unique_apts

# ============================
# 9. MODEL SAVING
# ============================
class ModelSaver:
    """Handles model saving and loading"""
    
    @staticmethod
    def save_model(model, apt_embeddings, apt_ids, X, apt_to_idx, 
                   filename=SiameseConfig.MODEL_PATH):
        """Save complete model with embeddings"""
        
        # Get vocabulary sizes
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
                'margin': SiameseConfig.MARGIN
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
    """Execute the Siamese Network training pipeline"""
    print("SIAMESE NETWORK FOR APT SIMILARITY")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. DATA LOADING")
    X, y, apt_to_idx, vocab_sizes, unique_apts = SiameseDataLoader.load_and_prepare()
    
    # 2. Create datasets
    print("\n2. DATASET CREATION")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=SiameseConfig.TEST_SIZE, 
        random_state=SiameseConfig.RANDOM_STATE
    )
    
    train_dataset = APTFeatureDataset(X_train, y_train, apt_to_idx)
    val_dataset = APTFeatureDataset(X_val, y_val, apt_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=SiameseConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=SiameseConfig.BATCH_SIZE)
    
    print(f"  Training set: {len(train_dataset)} samples")
    print(f"  Validation set: {len(val_dataset)} samples")
    
    # 3. Initialize model
    print("\n3. MODEL INITIALIZATION")
    model = SiameseFeatureEncoder(vocab_sizes).to(SiameseConfig.DEVICE)
    
    # 4. Train model
    print("\n4. MODEL TRAINING")
    trainer = SiameseTrainer(model)
    trained_model = trainer.train(train_loader, val_loader)
    
    # 5. Extract embeddings
    print("\n5. EMBEDDING EXTRACTION")
    extractor = EmbeddingExtractor(trained_model)
    apt_embeddings, apt_ids = extractor.extract_apt_embeddings(X, y, apt_to_idx)
    
    print(f"  Extracted embeddings for {len(apt_ids)} APTs")
    print(f"  Embedding shape: {apt_embeddings.shape}")
    
    # 6. Save model
    print("\n6. MODEL SAVING")
    save_data = ModelSaver.save_model(trained_model, apt_embeddings, apt_ids, X, apt_to_idx)
    
    print("\nTRAINING SUMMARY")
    print("=" * 60)
    print(f"• Model: Siamese Feature Encoder")
    print(f"• Embedding dimension: {SiameseConfig.EMBEDDING_DIM}")
    print(f"• Training epochs: {SiameseConfig.EPOCHS}")
    print(f"• Batch size: {SiameseConfig.BATCH_SIZE}")
    print(f"• Unique APTs: {len(unique_apts)}")
    print(f"• Saved to: {SiameseConfig.MODEL_PATH}")
    
    return save_data

if __name__ == "__main__":
    saved_data = main()



monitor.stop()