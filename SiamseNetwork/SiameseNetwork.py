# import pandas as pd
# import numpy as np
# from collections import defaultdict

# pwd = "./DataForGNN/"
# # 1. Import CSV file as DataFrame
# # Replace 'your_file.csv' with your actual file path
# df = pd.read_csv(pwd+'data_for_GNN.csv')
# # df = pd.read_csv(pwd+'comparison_Encodedtest_df.csv')

# # 2. Print the first 5 rows
# print("First 5 rows:")
# print(df.head(5))
# print("\n" + "="*50 + "\n")  # Separator line

# # 3. Print 5 random rows
# print("5 random rows:")
# print(df.sample(5))
# print("\n" + "="*50 + "\n")  # Separator line

# # 4. Print the last 5 rows
# print("Last 5 rows:")
# print(df.tail(5))

###
#data structure work
###

import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

pwd = "./DataForSiamese/"
# Load both files
df_encoded = pd.read_csv(pwd+'data_for_Siamese.csv')  # The file with encoded values
df_mapping = pd.read_csv(pwd+'comparison_Encodedtest_df.csv')  # The mapping file

# Step 1.1: Create a mapping dictionary for APT names
apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))

# Step 1.2: Create comprehensive profiles using actual APT names as keys
def create_apt_profiles_with_names(df_encoded, name_mapping):
    profiles = {}
    
    for _, row in df_encoded.iterrows():
        apt_encoded = row['APTname']
        apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
        
        # Initialize the profile if it doesn't exist
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
        
        # Add features to the profile
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

apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)

# Convert sets to lists and add counts
for apt_name in apt_profiles:
    profile = apt_profiles[apt_name]
    for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
        profile[key] = list(profile[key])
        profile[f'{key}_count'] = len(profile[key])

# Create mapping dictionaries
apt_ids = list(apt_profiles.keys())
feature_types = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']

# Get all unique features across all APTs
all_features = {}
for feature_type in feature_types:
    all_features[feature_type] = set()
    for apt_name in apt_ids:
        all_features[feature_type].update(apt_profiles[apt_name][feature_type])
    all_features[feature_type] = sorted(all_features[feature_type])

# Create binary feature vectors for each APT
def create_binary_feature_vector(apt_profile, all_features):
    vector = []
    for feature_type in feature_types:
        for feature in all_features[feature_type]:
            vector.append(1 if feature in apt_profile[feature_type] else 0)
    return np.array(vector)

# Create feature vectors for all APTs
apt_feature_vectors = {}
for apt_name in apt_ids:
    apt_feature_vectors[apt_name] = create_binary_feature_vector(
        apt_profiles[apt_name], all_features
    )

# Convert to numpy array for training
X = np.array([apt_feature_vectors[apt_name] for apt_name in apt_ids])
y = np.arange(len(apt_ids))  # Each APT has its own ID

###
# Improved Siamese Network Implementation
###

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        # Simplified architecture to prevent overfitting
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)  # Smaller final embedding
        )
        
    def forward(self, x):
        return self.embedding_net(x)
    
    def get_embedding(self, x):
        return self.forward(x)

# Improved contrastive loss with smaller margin
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):  # Reduced margin
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Alternative: Triplet Loss (often works better than contrastive)
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)

# Prepare data for training
from torch.utils.data import Dataset, DataLoader

class APTDataset(Dataset):
    def __init__(self, feature_vectors, apt_ids):
        self.feature_vectors = feature_vectors
        self.apt_ids = apt_ids
        self.num_apts = len(apt_ids)
        
    def __len__(self):
        return self.num_apts * 200  # More pairs for better training
        
    def __getitem__(self, idx):
        # Randomly select two APTs
        apt1_idx = np.random.randint(0, self.num_apts)
        apt2_idx = np.random.randint(0, self.num_apts)
        
        # Get feature vectors
        x1 = self.feature_vectors[self.apt_ids[apt1_idx]]
        x2 = self.feature_vectors[self.apt_ids[apt2_idx]]
        
        # Label: 0 if same APT, 1 if different
        label = 0 if apt1_idx == apt2_idx else 1
        
        return (torch.FloatTensor(x1), 
                torch.FloatTensor(x2), 
                torch.FloatTensor([label]))

# Create dataset and dataloader
dataset = APTDataset(apt_feature_vectors, apt_ids)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer with regularization
input_dim = X.shape[1]
model = SiameseNetwork(input_dim)
criterion = ContrastiveLoss(margin=0.5)  # Using improved contrastive loss
# criterion = TripletLoss(margin=1.0)  # Alternative: uncomment to use triplet loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization

# Enhanced training loop with monitoring
def train(model, dataloader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        embedding_stds = []
        
        for batch_idx, (x1, x2, label) in enumerate(dataloader):
            optimizer.zero_grad()
            
            output1 = model(x1)
            output2 = model(x2)
            
            # Monitor embedding diversity
            embedding_std = output1.std().item()
            embedding_stds.append(embedding_std)
            
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_embedding_std = np.mean(embedding_stds)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Embedding Std: {avg_embedding_std:.4f}')

# Train the model
print("Training Siamese Network with improved configuration...")
train(model, dataloader, criterion, optimizer, num_epochs=100)

# Get embeddings for all APTs
model.eval()
with torch.no_grad():
    apt_embeddings = {}
    for apt_name in apt_ids:
        features = torch.FloatTensor(apt_feature_vectors[apt_name]).unsqueeze(0)
        embedding = model.get_embedding(features).squeeze().numpy()
        apt_embeddings[apt_name] = embedding

# Convert to matrix for similarity calculation
embedding_matrix = np.array([apt_embeddings[apt_name] for apt_name in apt_ids])

# Diagnostic check for embedding quality
print("\nEmbedding Quality Diagnostics:")
print(f"Embedding mean: {embedding_matrix.mean():.6f}")
print(f"Embedding std: {embedding_matrix.std():.6f}")
print(f"Embedding range: [{embedding_matrix.min():.6f}, {embedding_matrix.max():.6f}]")

# Check if embeddings are diverse enough
if embedding_matrix.std() < 0.1:
    print("WARNING: Embeddings may still be too similar. Consider:")
    print("1. Further reducing network complexity")
    print("2. Trying triplet loss instead")
    print("3. Increasing regularization")

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embedding_matrix)

# Create similarity DataFrame
apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=apt_names,
    columns=apt_names
)

###
# Cross Matrix Reference Graph (Similarity Heatmap)
###

plt.figure(figsize=(12,10))
sns.heatmap(similarity_matrix, 
            xticklabels=apt_names,
            yticklabels=apt_names,
            cmap='viridis')
plt.title("APT Group Similarity Matrix (Improved Siamese Network)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

###
# Evidence Processing and Similarity Calculation
###

# Evidence for Lazarus Group
evidence = {
    'techniques_str': [
        'T0865', 'T1001.003', 'T1005', 'T1008', 'T1010', 'T1012', 'T1016', 
        'T1021.001', 'T1021.002', 'T1021.004', 'T1027.002', 'T1027.007', 
        'T1027.013', 'T1033', 'T1036.003', 'T1036.004', 'T1036.005', 
        'T1036.008', 'T1041', 'T1046', 'T1047', 'T1048.003', 'T1049', 
        'T1053.005', 'T1055.001', 'T1056.001', 'T1057', 'T1059.001', 
        'T1059.003', 'T1059.005', 'T1070', 'T1070.003', 'T1070.004', 
        'T1070.006', 'T1071.001', 'T1074.001', 'T1078', 'T1082', 'T1083', 
        'T1087.002', 'T1090.001', 'T1090.002', 'T1098', 'T1102.002', 
        'T1104', 'T1105', 'T1106', 'T1110.003', 'T1124', 'T1132.001', 
        'T1134.002', 'T1140', 'T1189', 'T1202', 'T1203', 'T1204.001', 
        'T1204.002', 'T1218', 'T1218.005', 'T1218.010', 'T1218.011', 
        'T1220', 'T1221', 'T1485', 'T1489', 'T1491.001', 'T1497.001', 
        'T1497.003', 'T1505.004', 'T1529', 'T1534', 'T1542.003', 
        'T1543.003', 'T1547.001', 'T1547.009', 'T1553.002', 'T1557.001', 
        'T1560', 'T1560.001', 'T1560.002', 'T1560.003', 'T1561.001', 
        'T1561.002', 'T1562.001', 'T1562.004', 'T1564.001', 'T1566.001', 
        'T1566.002', 'T1566.003', 'T1567.002', 'T1571', 'T1573.001', 
        'T1574.002', 'T1574.013', 'T1583.001', 'T1583.004', 'T1583.006', 
        'T1584.001', 'T1584.004', 'T1585.001', 'T1585.002', 'T1587.001', 
        'T1587.002', 'T1588.002', 'T1588.003', 'T1588.004', 'T1589.002', 
        'T1591', 'T1591.004', 'T1593.001', 'T1608.001', 'T1608.002', 
        'T1614.001', 'T1620', 'T1622', 'T1656'
    ],
    'software_str': [
        'S0181', 'S0239', 'S0245', 'S0246', 'S0586', 'S0238', 'S0498', 
        'S0520', 'S0567', 'S0665', 'S0694', 'S0376', 'S0431', 'S0180', 
        'S0241', 'S0103', 'S0271', 'S0366', 'S0678', 'S0263', 'S0347', 
        'S0497', 'S0584', 'S0593', 'S0364', 'S0174', 'S0108'
    ],
    'CAPEC_ID': [
        'CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-309', 
        'CAPEC-561', 'CAPEC-577', 'CAPEC-165', 'CAPEC-504', 'CAPEC-543', 
        'CAPEC-616', 'CAPEC-300', 'CAPEC-568', 'CAPEC-573', 'CAPEC-268', 
        'CAPEC-560', 'CAPEC-70', 'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 
        'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-565', 'CAPEC-295', 
        'CAPEC-196', 'CAPEC-698', 'CAPEC-163', 'CAPEC-552', 'CAPEC-478', 
        'CAPEC-132', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-578', 
        'CAPEC-641', 'CAPEC-640', 'CAPEC-542'
    ],
    'CWE_ID': [
        'CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 
        'CWE-1278', 'CWE-285', 'CWE-200', 'CWE-522', 'CWE-294', 'CWE-521', 
        'CWE-1021', 'CWE-117', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 
        'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 'CWE-693', 'CWE-441', 
        'CWE-384', 'CWE-664', 'CWE-829', 'CWE-451', 'CWE-284', 'CWE-59', 
        'CWE-20', 'CWE-327', 'CWE-290', 'CWE-325', 'CWE-706', 'CWE-114'
    ],
    'CVE_ID': [],
    'Pulse_ID': []
}

def sanitize_evidence(evidence, df_mapping):
    """Keep only evidence features that exist in our database"""
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
    
    return sanitized

def translate_to_encoded(evidence, df_mapping):
    """Convert sanitized evidence to encoded IDs"""
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

# Process evidence
clean_evidence = sanitize_evidence(evidence, df_mapping)
encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)

# Create evidence feature vector
evidence_vector = create_binary_feature_vector(
    {feature_type: encoded_evidence.get(feature_type, []) for feature_type in feature_types},
    all_features
)

# Get embedding for evidence
model.eval()
with torch.no_grad():
    evidence_tensor = torch.FloatTensor(evidence_vector).unsqueeze(0)
    evidence_embedding = model.get_embedding(evidence_tensor).squeeze().numpy()

# Calculate similarities with all APTs
similarities = cosine_similarity([evidence_embedding], embedding_matrix)[0]

# Feature weights for detailed scoring
feature_weights = {
    'techniques': 1.5,
    'software': 1.3,
    'capec': 1.0,
    'cwe': 1.0,
    'cve': 0.8,
    'pulse': 0.7
}

def calculate_weighted_similarity(encoded_evidence, apt_profiles):
    """Calculate weighted similarity scores"""
    similarities = []
    
    for apt_name in apt_ids:
        apt_data = apt_profiles[apt_name]
        total_score = 0
        total_weight = 0
        
        for feature_type, weight in feature_weights.items():
            if feature_type in encoded_evidence and feature_type in apt_data:
                evidence_set = set(encoded_evidence[feature_type])
                profile_set = set(apt_data[feature_type])
                common = len(evidence_set & profile_set)
                
                total_score += common * weight
                total_weight += len(encoded_evidence[feature_type]) * weight
        
        similarity = total_score / total_weight if total_weight > 0 else 0
        similarities.append(similarity)
    
    return np.array(similarities)

# Get weighted similarities
weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles)

# Combine Siamese and weighted similarities (adjust weights based on embedding quality)
embedding_quality = embedding_matrix.std()  # Use std as quality indicator
if embedding_quality < 0.1:
    # If embeddings are still poor, rely more on weighted similarity
    siamese_weight = 0.3
    weighted_weight = 0.7
    print(f"Using conservative weights due to low embedding diversity: Siamese={siamese_weight}, Weighted={weighted_weight}")
else:
    siamese_weight = 0.7
    weighted_weight = 0.3

final_similarities = siamese_weight * similarities + weighted_weight * weighted_similarities

# Create final ranking
full_results = []
sorted_indices = np.argsort(final_similarities)[::-1]

for rank, idx in enumerate(sorted_indices, 1):
    apt_name = apt_ids[idx]
    apt_data = apt_profiles[apt_name]
    
    # Calculate common features
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
        'Siamese_Score': similarities[idx],
        'Weighted_Score': weighted_similarities[idx],
        'Final_Score': final_similarities[idx],
        'Encoded_ID': apt_data['encoded_id'],
        **common_features
    })

# Convert to DataFrame
ranked_apts = pd.DataFrame(full_results)

# Display results
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)

print(f"\nComplete APT Ranking with Improved Siamese Network (Weights: Siamese={siamese_weight}, Weighted={weighted_weight}):")
print("="*140)
print(f"{'Rank':<5} | {'APT':<30} | {'Encoded ID':<10} | {'Final Score':<10} | {'Siamese':<8} | {'Weighted':<8} | {'Techniques':<10} | {'Software':<8} | {'CAPEC':<6} | {'CWE':<4} | {'CVE':<4} | {'Pulse':<5}")
print("-"*140)

for _, row in ranked_apts.iterrows():
    apt_name = (row['APT'][:27] + '..') if len(row['APT']) > 29 else row['APT']
    print(f"{row['Rank']:<5} | {apt_name:<30} | {row['Encoded_ID']:<10} | {row['Final_Score']:.4f} | {row['Siamese_Score']:.4f} | {row['Weighted_Score']:.4f} | {row['techniques']:<10} | {row['software']:<8} | {row['capec']:<6} | {row['cwe']:<4} | {row['cve']:<4} | {row['pulse']:<5}")

# Show top 10 most different Siamese scores for analysis
print(f"\nTop 10 most varied Siamese scores:")
top_varied = ranked_apts.nlargest(10, 'Siamese_Score').nsmallest(10, 'Siamese_Score')
for _, row in top_varied.iterrows():
    print(f"{row['APT']}: {row['Siamese_Score']:.6f}")