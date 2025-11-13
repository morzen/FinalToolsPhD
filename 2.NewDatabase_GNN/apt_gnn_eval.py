import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import random
from itertools import islice

class APTQueryModel(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwe, num_cve, num_pulse, hidden_dim=128):
        super().__init__()
        self.technique_embedding = nn.Embedding(num_techniques, hidden_dim, padding_idx=0)
        self.software_embedding = nn.Embedding(num_software, hidden_dim, padding_idx=0)
        self.capec_embedding = nn.Embedding(num_capec, hidden_dim, padding_idx=0)
        self.cwe_embedding = nn.Embedding(num_cwe, hidden_dim, padding_idx=0)
        self.cve_embedding = nn.Embedding(num_cve, hidden_dim, padding_idx=0)
        self.pulse_embedding = nn.Embedding(num_pulse, hidden_dim, padding_idx=0)
        
        # Combine all 6 feature types
        self.feature_combiner = nn.Linear(hidden_dim * 6, hidden_dim)
        
    def forward(self, techniques, software, capec, cwe, cves, pulse):
        tech_emb = self.technique_embedding(techniques).mean(dim=1)
        soft_emb = self.software_embedding(software).mean(dim=1)
        capec_emb = self.capec_embedding(capec).mean(dim=1)
        cwe_emb = self.cwe_embedding(cwe).mean(dim=1)
        cve_emb = self.cve_embedding(cves).mean(dim=1)
        pulse_emb = self.pulse_embedding(pulse).mean(dim=1)
        
        combined = torch.cat([tech_emb, soft_emb, capec_emb, cwe_emb, cve_emb, pulse_emb], dim=1)
        return self.feature_combiner(combined)

def load_trained_model():
    """Load the trained model and data"""
    print("🔍 LOADING TRAINED MODEL...")
    with open('trained_apt_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    return saved_data

def convert_evidence_to_encoded(evidence, mapping_file="../BaseData/comparison_Encoded_df.csv"):
    """Convert human-readable evidence to encoded IDs using the mapping file"""
    df_mapping = pd.read_csv(mapping_file)
    
    # Create mapping dictionaries for all 6 feature types
    technique_mapping = {}
    software_mapping = {} 
    capec_mapping = {}
    cwe_mapping = {}
    cve_mapping = {}
    pulse_mapping = {}
    
    for _, row in df_mapping.iterrows():
        # Technique mapping
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            if row['techniques_Origin'] not in technique_mapping:
                technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        
        # Software mapping
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            if row['software_Origin'] not in software_mapping:
                software_mapping[row['software_Origin']] = row['software_Encoded']
        
        # CAPEC mapping
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            if row['CAPEC_ID_Origin'] not in capec_mapping:
                capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']
        
        # CWE mapping
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            if row['CWE_ID_Origin'] not in cwe_mapping:
                cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']
        
        # CVE mapping
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            if row['CVE_ID_Origin'] not in cve_mapping:
                cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
        
        # Pulse mapping
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            if row['Pulse_ID_Origin'] not in pulse_mapping:
                pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']
    
    # Convert evidence
    encoded_evidence = {
        'techniques': [],
        'software': [],
        'capec': [],
        'cwe': [],
        'cves': [],
        'pulse': []
    }
    
    # Convert techniques
    for technique in evidence.get('techniques', []):
        if technique in technique_mapping:
            encoded_evidence['techniques'].append(technique_mapping[technique])
    
    # Convert software
    for software in evidence.get('software', []):
        if software in software_mapping:
            encoded_evidence['software'].append(software_mapping[software])
    
    # Convert CAPEC
    for capec in evidence.get('capec', []):
        if capec in capec_mapping:
            encoded_evidence['capec'].append(capec_mapping[capec])
    
    # Convert CWE
    for cwe in evidence.get('cwe', []):
        if cwe in cwe_mapping:
            encoded_evidence['cwe'].append(cwe_mapping[cwe])
    
    # Convert CVEs
    for cve in evidence.get('cves', []):
        if cve in cve_mapping:
            encoded_evidence['cves'].append(cve_mapping[cve])
    
    # Convert Pulse
    for pulse in evidence.get('pulse', []):
        if pulse in pulse_mapping:
            encoded_evidence['pulse'].append(pulse_mapping[pulse])
    
    return encoded_evidence

def query_apt_similarity(evidence, saved_data):
    """Query APT similarity using the trained model - SAME AS apt_query.py"""
    # Load the trained model
    model_config = saved_data['model_config']
    model = APTQueryModel(
        num_techniques=model_config['num_techniques'],
        num_software=model_config['num_software'],
        num_capec=model_config['num_capec'],
        num_cwe=model_config['num_cwe'],
        num_cve=model_config['num_cve'],
        num_pulse=model_config['num_pulse'],
        hidden_dim=model_config['hidden_dim']
    )
    model.load_state_dict(saved_data['model_state_dict'])
    model.eval()
    
    # Get feature mappings
    feature_mappings = saved_data['feature_mappings']
    technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, cve_to_idx, pulse_to_idx = feature_mappings
    
    # Get APT embeddings
    apt_embeddings = saved_data['apt_embeddings']
    apt_ids = saved_data['apt_ids']
    
    # Convert evidence to indices
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    capec = evidence.get('capec', [])
    cwe = evidence.get('cwe', [])
    cves = evidence.get('cves', [])
    pulse = evidence.get('pulse', [])
    
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
    
    # Generate query embedding using the trained model
    with torch.no_grad():
        query_embedding = model(
            tech_tensor.unsqueeze(0), 
            soft_tensor.unsqueeze(0),
            capec_tensor.unsqueeze(0),
            cwe_tensor.unsqueeze(0),
            cve_tensor.unsqueeze(0),
            pulse_tensor.unsqueeze(0)
        )
    
    # Use dot product for similarity - SAME AS TRAINING
    similarities = torch.matmul(query_embedding, apt_embeddings.t()).squeeze(0)
    
    return similarities, apt_ids

def create_similarity_matrix(saved_data):
    """Create similarity matrix between all APTs using the trained model"""
    print("Creating similarity matrix using trained GNN model...")
    
    apt_embeddings = saved_data['apt_embeddings']
    apt_ids = saved_data['apt_ids']
    profiles = saved_data['profiles']
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    # Get APT names in order
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    # Calculate cosine similarity matrix using the trained embeddings
    similarity_matrix = cosine_similarity(apt_embeddings.detach().numpy())
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=apt_names,
        columns=apt_names
    )
    
    return similarity_df, apt_names

def plot_similarity_heatmap(similarity_df):
    """Plot heatmap of similarity matrix"""
    plt.figure(figsize=(16, 14))
    
    # Create custom red-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    
    # Plot complete heatmap
    sns.heatmap(similarity_df, 
                mask=None,
                cmap=cmap,
                center=0.5,
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Similarity Score'})
    
    plt.title("APT Group Similarity Matrix (GNN Embeddings)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_GNN_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_apt_embeddings(saved_data):
    """2D visualization of APT embeddings using PCA"""
    apt_embeddings = saved_data['apt_embeddings']
    apt_ids = saved_data['apt_ids']
    profiles = saved_data['profiles']
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    # Convert embeddings to numpy
    embeddings = apt_embeddings.detach().numpy()
    names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color points by number of techniques (TTPs are KING)
    technique_counts = [len(profiles[apt_id]['techniques']) for apt_id in apt_ids]
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=technique_counts, cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Number of Techniques (TTPs)')
    
    # Add labels for prominent APTs
    for i, name in enumerate(names):
        if technique_counts[i] > 100:  # Label only APTs with many techniques
            plt.annotate(name, 
                        (embeddings_2d[i, 0] + 0.02, embeddings_2d[i, 1] + 0.02),
                        fontsize=8, 
                        alpha=0.8,
                        fontweight='bold')
    
    plt.title("APT Embeddings Visualization (Colored by TTP Count)", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('apt_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_similarity_matrix(similarity_df, apt_names):
    """Evaluate the similarity matrix for different top-X values"""
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX (FULL DATA)")
    print("="*80)
    
    # Evaluate for multiple top-X values
    top_x_values = [1, 5, 10, 20]
    
    for top_x in top_x_values:
        success = 0
        failure = 0
        
        # For each APT, check if it appears in its own top X most similar APTs
        for i, apt_name in enumerate(apt_names):
            # Get similarity scores for this APT
            similarities = similarity_df.iloc[i]
            
            # Get top X most similar APTs (including self)
            top_similar = similarities.nlargest(top_x)
            
            # Check if the APT itself is in its own top X most similar
            if apt_name in top_similar.index:
                success += 1
            else:
                failure += 1
        
        total = len(apt_names)
        print(f"\nSettings: Top {top_x} (Full Data)")
        print(f"Success/Failure: {success}/{failure} Total: {total}")
        print(f"Success rate: {(success/total)*100:.2f}%")

def evaluate_profile_division(saved_data, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance with divided profile data - USING SAME QUERY APPROACH"""
    profiles = saved_data['profiles']
    apt_ids = saved_data['apt_ids']
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    print(f"\nEvaluating with divider: {divider} (using {100/divider:.1f}% of profile data)")
    
    resultdict = {}
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        # Get the target APT's original data
        target_apt_id = next(apt for apt in apt_ids if apt_name_mapping.get(apt, f"APT{apt}") == target_apt_name)
        target_profile = profiles[target_apt_id]
        
        # Create divided evidence for all 6 feature types
        divided_evidence = {
            'techniques': [],
            'software': [],
            'capec': [],
            'cwe': [],
            'cves': [],
            'pulse': []
        }
        
        # Divide techniques
        if target_profile['techniques']:
            techniques_list = list(target_profile['techniques'])
            random.shuffle(techniques_list)
            division_size = max(1, len(techniques_list) // divider)
            divided_evidence['techniques'] = techniques_list[:division_size]
        
        # Divide software
        if target_profile['software']:
            software_list = list(target_profile['software'])
            random.shuffle(software_list)
            division_size = max(1, len(software_list) // divider)
            divided_evidence['software'] = software_list[:division_size]
        
        # Divide CAPEC
        if target_profile['capec']:
            capec_list = list(target_profile['capec'])
            random.shuffle(capec_list)
            division_size = max(1, len(capec_list) // divider)
            divided_evidence['capec'] = capec_list[:division_size]
        
        # Divide CWE
        if target_profile['cwe']:
            cwe_list = list(target_profile['cwe'])
            random.shuffle(cwe_list)
            division_size = max(1, len(cwe_list) // divider)
            divided_evidence['cwe'] = cwe_list[:division_size]
        
        # Divide CVEs
        if target_profile['cve']:
            cve_list = list(target_profile['cve'])
            random.shuffle(cve_list)
            division_size = max(1, len(cve_list) // divider)
            divided_evidence['cves'] = cve_list[:division_size]
        
        # Divide Pulse
        if target_profile['pulse']:
            pulse_list = list(target_profile['pulse'])
            random.shuffle(pulse_list)
            division_size = max(1, len(pulse_list) // divider)
            divided_evidence['pulse'] = pulse_list[:division_size]
        
        # Convert to human-readable for encoding (to match apt_query.py process)
        human_evidence = {
            'techniques': [],
            'software': [],
            'capec': [],
            'cwe': [],
            'cves': [],
            'pulse': []
        }
        
        # Convert encoded features back to original format
        technique_mapping = dict(zip(df_mapping['techniques_Encoded'], df_mapping['techniques_Origin']))
        software_mapping = dict(zip(df_mapping['software_Encoded'], df_mapping['software_Origin']))
        capec_mapping = dict(zip(df_mapping['CAPEC_ID_Encoded'], df_mapping['CAPEC_ID_Origin']))
        cwe_mapping = dict(zip(df_mapping['CWE_ID_Encoded'], df_mapping['CWE_ID_Origin']))
        cve_mapping = dict(zip(df_mapping['CVE_ID_Encoded'], df_mapping['CVE_ID_Origin']))
        pulse_mapping = dict(zip(df_mapping['Pulse_ID_Encoded'], df_mapping['Pulse_ID_Origin']))
        
        for tech in divided_evidence['techniques']:
            if tech in technique_mapping:
                human_evidence['techniques'].append(technique_mapping[tech])
        
        for soft in divided_evidence['software']:
            if soft in software_mapping:
                human_evidence['software'].append(software_mapping[soft])
        
        for capec in divided_evidence['capec']:
            if capec in capec_mapping:
                human_evidence['capec'].append(capec_mapping[capec])
        
        for cwe in divided_evidence['cwe']:
            if cwe in cwe_mapping:
                human_evidence['cwe'].append(cwe_mapping[cwe])
        
        for cve in divided_evidence['cves']:
            if cve in cve_mapping:
                human_evidence['cves'].append(cve_mapping[cve])
        
        for pulse in divided_evidence['pulse']:
            if pulse in pulse_mapping:
                human_evidence['pulse'].append(pulse_mapping[pulse])
        
        # Convert back to encoded using the same function as apt_query.py
        encoded_evidence = convert_evidence_to_encoded(human_evidence)
        
        # Query using the SAME approach as apt_query.py
        similarities, query_apt_ids = query_apt_similarity(encoded_evidence, saved_data)
        
        # Create ranked results
        ranked_results = {}
        sorted_indices = torch.argsort(similarities, descending=True)
        
        for idx in sorted_indices:
            apt_id = query_apt_ids[idx.item()]
            apt_real_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
            ranked_results[apt_real_name] = similarities[idx].item()
        
        resultdict[target_apt_name] = ranked_results
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SucessFailDict = {}
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Get top X most similar APTs
            top_x_similar = dict(islice(similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success += 1
                SucessFailDict[target_apt] = "Pass"
            else:
                failure += 1
                SucessFailDict[target_apt] = "Fail"
        
        print(f"\nSettings: Top {topX}, Divider: {divider}")
        print(f"Success/Failure: {success}/{failure} Total: {lengthResultDict}")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")

def test_sample_queries(saved_data):
    """Test sample queries to verify the model works correctly"""
    print("\n" + "="*80)
    print("TESTING SAMPLE QUERIES")
    print("="*80)
    
    # Test the same evidence as in apt_query.py
    test_evidence = {
        'techniques': ["T1001.002", "T1003", "T1558", "T1217"],
        'software': ["S0672"], 
        'capec': ["CAPEC-100", "CAPEC-200"],
        'cwe': ["CWE-79", "CWE-89"],
        'cves': ["CVE-2024-27894", "CVE-2024-31141"],
        'pulse': ["PULSE-1", "PULSE-2"]
    }
    
    print("Testing with evidence:")
    print(f"  Techniques: {test_evidence['techniques']}")
    print(f"  Software: {test_evidence['software']}")
    print(f"  CAPEC: {test_evidence['capec']}")
    print(f"  CWE: {test_evidence['cwe']}")
    print(f"  CVEs: {test_evidence['cves']}")
    print(f"  Pulse: {test_evidence['pulse']}")
    
    # Convert and query
    encoded_evidence = convert_evidence_to_encoded(test_evidence)
    similarities, apt_ids = query_apt_similarity(encoded_evidence, saved_data)
    
    # Get top 5 results
    top5_indices = torch.topk(similarities, 5).indices
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    print("\nTop 5 most similar APTs:")
    for i, idx in enumerate(top5_indices):
        apt_id = apt_ids[idx.item()]
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        similarity = similarities[idx].item()
        print(f"  {i+1}. {apt_name}: {similarity:.4f}")

def main():
    print("🚀 GNN APT SIMILARITY MODEL EVALUATION")
    print("=" * 60)
    
    # Load trained model
    saved_data = load_trained_model()
    
    print(f"✅ Model loaded successfully!")
    print(f"   APTs: {len(saved_data['apt_ids'])}")
    print(f"   Embedding dimension: {saved_data['apt_embeddings'].shape[1]}")
    print(f"   Profiles: {len(saved_data['profiles'])}")
    
    # Print feature statistics
    feature_mappings = saved_data['feature_mappings']
    print(f"   Features: {len(feature_mappings[0])} techniques, {len(feature_mappings[1])} software, "
          f"{len(feature_mappings[2])} CAPEC, {len(feature_mappings[3])} CWE, {len(feature_mappings[4])} CVEs, {len(feature_mappings[5])} Pulse")
    
    # 1. Test sample queries
    test_sample_queries(saved_data)
    
    # 2. Create and display similarity matrix
    print("\n" + "="*80)
    print("CREATING SIMILARITY MATRIX")
    print("="*80)
    
    similarity_df, apt_names = create_similarity_matrix(saved_data)
    
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 3. Plot heatmap
    print("\nPlotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # 4. Visualize embeddings
    print("\nVisualizing APT embeddings...")
    visualize_apt_embeddings(saved_data)
    
    # 5. Evaluate similarity matrix
    evaluate_similarity_matrix(similarity_df, apt_names)
    
    # 6. Evaluate profile division (only divider 2 with Top 20)
    print("\n" + "="*80)
    print("EVALUATING PROFILE DIVISION PERFORMANCE")
    print("="*80)
    
    evaluate_profile_division(saved_data, divider=2, top_x_values=[1, 5, 10, 20])

if __name__ == "__main__":
    main()