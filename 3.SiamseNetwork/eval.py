# siamese_query.py - EVALUATION ONLY (no custom queries)
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from itertools import islice
import random

# ---------- Recreate the APTSiameseNetwork definition ----------
import torch.nn as nn
class APTSiameseNetwork(nn.Module):
    def __init__(self, num_apts, embedding_dim=64):
        super().__init__()
        self.apt_embedding = nn.Embedding(num_apts, embedding_dim)
        
    def forward(self, apt_idx1, apt_idx2):
        emb1 = self.apt_embedding(apt_idx1)
        emb2 = self.apt_embedding(apt_idx2)
        return emb1, emb2
    
    def get_similarity(self, emb1, emb2):
        emb1 = emb1.view(1, -1)
        emb2 = emb2.view(1, -1)
        return F.cosine_similarity(emb1, emb2).squeeze()
    
    def get_all_embeddings(self):
        return self.apt_embedding.weight

# ---------- Helper functions for evaluation ----------
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
            valid_values = set(df_mapping[origin_col].dropna().astype(str))
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
    
    if missing:
        print("WARNING: These features were not found in our database:")
        for feature_type, items in missing.items():
            print(f"- {feature_type}: {items[:5]}")
            if len(items) > 5:
                print(f"  (and {len(items)-5} more)")
    
    return sanitized

def translate_to_encoded(evidence, df_mapping):
    """Convert sanitized evidence (orig values) to encoded IDs expected by apt profiles"""
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
            mapping = dict(zip(df_mapping[origin_col].astype(str), df_mapping[encoded_col]))
            
            encoded_values = []
            for val in evidence[evidence_key]:
                if val in mapping:
                    encoded_values.append(mapping[val])
            if encoded_values:
                encoded_evidence[output_key] = encoded_values
    
    return encoded_evidence

def calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids):
    """Calculate weighted similarity scores for all APTs"""
    feature_weights = {
        'techniques': 1.5,
        'software': 1.3,
        'capec': 1.0,
        'cwe': 1.0,
        'cve': 0.8,
        'pulse': 0.7
    }
    
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
    
    return torch.tensor(similarities)

def create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids):
    """Create similarity DataFrame from embeddings"""
    if torch.is_tensor(apt_embeddings):
        emb_np = apt_embeddings.detach().cpu().numpy()
    else:
        emb_np = np.array(apt_embeddings)
    
    similarity_matrix = cosine_similarity(emb_np)
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    similarity_df = pd.DataFrame(similarity_matrix, index=apt_names, columns=apt_names)
    return similarity_df

def plot_similarity_heatmap(similarity_df):
    plt.figure(figsize=(16, 14))
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    sns.heatmap(similarity_df, mask=None, cmap=cmap, center=0.5, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={'shrink':0.8, 'label':'Similarity Score'})
    plt.title("APT Group Similarity Matrix (Siamese embeddings)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

# ---------- Main evaluation functions ----------
def load_siamese_complete(path="./siamese_complete_model.pkl"):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    model = APTSiameseNetwork(data['model_config']['num_apts'], data['model_config']['embedding_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    apt_embeddings = data.get('apt_embeddings')
    if isinstance(apt_embeddings, np.ndarray):
        apt_embeddings = torch.tensor(apt_embeddings, dtype=torch.float)
    elif torch.is_tensor(apt_embeddings):
        apt_embeddings = apt_embeddings.float()
    else:
        apt_embeddings = model.get_all_embeddings().detach().float()
    
    return model, data, apt_embeddings

def evaluate_similarity_matrix(similarity_df):
    """Evaluate similarity matrix performance (Top-1, 5, 10, 20)"""
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX (Top-1, 5, 10, 20)")
    print("="*80)

    # Get APT names from the similarity matrix
    apt_names_sim = list(similarity_df.index)
    sim_matrix_results = {}

    # For each APT, get its similarity scores from the matrix
    for target_apt in apt_names_sim:
        similarities = similarity_df.loc[target_apt].to_dict()
        sim_matrix_results[target_apt] = similarities

    # Evaluate for multiple top-X values
    top_x_values_sim = [1, 5, 10, 20]

    for topX in top_x_values_sim:
        lengthResultDict = len(sim_matrix_results)
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in sim_matrix_results.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs (exclude self if needed, but include for fair comparison)
            top_x_similar = dict(list(sorted_similarities.items())[:topX])
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success += 1
            else:
                failure += 1
        
        print(f"\nSimilarity Matrix - TopX={topX}:")
        print(f"Success: {success}/{lengthResultDict} | Failure: {failure}/{lengthResultDict}")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")

def divide_apt_profile(apt_name, apt_profiles, apt_name_dict, apt_ids, df_mapping, divider=2):
    """Divide an APT's profile data and return the subset evidence"""
    # Find the APT ID from the name
    target_apt_id = None
    for apt_id in apt_ids:
        if apt_name_dict[apt_profiles[apt_id]['encoded_id']] == apt_name:
            target_apt_id = apt_id
            break
    
    if not target_apt_id:
        return {}
    
    target_data = apt_profiles[target_apt_id]
    divided_evidence = {}
    
    for feature_type, evidence_key in [('techniques', 'techniques_str'),
                                    ('software', 'software_str'),
                                    ('capec', 'CAPEC_ID'),
                                    ('cwe', 'CWE_ID'),
                                    ('cve', 'CVE_ID'),
                                    ('pulse', 'Pulse_ID')]:
        
        if feature_type in target_data and target_data[feature_type]:
            feature_list = target_data[feature_type].copy()
            random.shuffle(feature_list)
            division_size = max(1, len(feature_list) // divider)
            divided_features = feature_list[:division_size]
            
            # Convert encoded features back to original format
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
    
    return divided_evidence

def evaluate_profile_division_performance(apt_profiles, apt_name_dict, apt_ids, df_mapping, similarity_df, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate performance when using divided profile data"""
    apt_names = list(similarity_df.index)
    resultdict = {}

    print(f"\nEvaluating Siamese similarity matrix with divider={divider} (using {100/divider:.1f}% of profile data)")

    # For each APT, divide its profile and calculate similarities
    for target_apt_name in apt_names:
        
        # Divide the profile
        divided_evidence = divide_apt_profile(target_apt_name, apt_profiles, apt_name_dict, apt_ids, df_mapping, divider)
        
        # Process the divided evidence
        clean_evidence = sanitize_evidence(divided_evidence, df_mapping)
        encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
        
        # Calculate weighted similarities
        weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids)
        
        # Create ranked results
        ranked_results = {}
        for idx in weighted_similarities.sort(descending=True).indices:
            apt_name = apt_ids[idx]
            apt_real_name = apt_name_dict[apt_profiles[apt_name]['encoded_id']]
            ranked_results[apt_real_name] = weighted_similarities[idx].item()
        
        resultdict[target_apt_name] = ranked_results

    # Evaluate Top-X
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SuccessFailDict = {}

        for target_apt, similarities in resultdict.items():
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            top_x_similar = dict(islice(sorted_similarities.items(), topX))

            if target_apt in top_x_similar:
                success += 1
                SuccessFailDict[target_apt] = "Pass"
            else:
                failure += 1
                SuccessFailDict[target_apt] = "Fail"

        print(f"\nSettings: TopX={topX}, Divider={divider}")
        print(f"Siamese model success/failure: {success}/{failure}  (Total={lengthResultDict})")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")

# ---------- Main evaluation script ----------
def main():

    model_path = os.path.join('./siamese_complete_model.pkl')
    print("Loading siamese complete model & metadata...")
    model, data, apt_embeddings = load_siamese_complete(model_path)
    apt_profiles = data['apt_profiles']
    apt_name_dict = data['apt_name_dict']
    apt_ids = data['apt_ids']
    df_mapping = data['df_mapping']
    
    # 1) Create full similarity DataFrame & show a sample
    print("Creating similarity matrix (from siamese embeddings)...")
    similarity_df = create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids)
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 2) Plot heatmap
    print("Plotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # 3) Evaluate similarity matrix performance
    evaluate_similarity_matrix(similarity_df)
    
    # 4) Evaluate profile division performance
    evaluate_profile_division_performance(
        apt_profiles=apt_profiles,
        apt_name_dict=apt_name_dict,
        apt_ids=apt_ids,
        df_mapping=df_mapping,
        similarity_df=similarity_df,
        divider=2,
        top_x_values=[1, 5, 10, 20]
    )

if __name__ == "__main__":
    main()