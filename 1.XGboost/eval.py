import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from itertools import islice
from matplotlib.colors import LinearSegmentedColormap

#####################################################################
# EFFICIENT MODEL PREDICTOR FOR 6D FEATURES
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model - NOW 6D"""
    def __init__(self, model, apt_features, apt_encoder):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
    
    def predict_similarity(self, features1, features2):
        """Predict similarity between two 6D feature sets"""
        # Skip if first 3 features are all -1 (same as original logic)
        if all(x == -1 for x in features1[:3]):
            return 0.0
        if all(x == -1 for x in features2[:3]):
            return 0.0
            
        # XGBoost gets ALL 6 features (your model expects 6D)
        prob1 = self.model.predict_proba([features1])[0]
        prob2 = self.model.predict_proba([features2])[0]
        
        # Cosine similarity (unchanged)
        cosine_sim = np.dot(prob1, prob2) / (np.linalg.norm(prob1) * np.linalg.norm(prob2))
        
        # 6D Feature overlap (EXTENDED)
        feature_overlap = self._feature_overlap_similarity(features1, features2)
        
        # Combined (same weights)
        combined_similarity = 0.7 * cosine_sim + 0.3 * feature_overlap
        return np.clip(combined_similarity, 0.0, 1.0)
    
    def _feature_overlap_similarity(self, features1, features2):
        """6D feature overlap similarity"""
        overlap_score = 0.0
        
        # Technique (40%)
        if features1[0] == features2[0] and features1[0] != -1: overlap_score += 0.40
        # Software (40%)  
        if features1[1] == features2[1] and features1[1] != -1: overlap_score += 0.40
        # CVE (20%)
        if features1[2] == features2[2] and features1[2] != -1: overlap_score += 0.20
        
        # NEW FEATURES (bonus overlap)
        # CAPEC (5%)
        if len(features1) > 3 and len(features2) > 3 and features1[3] == features2[3] and features1[3] != -1: 
            overlap_score += 0.05
        # CWE (3%)
        if len(features1) > 4 and len(features2) > 4 and features1[4] == features2[4] and features1[4] != -1: 
            overlap_score += 0.03
        # Pulse (2%)
        if len(features1) > 5 and len(features2) > 5 and features1[5] == features2[5] and features1[5] != -1: 
            overlap_score += 0.02
            
        return overlap_score

def load_efficient_xgboost_model():
    """Load the efficient XGBoost model and data"""
    print("🔍 LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('efficient_apt_similarity.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print("✅ Loaded efficient_apt_similarity.pkl")
        return saved_data
    except FileNotFoundError:
        print("❌ efficient_apt_similarity.pkl not found! Please run train.py first.")
        return None

def load_apt_profiles_from_data():
    """Load 6D APT profiles from EncodedData.csv - YOUR CORRECT COLUMNS"""
    print("📊 LOADING 6D APT PROFILES FROM EncodedData.csv...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        
        # YOUR EXACT COLUMNS FROM EncodedData.csv
        tech_max = apt_data['techniques_str'].max() if 'techniques_str' in apt_data.columns else -1
        soft_max = apt_data['software_str'].max() if 'software_str' in apt_data.columns else -1
        cve_max  = apt_data['CVE_ID'].max() if 'CVE_ID' in apt_data.columns else -1
        cap_max  = apt_data['CAPEC_ID'].max() if 'CAPEC_ID' in apt_data.columns else -1
        cwe_max  = apt_data['CWE_ID'].max() if 'CWE_ID' in apt_data.columns else -1
        pul_max  = apt_data['Pulse_ID'].max() if 'Pulse_ID' in apt_data.columns else -1
        
        # 6D profile: [tech, software, cve, capec, cwe, pulse]
        profiles[apt_id] = np.array([tech_max, soft_max, cve_max, cap_max, cwe_max, pul_max], dtype=float)
    
    print(f"✅ Loaded {len(profiles)} 6D APT profiles")
    return profiles

def create_similarity_matrix_efficient(saved_data):
    """Create 6D similarity matrix - SAME LOGIC"""
    print("🎯 CREATING 6D SIMILARITY MATRIX...")
    
    # Load 6D profiles
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Create predictor
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder']
    )
    
    # APT name mapping (unchanged)
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    # Build 6D feature matrix
    X_apts = np.array([profiles[apt_id] for apt_id in apt_ids])
    
    # Calculate pairwise similarities (SAME LOOP)
    num_apts = len(apt_ids)
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    print("Computing 6D pairwise similarities...")
    for i in range(num_apts):
        if i % 20 == 0:
            print(f"  Processed {i}/{num_apts} APTs...")
        for j in range(num_apts):
            similarity = predictor.predict_similarity(X_apts[i], X_apts[j])
            similarity_matrix[i, j] = similarity
    
    similarity_df = pd.DataFrame(similarity_matrix, index=apt_names, columns=apt_names)
    return similarity_df, apt_names

def plot_similarity_heatmap_efficient(similarity_df):
    """Plot 6D similarity heatmap - UNCHANGED"""
    plt.figure(figsize=(16, 14))
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    
    sns.heatmap(similarity_df, 
                cmap=cmap, center=0.5, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': '6D Similarity Score'})
    
    plt.title("APT Group 6D Similarity Matrix (XGBoost + T/S/CVE/CAPEC/CWE/Pulse)", 
              fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_efficient_xgboost_6d_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_similarity_matrix_efficient(similarity_df, apt_names):
    """Evaluate 6D matrix - SAME LOGIC"""
    print("\n" + "="*80)
    print("EVALUATING 6D XGBOOST SIMILARITY MATRIX")
    print("="*80)
    
    top_x_values = [1, 5, 10, 20]
    for top_x in top_x_values:
        success = 0
        failure = 0
        for i, apt_name in enumerate(apt_names):
            similarities = similarity_df.iloc[i]
            top_similar = similarities.nlargest(top_x)
            if apt_name in top_similar.index:
                success += 1
            else:
                failure += 1
        
        total = len(apt_names)
        print(f"\nTop {top_x}: {success}/{failure} ({(success/total)*100:.1f}%)")

def evaluate_profile_division_efficient(saved_data, divider=2, top_x_values=[1, 5, 10, 20]):
    """6D Profile division evaluation - FIXED"""
    print(f"\n🎯 6D PROFILE DIVISION EVALUATION (divider={divider})")
    
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    predictor = EfficientXGBoostPredictor(saved_data['model'], saved_data['apt_features'], saved_data['apt_encoder'])
    X_apts = np.array([profiles[apt_id] for apt_id in apt_ids])
    
    # Build mappings for divided evidence
    df_map = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    mappings = {}
    for ftype, origin_col, encoded_col in [
        ('tech', 'techniques_Origin', 'techniques_Encoded'),
        ('soft', 'software_Origin', 'software_Encoded'),
        ('cve', 'CVE_ID_Origin', 'CVE_ID_Encoded'),
        ('capec', 'CAPEC_ID_Origin', 'CAPEC_ID_Encoded'),
        ('cwe', 'CWE_ID_Origin', 'CWE_ID_Encoded'),
        ('pulse', 'Pulse_ID_Origin', 'Pulse_ID_Encoded')
    ]:
        mappings[ftype] = {}
        for _, row in df_map.iterrows():
            if pd.notna(row.get(origin_col)) and pd.notna(row.get(encoded_col)):
                mappings[ftype][str(row[origin_col])] = int(row[encoded_col])
    
    resultdict = {}
    for i, target_apt_name in enumerate(apt_names):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_names)}...")
        
        target_apt_id = apt_ids[i]
        full_profile_df = pd.read_csv("../BaseData/EncodedData.csv")
        target_data = full_profile_df[full_profile_df['APTname'] == target_apt_id]
        
        divided_features = [-1] * 6
        for idx, (col, mtype) in enumerate([
            ('techniques_str', 'tech'),
            ('software_str', 'soft'), 
            ('CVE_ID', 'cve'),
            ('CAPEC_ID', 'capec'),
            ('CWE_ID', 'cwe'),
            ('Pulse_ID', 'pulse')
        ]):
            if col in target_data.columns:
                values = target_data[col].dropna().unique()
                if len(values) > 0:
                    sample_size = max(1, len(values) // divider)
                    sample = random.sample(list(values), min(sample_size, len(values)))
                    encoded = [mappings[mtype].get(str(v), -1) for v in sample]
                    encoded = [e for e in encoded if e != -1]
                    if encoded:
                        divided_features[idx] = max(encoded)
        
        query_features = np.array(divided_features, dtype=float)
        similarities = [predictor.predict_similarity(query_features, apt_feat) for apt_feat in X_apts]
        ranked_results = {apt_names[j]: similarities[j] for j in range(len(apt_names))}
        resultdict[target_apt_name] = dict(sorted(ranked_results.items(), key=lambda x: x[1], reverse=True))
    
    # FIXED EVALUATION
    for topX in top_x_values:
        success = 0
        total = len(resultdict)
        for target, sims in resultdict.items():
            top_x_similar = dict(islice(sims.items(), topX))
            if target in top_x_similar:
                success += 1
        failure = total - success
        print(f"Top {topX}, Divider {divider}: {success}/{failure} ({success/total*100:.1f}%)")



def main():
    print("🚀 EFFICIENT XGBOOST 6D EVALUATION")
    print("=" * 60)
    
    saved_data = load_efficient_xgboost_model()
    if saved_data is None:
        return
    
    print(f"✅ 6D Model: {saved_data['training_info']['unique_apts']} APTs, {saved_data['training_info']['training_records']} records")
    
    # 1. Full matrix
    print("\n" + "="*80); print("6D SIMILARITY MATRIX")
    similarity_df, apt_names = create_similarity_matrix_efficient(saved_data)
    print(similarity_df.iloc[:5, :5]); print(f"Shape: {similarity_df.shape}")
    
    # 2. Heatmap
    print("\nPlotting 6D heatmap...")
    plot_similarity_heatmap_efficient(similarity_df)
    
    # 3. Matrix evaluation
    evaluate_similarity_matrix_efficient(similarity_df, apt_names)
    
    # 4. Profile division
    print("\n" + "="*80); print("6D PROFILE DIVISION")
    evaluate_profile_division_efficient(saved_data, divider=2)

if __name__ == "__main__":
    main()
