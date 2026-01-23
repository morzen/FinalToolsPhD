import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from itertools import islice

#####################################################################
# EFFICIENT MODEL PREDICTOR FOR EVALUATION
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model"""
    def __init__(self, model, apt_features, apt_encoder):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
    
    def predict_similarity(self, features1, features2):
        """Predict similarity between two feature sets"""
        if features1[0] == -1 and features1[1] == -1 and features1[2] == -1:
            return 0.0
        if features2[0] == -1 and features2[1] == -1 and features2[2] == -1:
            return 0.0
            
        # Get probability distributions
        prob1 = self.model.predict_proba([features1])[0]
        prob2 = self.model.predict_proba([features2])[0]
        
        # Multiple similarity measures
        cosine_sim = np.dot(prob1, prob2) / (np.linalg.norm(prob1) * np.linalg.norm(prob2))
        
        # Feature overlap similarity
        feature_overlap = self._feature_overlap_similarity(features1, features2)
        
        # Combined similarity
        combined_similarity = cosine_sim #+ 0.3 * feature_overlap
        
        return np.clip(combined_similarity, 0.0, 1.0)
    
    def _feature_overlap_similarity(self, features1, features2):
        """Calculate similarity based on feature overlap"""
        overlap_score = 0.0
        
        # Technique similarity
        if features1[0] == features2[0] and features1[0] != -1:
            overlap_score += 0.6
        
        # Software similarity  
        if features1[1] == features2[1] and features1[1] != -1:
            overlap_score += 0.5
            
        # Capec similarity
        if features1[2] == features2[2] and features1[2] != -1:
            overlap_score += 0.4
        
        # CWE similarity
        if features1[3] == features2[3] and features1[3] != -1:
            overlap_score += 0.3

        # CVE similarity
        if features1[4] == features2[4] and features1[4] != -1:
            overlap_score += 0.2
        
        # Pulse similarity
        if features1[5] == features2[5] and features1[5] != -1:
            overlap_score += 0.1
            
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
    """Load APT profiles from the encoded data file"""
    print("📊 LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].unique()),
            'software': set(apt_data['software_str'].unique()),
            'capec': set(apt_data['CAPEC_ID'].unique()),
            'cwe': set(apt_data['CWE_ID'].unique()),
            'cve': set(apt_data['CVE_ID'].unique()),
            'pulse': set(apt_data['Pulse_ID'].unique()),
            'total_features': len(apt_data)
        }
    
    print(f"✅ Loaded profiles for {len(profiles)} APTs")
    return profiles

def create_similarity_matrix_efficient(saved_data):
    """Create similarity matrix between all APTs using the efficient XGBoost model"""
    print("Creating similarity matrix using efficient XGBoost model...")
    
    # Load APT profiles
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Create predictor
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder']
    )
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    # Get APT names in order
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    # Create feature matrix for all APTs (using representative features)
    X_apts = []
    for apt_id in apt_ids:
        profile = profiles[apt_id]
        # Use most common technique, software, CVE for this APT
        technique_item = next(iter(profile['techniques'])) if profile['techniques'] else -1
        software_item = next(iter(profile['software'])) if profile['software'] else -1
        capec_item = next(iter(profile['capec'])) if profile['capec'] else -1
        cwe_item = next(iter(profile['cwe'])) if profile['cwe'] else -1
        cve_item = next(iter(profile['cve'])) if profile['cve'] else -1
        pulse_item = next(iter(profile['pulse'])) if profile['pulse'] else -1
        
        apt_features = np.array([technique_item, software_item, capec_item, cwe_item, cve_item, pulse_item])
        X_apts.append(apt_features)
    
    X_apts = np.array(X_apts)
    
    # Calculate similarity matrix using XGBoost predictions
    num_apts = len(apt_ids)
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    print("Computing pairwise similarities...")
    for i in range(num_apts):
        if i % 20 == 0:
            print(f"  Processed {i}/{num_apts} APTs...")
        
        for j in range(num_apts):
            similarity = predictor.predict_similarity(X_apts[i], X_apts[j])
            similarity_matrix[i, j] = similarity
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=apt_names,
        columns=apt_names
    )
    
    return similarity_df, apt_names

def plot_similarity_heatmap_efficient(similarity_df):
    """Plot heatmap of similarity matrix for efficient XGBoost"""
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
    
    plt.title("APT Group Similarity Matrix (Efficient XGBoost)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_efficient_xgboost_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_similarity_matrix_efficient(similarity_df, apt_names):
    """Evaluate the similarity matrix for different top-X values"""
    print("\n" + "="*80)
    print("EVALUATING EFFICIENT XGBOOST SIMILARITY MATRIX (FULL DATA)")
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

def evaluate_profile_division_efficient(saved_data, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate efficient XGBoost model performance with divided profile data"""
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    # Create predictor
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder']
    )
    
    # Create feature matrix for all APTs
    X_apts = []
    for apt_id in apt_ids:
        profile = profiles[apt_id]
        technique_item = next(iter(profile['techniques'])) if profile['techniques'] else -1
        software_item = next(iter(profile['software'])) if profile['software'] else -1
        capec_item = next(iter(profile['capec'])) if profile['capec'] else -1
        cwe_item = next(iter(profile['cwe'])) if profile['cwe'] else -1
        cve_item = next(iter(profile['cve'])) if profile['cve'] else -1
        pulse_item = next(iter(profile['pulse'])) if profile['pulse'] else -1
        apt_features = np.array([technique_item, software_item, capec_item, cwe_item, cve_item, pulse_item])
        X_apts.append(apt_features)
    
    X_apts = np.array(X_apts)
    
    print(f"\nEvaluating Efficient XGBoost with divider: {divider} (using {100/divider:.1f}% of profile data)")
    
    resultdict = {}
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(apt_names)} APTs...")
        
        # Get the target APT's original data
        target_apt_id = next(apt for apt in apt_ids if apt_name_mapping.get(apt, f"APT{apt}") == target_apt_name)
        target_profile = profiles[target_apt_id]
        
        # Create divided evidence for each feature type
        divided_evidence = {
            'techniques': [],
            'software': [],
            'capec': [],
            'cwes': [],
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

        # Divide CWEs
        if target_profile['cwe']:
            cwe_list = list(target_profile['cwe'])
            random.shuffle(cwe_list)
            division_size = max(1, len(cwe_list) // divider)
            divided_evidence['cwes'] = cwe_list[:division_size]

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
        
        # Create query features from divided evidence
        technique_item = divided_evidence['techniques'][0] if divided_evidence['techniques'] else -1
        software_item = divided_evidence['software'][0] if divided_evidence['software'] else -1
        capec_item = divided_evidence['capec'][0] if divided_evidence['capec'] else -1
        cwe_item = divided_evidence['cwes'][0] if divided_evidence['cwes'] else -1
        cve_item = divided_evidence['cves'][0] if divided_evidence['cves'] else -1
        pulse_item = divided_evidence['pulse'][0] if divided_evidence['pulse'] else -1
        
        query_features = np.array([technique_item, software_item, capec_item, cwe_item, cve_item, pulse_item])
        
        # Calculate similarities with all APTs
        similarities = []
        for apt_features in X_apts:
            similarity = predictor.predict_similarity(query_features, apt_features)
            similarities.append(similarity)
        
        # Create ranked results
        ranked_results = {}
        sorted_indices = np.argsort(similarities)[::-1]  # Sort descending
        
        for idx in sorted_indices:
            apt_id = apt_ids[idx]
            apt_real_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
            ranked_results[apt_real_name] = similarities[idx]
        
        resultdict[target_apt_name] = ranked_results
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Get top X most similar APTs
            top_x_similar = dict(islice(similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success += 1
            else:
                failure += 1
        
        print(f"\nSettings: Top {topX}, Divider: {divider}")
        print(f"Success/Failure: {success}/{failure} Total: {lengthResultDict}")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")

def main():
    print("🚀 EFFICIENT XGBOOST APT SIMILARITY MODEL EVALUATION")
    print("=" * 60)
    
    # Load trained XGBoost model
    saved_data = load_efficient_xgboost_model()
    
    if saved_data is None:
        print("❌ Cannot proceed without trained model. Please run train.py first.")
        return
    
    print(f"✅ Efficient XGBoost model loaded successfully!")
    print(f"   APTs in training: {saved_data['training_info']['unique_apts']}")
    print(f"   Training records: {saved_data['training_info']['training_records']}")
    
    # 1. Create and display similarity matrix
    print("\n" + "="*80)
    print("CREATING EFFICIENT XGBOOST SIMILARITY MATRIX")
    print("="*80)
    
    similarity_df, apt_names = create_similarity_matrix_efficient(saved_data)
    
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 2. Plot heatmap
    print("\nPlotting Efficient XGBoost similarity heatmap...")
    plot_similarity_heatmap_efficient(similarity_df)
    
    # 3. Evaluate similarity matrix
    evaluate_similarity_matrix_efficient(similarity_df, apt_names)
    
    # 4. Evaluate profile division
    print("\n" + "="*80)
    print("EVALUATING EFFICIENT XGBOOST PROFILE DIVISION PERFORMANCE")
    print("="*80)
    
    evaluate_profile_division_efficient(saved_data, divider=2, top_x_values=[1, 5, 10, 20])

if __name__ == "__main__":
    main()