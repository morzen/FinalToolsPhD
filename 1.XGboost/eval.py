import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from itertools import islice

#####################################################################
# EFFICIENT MODEL PREDICTOR FOR EVALUATION (UPDATED FOR PCA MODEL)
#####################################################################

class EfficientXGBoostPredictor:
    """Predictor for the efficient XGBoost model with PCA"""
    def __init__(self, model, apt_features, apt_encoder, feature_metadata):
        self.model = model
        self.apt_features = apt_features
        self.apt_encoder = apt_encoder
        self.reverse_encoder = {v: k for k, v in apt_encoder.items()}
        self.feature_metadata = feature_metadata
    
    def predict_vector_similarity(self, vector1, vector2):
        """Predict similarity between two PCA vectors using cosine similarity"""
        # Simple cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, 0.0, 1.0)
    
    def create_vector_from_evidence(self, evidence):
        """Create PCA vector from evidence dictionary"""
        techniques = evidence.get('techniques', [])
        software = evidence.get('software', [])
        capec = evidence.get('capec', [])
        cwe = evidence.get('cwe', [])
        cve = evidence.get('cve', [])
        pulse = evidence.get('pulse', [])
        
        # Create binary vector
        vector = np.zeros(self.feature_metadata['pulse_offset'] + 
                        len(self.feature_metadata['all_pulse']), dtype=np.float32)
        
        # Mark all features from evidence
        for t in techniques:
            if t in self.feature_metadata['all_techniques']:
                idx = self.feature_metadata['all_techniques'].index(t)
                vector[idx] = 1.0
        
        for s in software:
            if s in self.feature_metadata['all_software']:
                idx = self.feature_metadata['all_software'].index(s) + self.feature_metadata['software_offset']
                vector[idx] = 1.0
        
        for c in capec:
            if c in self.feature_metadata['all_capec']:
                idx = self.feature_metadata['all_capec'].index(c) + self.feature_metadata['capec_offset']
                vector[idx] = 1.0
        
        for w in cwe:
            if w in self.feature_metadata['all_cwe']:
                idx = self.feature_metadata['all_cwe'].index(w) + self.feature_metadata['cwe_offset']
                vector[idx] = 1.0
        
        for v in cve:
            if v in self.feature_metadata['all_cve']:
                idx = self.feature_metadata['all_cve'].index(v) + self.feature_metadata['cve_offset']
                vector[idx] = 1.0
        
        for p in pulse:
            if p in self.feature_metadata['all_pulse']:
                idx = self.feature_metadata['all_pulse'].index(p) + self.feature_metadata['pulse_offset']
                vector[idx] = 1.0
        
        # Apply PCA transformation
        if 'pca' in self.feature_metadata:
            pca_model = self.feature_metadata['pca']
            vector_pca = pca_model.transform([vector])[0]
            return vector_pca
        else:
            print("⚠  WARNING: No PCA model found")
            return vector

def load_efficient_xgboost_model():
    """Load the efficient XGBoost model and data"""
    print(" LOADING EFFICIENT XGBOOST MODEL...")
    try:
        with open('6feat_efficient_apt_similarity.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        print(" Loaded efficient_apt_similarity.pkl")
        return saved_data
    except FileNotFoundError:
        print(" efficient_apt_similarity.pkl not found! Please run train.py first.")
        return None

def load_apt_profiles_from_data():
    """Load APT profiles from the encoded data file"""
    print(" LOADING APT PROFILES FROM DATA...")
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
    
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles

def create_similarity_matrix_efficient(saved_data):
    """Create similarity matrix between all APTs using the efficient XGBoost model with PCA"""
    print("Creating similarity matrix using efficient XGBoost model with PCA...")
    
    # Load APT profiles
    profiles = load_apt_profiles_from_data()
    apt_ids = list(profiles.keys())
    
    # Create predictor
    predictor = EfficientXGBoostPredictor(
        saved_data['model'], 
        saved_data['apt_features'], 
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
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
    
    # Get all APT vectors (already PCA-transformed from training)
    all_apt_vectors = saved_data.get('all_apt_vectors', {})
    
    if not all_apt_vectors:
        print("⚠  WARNING: No APT vectors found in saved data")
        return None, apt_names
    
    print(f" Using {len(all_apt_vectors)} pre-computed PCA vectors for similarity calculation")
    
    # Create vectors list in the same order as apt_names
    apt_vectors = []
    for apt_id in apt_ids:
        if apt_id in all_apt_vectors:
            apt_vectors.append(all_apt_vectors[apt_id])
        else:
            # If vector not found, create one from profile
            profile = profiles[apt_id]
            evidence = {
                'techniques': list(profile['techniques']),
                'software': list(profile['software']),
                'capec': list(profile['capec']),
                'cwe': list(profile['cwe']),
                'cve': list(profile['cve']),
                'pulse': list(profile['pulse'])
            }
            vector = predictor.create_vector_from_evidence(evidence)
            apt_vectors.append(vector)
    
    # Calculate similarity matrix using PCA vectors
    num_apts = len(apt_names)
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    print("Computing pairwise similarities using cosine similarity on PCA vectors...")
    for i in range(num_apts):
        if i % 20 == 0:
            print(f"  Processed {i}/{num_apts} APTs...")
        
        for j in range(num_apts):
            similarity = predictor.predict_vector_similarity(apt_vectors[i], apt_vectors[j])
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
    
    plt.title("APT Group Similarity Matrix (Efficient XGBoost with PCA)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_efficient_xgboost_pca_similarity_matrix.png', dpi=300, bbox_inches='tight')
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
        saved_data['apt_encoder'],
        saved_data['feature_metadata']
    )
    
    # Get all APT vectors (already PCA-transformed)
    all_apt_vectors = saved_data.get('all_apt_vectors', {})
    
    # Create vectors list in the same order as apt_names
    apt_vectors = []
    for apt_id in apt_ids:
        if apt_id in all_apt_vectors:
            apt_vectors.append(all_apt_vectors[apt_id])
        else:
            # If vector not found, create one from profile
            profile = profiles[apt_id]
            evidence = {
                'techniques': list(profile['techniques']),
                'software': list(profile['software']),
                'capec': list(profile['capec']),
                'cwe': list(profile['cwe']),
                'cve': list(profile['cve']),
                'pulse': list(profile['pulse'])
            }
            vector = predictor.create_vector_from_evidence(evidence)
            apt_vectors.append(vector)
    
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
            'cwe': [],
            'cve': [],
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

        # Divide cwe
        if target_profile['cwe']:
            cwe_list = list(target_profile['cwe'])
            random.shuffle(cwe_list)
            division_size = max(1, len(cwe_list) // divider)
            divided_evidence['cwe'] = cwe_list[:division_size]

        # Divide cve
        if target_profile['cve']:
            cve_list = list(target_profile['cve'])
            random.shuffle(cve_list)
            division_size = max(1, len(cve_list) // divider)
            divided_evidence['cve'] = cve_list[:division_size]

        # Divide Pulse
        if target_profile['pulse']:
            pulse_list = list(target_profile['pulse'])
            random.shuffle(pulse_list)
            division_size = max(1, len(pulse_list) // divider)
            divided_evidence['pulse'] = pulse_list[:division_size]
        
        # Create PCA vector from divided evidence
        query_vector = predictor.create_vector_from_evidence(divided_evidence)
        
        # Calculate similarities with all APT PCA vectors
        similarities = []
        for apt_vector in apt_vectors:
            similarity = predictor.predict_vector_similarity(query_vector, apt_vector)
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
    print(" EFFICIENT XGBOOST APT SIMILARITY MODEL EVALUATION (WITH PCA)")
    print("=" * 60)
    
    # Load trained XGBoost model
    saved_data = load_efficient_xgboost_model()
    
    if saved_data is None:
        print(" Cannot proceed without trained model. Please run train.py first.")
        return
    
    print(f" Efficient XGBoost model loaded successfully!")
    print(f"   APTs in training: {saved_data['training_info']['unique_apts']}")
    print(f"   Feature vector length: {saved_data['training_info']['feature_vector_length']}")
    print(f"   Original features: {saved_data['training_info']['feature_counts']}")
    
    # Check if model uses PCA
    if 'feature_metadata' in saved_data and 'pca' in saved_data['feature_metadata']:
        print(f"   Using PCA with {saved_data['feature_metadata']['pca'].n_components_} components")
    else:
        print("⚠  WARNING: Model does not appear to use PCA")
    
    # 1. Create and display similarity matrix
    print("\n" + "="*80)
    print("CREATING EFFICIENT XGBOOST SIMILARITY MATRIX (PCA VECTORS)")
    print("="*80)
    
    similarity_df, apt_names = create_similarity_matrix_efficient(saved_data)
    
    if similarity_df is not None:
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
    else:
        print(" Failed to create similarity matrix")

if __name__ == "__main__":
    main()