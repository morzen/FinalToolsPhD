import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import random

def load_model():
    """Load the pre-trained model and all necessary data"""
    model_data = joblib.load('apt_similarity_model.joblib')
    print("Model loaded successfully")
    return model_data

def create_similarity_matrix(rf_model, X, apt_names, apt_name_dict, apt_profiles):
    """Create similarity matrix between all APTs using Random Forest"""
    # Get leaf indices for all APTs
    leaf_indices = []
    for tree in rf_model.estimators_:
        leaf_indices.append(tree.apply(X))
    leaf_indices = np.array(leaf_indices)
    
    # Compute similarity matrix
    similarity_matrix = np.zeros((len(apt_names), len(apt_names)))
    for i in range(len(apt_names)):
        for j in range(len(apt_names)):
            same_leaves = np.sum(leaf_indices[:, i] == leaf_indices[:, j])
            similarity_matrix[i, j] = same_leaves / len(rf_model.estimators_)
    
    # Get APT names in order
    apt_display_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_names]
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=apt_display_names,
        columns=apt_display_names
    )
    
    return similarity_df

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
    
    plt.title("APT Group Similarity Matrix (Random Forest)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.savefig('apt_RF_featureEngineering_heatmap_matrix.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def evaluate_with_similarity_matrix(similarity_df, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance using the precomputed similarity matrix"""
    print("\n" + "="*80)
    print("SIMILARITY MATRIX EVALUATION (NO DATA DIVISION)")
    print("="*80)
    
    apt_names = similarity_df.index.tolist()
    resultdict = {}
    
    print("Using complete similarity matrix without data division...")
    
    # For each APT, get its similarity scores from the matrix
    for target_apt_name in apt_names:
        # Get similarity scores for this APT from the matrix
        similarities = similarity_df[target_apt_name].to_dict()
        resultdict[target_apt_name] = similarities
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
            else:
                failure = failure + 1
        
        print("\nsettings: Topx: " + str(topX))
        print("Random Forest for this set had a success failure rate of ")
        print(str(success) + "/" + str(failure) + " Total: " + str(lengthResultDict))
        print("in percentage success rate: " + str((success/lengthResultDict)*100))

def evaluate_profile_division(model_data, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance with divided profile data"""
    # Extract components from model data
    apt_profiles = model_data['apt_profiles']
    apt_name_dict = model_data['apt_name_dict']
    apt_ids = model_data['apt_names']
    df_mapping = model_data['df_mapping']
    binarizers = model_data['binarizers']
    all_features = model_data['all_features']
    feature_types = model_data['feature_types']
    rf = model_data['rf_model']
    X = model_data['X']
    
    resultdict = {}
    apt_display_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    
    print(f"\nEvaluating with divider: {divider} (using {100/divider:.1f}% of profile data)")
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_display_names):
        # Get the target APT's original data
        target_apt_id = next(apt for apt in apt_ids if apt_name_dict[apt_profiles[apt]['encoded_id']] == target_apt_name)
        target_data = apt_profiles[target_apt_id]
        
        # Create evidence with divided data for each feature type
        divided_evidence = {}
        
        for feature_type, evidence_key in [('techniques', 'techniques_str'),
                                         ('software', 'software_str'),
                                         ('cve', 'CVE_ID'),]:
            
            if feature_type in target_data and target_data[feature_type]:
                feature_list = target_data[feature_type].copy()
                random.shuffle(feature_list)
                division_size = max(1, len(feature_list) // divider)
                divided_features = feature_list[:division_size]
                
                # Convert encoded features back to original format for sanitization
                if feature_type == 'techniques':
                    mapping = dict(zip(df_mapping['techniques_Encoded'], df_mapping['techniques_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'software':
                    mapping = dict(zip(df_mapping['software_Encoded'], df_mapping['software_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'cve':
                    mapping = dict(zip(df_mapping['CVE_ID_Encoded'], df_mapping['CVE_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
        
        # Process the divided evidence
        clean_evidence = sanitize_evidence(divided_evidence, df_mapping)
        encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
        
        # Create evidence vector and calculate RF similarity
        evidence_vector = create_evidence_vector(encoded_evidence, binarizers, all_features, feature_types)
        rf_similarities = calculate_rf_similarity(evidence_vector, rf, X, apt_ids)
        
        # Create ranked results
        ranked_results = {}
        sorted_indices = np.argsort(rf_similarities)[::-1]
        for idx in sorted_indices:
            apt_name = apt_ids[idx]
            apt_real_name = apt_name_dict[apt_profiles[apt_name]['encoded_id']]
            ranked_results[apt_real_name] = rf_similarities[idx]
        
        resultdict[target_apt_name] = ranked_results
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SucessFailDict = {}
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
                SucessFailDict[target_apt] = "Pass"
            else:
                failure = failure + 1
                SucessFailDict[target_apt] = "Fail"
        
        print("\nsettings: Topx: " + str(topX) + " Divider: " + str(divider))
        print("Random Forest for this set had a success failure rate of ")
        print(str(success) + "/" + str(failure) + " Total: " + str(lengthResultDict))
        print("in percentage success rate: " + str((success/lengthResultDict)*100))

def sanitize_evidence(evidence, df_mapping):
    """Keep only evidence features that exist in our database"""
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Origin', 'techniques_Encoded'),
        'software_str': ('software_Origin', 'software_Encoded'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded')
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
    
    if missing:
        print("WARNING: These features were not found in our database:")
        for feature_type, items in missing.items():
            print(f"- {feature_type}: {items[:5]}")
            if len(items) > 5:
                print(f"  (and {len(items)-5} more)")
    
    return sanitized

def translate_to_encoded(evidence, df_mapping):
    """Convert sanitized evidence to encoded IDs"""
    encoded_evidence = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Encoded', 'techniques'),
        'software_str': ('software_Encoded', 'software'), 
        'CVE_ID': ('CVE_ID_Encoded', 'cve'),
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

def create_evidence_vector(encoded_evidence, binarizers, all_features, feature_types):
    """Create a feature vector for the evidence"""
    evidence_vector = []
    
    for ft in feature_types:
        if ft in encoded_evidence:
            valid_features = [f for f in encoded_evidence[ft] if f in all_features[ft]]
            binary_vec = binarizers[ft].transform([valid_features]).flatten()
        else:
            binary_vec = np.zeros(len(all_features[ft]))
        evidence_vector.extend(binary_vec)
    
    return np.array(evidence_vector)

def calculate_rf_similarity(evidence_vector, rf_model, X, apt_names):
    """Calculate similarity scores using Random Forest"""
    evidence_leaves = np.array([tree.apply(evidence_vector.reshape(1, -1)) for tree in rf_model.estimators_]).flatten()
    
    apt_leaves = []
    for tree in rf_model.estimators_:
        apt_leaves.append(tree.apply(X))
    apt_leaves = np.array(apt_leaves)
    
    similarities = np.mean(apt_leaves == evidence_leaves.reshape(-1, 1), axis=0)
    
    return similarities

def main():
    # Load the model
    model_data = load_model()
    
    # Extract components from model data
    rf = model_data['rf_model']
    apt_profiles = model_data['apt_profiles']
    apt_names = model_data['apt_names']
    apt_name_dict = model_data['apt_name_dict']
    X = model_data['X']
    df_mapping = model_data['df_mapping']
    binarizers = model_data['binarizers']
    all_features = model_data['all_features']
    feature_types = model_data['feature_types']
    
    # 1. Create and display similarity matrix for all APTs
    print("Creating similarity matrix for all APTs...")
    similarity_df = create_similarity_matrix(rf, X, apt_names, apt_name_dict, apt_profiles)
    
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 2. Plot heatmap
    print("Plotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # 3. Evaluate using the similarity matrix (no data division)
    evaluate_with_similarity_matrix(similarity_df)
    
    # 4. Evaluate with profile division (divider = 2)
    print("\n" + "="*80)
    print("PROFILE DIVISION EVALUATION")
    print("="*80)
    evaluate_profile_division(model_data, divider=2)

if __name__ == "__main__":
    main()