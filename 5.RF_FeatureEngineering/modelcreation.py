import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pickle

def create_and_save_model():
    pwd = "./DataForRF/"
    
    # Load both files
    df_encoded = pd.read_csv(pwd+'data_for_RF.csv')
    df_mapping = pd.read_csv(pwd+'comparison_Encodedtest_df.csv')

    # Step 1.1: Create a mapping dictionary for APT names
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))

    # Step 1.2: Create comprehensive profiles using actual APT names as keys
    def create_apt_profiles_with_names(df_encoded, name_mapping):
        profiles = {}
        
        for _, row in df_encoded.iterrows():
            apt_encoded = row['APTname']
            apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
            
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

    # Create feature vectors for each APT
    feature_types = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']

    # Initialize binarizers for each feature type
    binarizers = {ft: MultiLabelBinarizer() for ft in feature_types}

    # Get all possible features for each type
    all_features = {}
    for ft in feature_types:
        features = set()
        for apt in apt_profiles.values():
            features.update(apt[ft])
        all_features[ft] = sorted(features)
        binarizers[ft].fit([all_features[ft]])

    # Create feature vectors for each APT
    apt_features = {}
    for apt_name, profile in apt_profiles.items():
        feature_vector = []
        for ft in feature_types:
            binary_vec = binarizers[ft].transform([profile[ft]]).flatten()
            feature_vector.extend(binary_vec)
        apt_features[apt_name] = feature_vector

    # Create a matrix of all APT features
    apt_names = list(apt_profiles.keys())
    X = np.array([apt_features[name] for name in apt_names])
    y = np.arange(len(apt_names))

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Prepare all data needed for querying
    model_data = {
        'rf_model': rf,
        'binarizers': binarizers,
        'all_features': all_features,
        'apt_profiles': apt_profiles,
        'apt_names': apt_names,
        'apt_name_dict': apt_name_dict,
        'X': X,
        'feature_types': feature_types,
        'df_mapping': df_mapping  # Keep mapping for evidence processing
    }

    # Save the complete model as a single file
    joblib.dump(model_data, 'apt_similarity_model.joblib')
    print("Model saved successfully as 'apt_similarity_model.joblib'")
    
    return model_data

if __name__ == "__main__":
    create_and_save_model()