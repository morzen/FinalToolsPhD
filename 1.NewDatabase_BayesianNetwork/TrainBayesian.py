import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pickle
import os

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

def prepare_bayesian_data(apt_profiles):
    data = []
    
    for apt_name, profile in apt_profiles.items():
        for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            for feature in profile[feature_type]:
                data.append({
                    'APT': apt_name,
                    'Feature_Type': feature_type,
                    'Feature': feature
                })
    
    return pd.DataFrame(data)

def create_bayesian_model(df_bayesian):
    model = DiscreteBayesianNetwork([
        ('APT', 'Feature_Type'),
        ('Feature_Type', 'Feature'),
        ('APT', 'Feature')  # Direct relationship
    ])
    
    model.fit(df_bayesian, estimator=MaximumLikelihoodEstimator)
    return model

def main():
    # Load the data
    pwd = "./DataForBayesianNetworkPython/"
    df_encoded = pd.read_csv(pwd+'data_for_bn.csv')
    df_mapping = pd.read_csv(pwd+'comparison_Encodedtest_df.csv')
    
    # Create APT profiles
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
    
    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)
    
    # Convert sets to lists and add counts
    for apt_name in apt_profiles:
        profile = apt_profiles[apt_name]
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            profile[key] = list(profile[key])
            profile[f'{key}_count'] = len(profile[key])
    
    # Prepare the data and create model
    df_bayesian = prepare_bayesian_data(apt_profiles)
    model = create_bayesian_model(df_bayesian)
    
    # Save everything in a single file
    model_data = {
        'model': model,
        'apt_profiles': apt_profiles,
        'apt_name_dict': apt_name_dict,
        'df_mapping': df_mapping
    }
    
    
    with open('./complete_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Complete model saved to {output_dir}complete_model.pkl")
    print(f"Number of APT groups: {len(apt_profiles)}")
    print(f"Bayesian network nodes: {model.nodes()}")
    print(f"Bayesian network edges: {model.edges()}")

if __name__ == "__main__":
    main()