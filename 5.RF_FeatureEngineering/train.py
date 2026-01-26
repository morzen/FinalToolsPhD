"""
STANDARDIZED RANDOM FOREST FOR APT SIMILARITY
=============================================
Author: [Your Name]
Date: [Current Date]
Description: Random Forest classifier for APT similarity detection
"""

# ============================
# 1. IMPORTS
# ============================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pickle

# ============================
# 2. CONFIGURATION
# ============================
class RandomForestConfig:
    """Configuration for Random Forest model"""
#variable shared by all models:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "apt_similarity_model.joblib"


#variables share by some models:
    MAPPING_PATH = "../BaseData/comparison_Encoded_df.csv"
    RANDOM_STATE = 42

#variables unique to this model:
    # Model parameters
    N_ESTIMATORS = 100
    N_JOBS = -1  # Use all available cores
    # Feature types
    FEATURE_TYPES = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']


# ============================
# 3. DATA LOADING & MAPPING
# ============================
class RFDataLoader:
    """Handles data loading and APT name mapping"""
    
    def __init__(self):
        self.df_encoded = None
        self.df_mapping = None
        self.apt_name_dict = {}
        
    def load_data(self):
        """Load both encoded data and mapping files"""
        print("Loading data files...")
        self.df_encoded = pd.read_csv(RandomForestConfig.DATA_PATH)
        self.df_mapping = pd.read_csv(RandomForestConfig.MAPPING_PATH)
        
        print(f"Encoded data: {len(self.df_encoded)} records")
        print(f"Mapping data: {len(self.df_mapping)} records")
        
        return self.df_encoded, self.df_mapping
    
    def create_apt_name_mapping(self):
        """Create mapping from encoded IDs to original APT names"""
        print("\nCreating APT name mapping...")
        
        apt_name_mapping = self.df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
        self.apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], 
                                      apt_name_mapping['APTnameOrigin']))
        
        print(f"Created mapping for {len(self.apt_name_dict)} APTs")
        return self.apt_name_dict

# ============================
# 4. APT PROFILE CREATION
# ============================
class APTProfileCreator:
    """Creates comprehensive APT profiles"""
    
    def __init__(self, apt_name_dict):
        self.apt_name_dict = apt_name_dict
        self.profiles = {}
    
    def create_profiles_from_dataframe(self, df_encoded):
        """Create APT profiles from encoded dataframe"""
        print("\nCreating APT profiles...")
        
        for _, row in df_encoded.iterrows():
            apt_encoded = row['APTname']
            apt_name = self.apt_name_dict.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
            
            if apt_name not in self.profiles:
                self.profiles[apt_name] = {
                    'encoded_id': apt_encoded,
                    'techniques': set(),
                    'software': set(),
                    'capec': set(),
                    'cwe': set(),
                    'cve': set(),
                    'pulse': set()
                }
            
            # Add features if present
            self._add_feature_to_profile(self.profiles[apt_name], 'techniques', row['techniques_str'])
            self._add_feature_to_profile(self.profiles[apt_name], 'software', row['software_str'])
            self._add_feature_to_profile(self.profiles[apt_name], 'capec', row['CAPEC_ID'])
            self._add_feature_to_profile(self.profiles[apt_name], 'cwe', row['CWE_ID'])
            self._add_feature_to_profile(self.profiles[apt_name], 'cve', row['CVE_ID'])
            self._add_feature_to_profile(self.profiles[apt_name], 'pulse', row['Pulse_ID'])
        
        self._finalize_profiles()
        self._print_profile_statistics()
        
        return self.profiles
    
    def _add_feature_to_profile(self, profile, feature_name, value):
        """Add feature value to profile if valid"""
        if value != -1:
            profile[feature_name].add(value)
    
    def _finalize_profiles(self):
        """Convert sets to lists and add counts"""
        for apt_name in self.profiles:
            profile = self.profiles[apt_name]
            for feature_type in RandomForestConfig.FEATURE_TYPES:
                profile[feature_type] = list(profile[feature_type])
                profile[f'{feature_type}_count'] = len(profile[feature_type])
    
    def _print_profile_statistics(self):
        """Print statistics about the created profiles"""
        print(f"\nProfile Statistics:")
        print("=" * 50)
        print(f"Total APTs: {len(self.profiles)}")
        
        for feature_type in RandomForestConfig.FEATURE_TYPES:
            total_features = sum(len(p[feature_type]) for p in self.profiles.values())
            print(f"Total {feature_type}: {total_features}")
        
        # Show sample profiles
        print("\nSample Profiles (first 3 APTs):")
        sample_apts = list(self.profiles.keys())[:3]
        for apt_name in sample_apts:
            profile = self.profiles[apt_name]
            print(f"\n{apt_name}:")
            for feature_type in RandomForestConfig.FEATURE_TYPES:
                count = len(profile[feature_type])
                sample = profile[feature_type][:3] if count > 0 else []
                print(f"  {feature_type}: {count} features, sample: {sample}")

# ============================
# 5. FEATURE ENGINEERING
# ============================
class FeatureEngineer:
    """Handles feature engineering for Random Forest"""
    
    def __init__(self):
        self.binarizers = {ft: MultiLabelBinarizer() for ft in RandomForestConfig.FEATURE_TYPES}
        self.all_features = {}
        self.apt_features = {}
    
    def prepare_feature_binarizers(self, apt_profiles):
        """Prepare binarizers for each feature type"""
        print("\nPreparing feature binarizers...")
        
        for feature_type in RandomForestConfig.FEATURE_TYPES:
            # Collect all features of this type
            features = set()
            for profile in apt_profiles.values():
                features.update(profile[feature_type])
            
            self.all_features[feature_type] = sorted(features)
            self.binarizers[feature_type].fit([self.all_features[feature_type]])
            
            print(f"  {feature_type}: {len(self.all_features[feature_type])} unique features")
        
        return self.binarizers, self.all_features
    
    def create_feature_vectors(self, apt_profiles):
        """Create binary feature vectors for each APT"""
        print("\nCreating feature vectors...")
        
        apt_names = list(apt_profiles.keys())
        
        for apt_name in apt_names:
            profile = apt_profiles[apt_name]
            feature_vector = []
            
            for feature_type in RandomForestConfig.FEATURE_TYPES:
                # Binarize the features
                binary_vec = self.binarizers[feature_type].transform([profile[feature_type]]).flatten()
                feature_vector.extend(binary_vec)
            
            self.apt_features[apt_name] = feature_vector
        
        print(f"Created feature vectors for {len(self.apt_features)} APTs")
        print(f"Feature vector length: {len(self.apt_features[apt_names[0]])}")
        
        return self.apt_features
    
    def create_feature_matrix(self, apt_profiles, apt_features):
        """Create feature matrix X and target vector y"""
        apt_names = list(apt_profiles.keys())
        
        X = np.array([apt_features[name] for name in apt_names])
        y = np.arange(len(apt_names))  # Use indices as labels
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        print(f"Target Vector Shape: {y.shape}")
        
        return X, y, apt_names

# ============================
# 6. MODEL TRAINING
# ============================
class RandomForestTrainer:
    """Handles Random Forest model training"""
    
    def __init__(self, n_estimators=RandomForestConfig.N_ESTIMATORS, 
                 random_state=RandomForestConfig.RANDOM_STATE):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=RandomForestConfig.N_JOBS
        )
    
    def train(self, X, y):
        """Train the Random Forest model"""
        print(f"\nTraining Random Forest model...")
        print(f"  Number of estimators: {self.model.n_estimators}")
        print(f"  Number of features: {X.shape[1]}")
        print(f"  Number of samples: {X.shape[0]}")
        
        self.model.fit(X, y)
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)
        print(f"  Training accuracy: {train_accuracy:.4f}")
        
        return self.model

# ============================
# 7. MODEL SAVING
# ============================
class RFModelSaver:
    """Handles saving of Random Forest model"""
    
    @staticmethod
    def save_model(rf_model, binarizers, all_features, apt_profiles, 
                   apt_names, apt_name_dict, X, df_mapping, 
                   filename=RandomForestConfig.MODEL_PATH):
        """Save complete Random Forest model with all data"""
        print(f"\nSaving model to '{filename}'...")
        
        model_data = {
            'rf_model': rf_model,
            'binarizers': binarizers,
            'all_features': all_features,
            'apt_profiles': apt_profiles,
            'apt_names': apt_names,
            'apt_name_dict': apt_name_dict,
            'X': X,
            'feature_types': RandomForestConfig.FEATURE_TYPES,
            'df_mapping': df_mapping,
            'config': {
                'n_estimators': RandomForestConfig.N_ESTIMATORS,
                'random_state': RandomForestConfig.RANDOM_STATE,
                'model_type': 'RandomForestClassifier'
            }
        }
        
        joblib.dump(model_data, filename)
        
        print(f"Model saved successfully!")
        print(f"  APT profiles: {len(apt_profiles)}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Feature types: {RandomForestConfig.FEATURE_TYPES}")
        
        return model_data

# ============================
# 8. MAIN EXECUTION
# ============================
def main():
    """Execute the complete Random Forest training pipeline"""
    print("RANDOM FOREST FOR APT SIMILARITY")
    print("=" * 60)
    
    # Step 1: Load data and create mappings
    print("\n1. LOADING DATA")
    data_loader = RFDataLoader()
    df_encoded, df_mapping = data_loader.load_data()
    apt_name_dict = data_loader.create_apt_name_mapping()
    
    # Step 2: Create APT profiles
    print("\n2. CREATING APT PROFILES")
    profile_creator = APTProfileCreator(apt_name_dict)
    apt_profiles = profile_creator.create_profiles_from_dataframe(df_encoded)
    
    # Step 3: Feature engineering
    print("\n3. FEATURE ENGINEERING")
    feature_engineer = FeatureEngineer()
    binarizers, all_features = feature_engineer.prepare_feature_binarizers(apt_profiles)
    apt_features = feature_engineer.create_feature_vectors(apt_profiles)
    X, y, apt_names = feature_engineer.create_feature_matrix(apt_profiles, apt_features)
    
    # Step 4: Train Random Forest model
    print("\n4. MODEL TRAINING")
    trainer = RandomForestTrainer()
    rf_model = trainer.train(X, y)
    
    # Step 5: Save model
    print("\n5. MODEL SAVING")
    model_data = RFModelSaver.save_model(
        rf_model, binarizers, all_features, apt_profiles,
        apt_names, apt_name_dict, X, df_mapping
    )
    
    print("\nTRAINING SUMMARY")
    print("=" * 60)
    print(f"• Model: Random Forest Classifier")
    print(f"• Number of estimators: {RandomForestConfig.N_ESTIMATORS}")
    print(f"• Number of APTs: {len(apt_profiles)}")
    print(f"• Feature vector length: {X.shape[1]}")
    print(f"• Training accuracy: {rf_model.score(X, y):.4f}")
    print(f"• Saved to: {RandomForestConfig.MODEL_PATH}")
    
    return model_data

if __name__ == "__main__":
    saved_data = main()