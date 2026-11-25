import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from query import RFScorer

def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3feat_apt_similarity_model.joblib')
    return joblib.load(model_path)

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
    
    # 3. Attribution evaluation with the shared protocol (full data, shuffled & divided by 2)
    print("\n" + "="*80)
    print("ATTRIBUTION EVALUATION (full profile and shuffled & divided by 2)")
    print("="*80)
    P.evaluate_attribution(RFScorer(model_data), "Random Forest", results_dir="results",
                           smoke=P.smoke_flag())

if __name__ == "__main__":
    main()