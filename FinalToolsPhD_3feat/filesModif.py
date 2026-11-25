import pandas as pd
import os

def remove_columns_and_duplicates_from_csv_files(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for filename in csv_files[:3]:  # Process only first 3 CSV files
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        print(f"\nProcessing {filename}:")
        print(f"Original shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        print(f"All columns: {list(df.columns)}")  # Debug: Show all columns
        
        # STEP 1: Remove columns containing CWE, CAPEC, or Pulse_ID
        columns_to_remove = []
        for col in df.columns:
            col_upper = col.upper()
            if any(keyword in col_upper for keyword in ['CWE', 'CAPEC', 'PULSE_ID']):
                columns_to_remove.append(col)
                print(f"  Found column to remove: '{col}' -> matches pattern")
        
        # Remove the columns
        df_cleaned = df.drop(columns=columns_to_remove)
        
        print(f"After column removal: {df_cleaned.shape}")
        if columns_to_remove:
            print(f"Removed columns: {columns_to_remove}")
        else:
            print("No columns matched the removal criteria")
        
        # STEP 2: Now remove duplicate rows (after column removal)
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        final_rows = len(df_cleaned)
        duplicates_removed = initial_rows - final_rows
        
        print(f"After duplicate removal: {df_cleaned.shape}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        # Save back to the same file (overwrite)
        df_cleaned.to_csv(file_path, index=False)
        
        print(f"✅ Final file saved: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

# Usage
folder_path = "./BaseData"  # Replace with your folder path
remove_columns_and_duplicates_from_csv_files(folder_path)