import pandas as pd


def process_apt_data(csv_file_path, apt_name):
    """
    Process CSV file and extract all 7 features for a specific APT
    """
    # Read CSV data from file
    df = pd.read_csv(csv_file_path)
    
    # Filter data for the specified APT
    apt_data = df[df['APTname'] == apt_name]
    
    # Extract unique techniques (excluding empty values and -1)
    techniques = sorted(apt_data['techniques_str'].dropna().unique().tolist())
    
    # Extract unique software (excluding empty values and -1)
    software = sorted(apt_data['software_str'].dropna().unique().tolist())
    
    # Extract unique CAPEC_ID (excluding empty values and -1)
    capec = sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CWE_ID (excluding empty values and -1)
    cwe = sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CVEs (excluding empty values and -1)
    cves = sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique Pulse_ID (excluding empty values and -1)
    pulse = sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    
    return {
        'techniques': techniques,
        'software': software,
        'capec': capec,
        'cwe': cwe,
        'cves': cves,
        'pulse': pulse
    }


def main():
    # Get CSV file path from user
    csv_file_path = "./BaseData/TestFinal_df.csv"
    
    # Get APT name from user
    apt_name = "APT29"
    
    try:
        # Process the data
        result = process_apt_data(csv_file_path, apt_name)
        
        import json

        # Build a dictionary with all the data
        output = {
            "techniques": result["techniques"],
            "software": result["software"],
            "capec": result["capec"],
            "cwe": result["cwe"],
            "cve": result["cves"],
            "pulse": result["pulse"]
        }

        with open("query_input.json", "w") as f:
            json.dump(output, f, indent=4)
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
