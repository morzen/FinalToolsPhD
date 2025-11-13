import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np

class APTQueryModel(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwe, num_cve, num_pulse, hidden_dim=128):
        super().__init__()
        self.technique_embedding = nn.Embedding(num_techniques, hidden_dim, padding_idx=0)
        self.software_embedding = nn.Embedding(num_software, hidden_dim, padding_idx=0)
        self.capec_embedding = nn.Embedding(num_capec, hidden_dim, padding_idx=0)
        self.cwe_embedding = nn.Embedding(num_cwe, hidden_dim, padding_idx=0)
        self.cve_embedding = nn.Embedding(num_cve, hidden_dim, padding_idx=0)
        self.pulse_embedding = nn.Embedding(num_pulse, hidden_dim, padding_idx=0)
        
        # Combine all 6 feature types
        self.feature_combiner = nn.Linear(hidden_dim * 6, hidden_dim)
        
    def forward(self, techniques, software, capec, cwe, cves, pulse):
        tech_emb = self.technique_embedding(techniques).mean(dim=1)
        soft_emb = self.software_embedding(software).mean(dim=1)
        capec_emb = self.capec_embedding(capec).mean(dim=1)
        cwe_emb = self.cwe_embedding(cwe).mean(dim=1)
        cve_emb = self.cve_embedding(cves).mean(dim=1)
        pulse_emb = self.pulse_embedding(pulse).mean(dim=1)
        
        combined = torch.cat([tech_emb, soft_emb, capec_emb, cwe_emb, cve_emb, pulse_emb], dim=1)
        return self.feature_combiner(combined)

def query_apt_similarity(evidence):
    """Query APT similarity using the trained model"""
    print("🔍 Loading trained model...")
    
    with open('trained_apt_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    # Load the trained model
    model_config = saved_data['model_config']
    model = APTQueryModel(
        num_techniques=model_config['num_techniques'],
        num_software=model_config['num_software'],
        num_capec=model_config['num_capec'],
        num_cwe=model_config['num_cwe'],
        num_cve=model_config['num_cve'],
        num_pulse=model_config['num_pulse'],
        hidden_dim=model_config['hidden_dim']
    )
    model.load_state_dict(saved_data['model_state_dict'])
    model.eval()
    
    # Get feature mappings - note the order: techniques, software, capec, cwe, cve, pulse
    feature_mappings = saved_data['feature_mappings']
    technique_to_idx, software_to_idx, capec_to_idx, cwe_to_idx, cve_to_idx, pulse_to_idx = feature_mappings
    
    # Get APT embeddings
    apt_embeddings = saved_data['apt_embeddings']
    apt_ids = saved_data['apt_ids']
    
    # Convert evidence to indices
    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    capec = evidence.get('capec', [])
    cwe = evidence.get('cwe', [])
    cves = evidence.get('cves', [])
    pulse = evidence.get('pulse', [])
    
    tech_indices = [technique_to_idx.get(t, 0) for t in techniques] if techniques else [0]
    soft_indices = [software_to_idx.get(s, 0) for s in software] if software else [0]
    capec_indices = [capec_to_idx.get(c, 0) for c in capec] if capec else [0]
    cwe_indices = [cwe_to_idx.get(c, 0) for c in cwe] if cwe else [0]
    cve_indices = [cve_to_idx.get(c, 0) for c in cves] if cves else [0]
    pulse_indices = [pulse_to_idx.get(p, 0) for p in pulse] if pulse else [0]
    
    # Convert to tensors
    tech_tensor = torch.tensor(tech_indices, dtype=torch.long)
    soft_tensor = torch.tensor(soft_indices, dtype=torch.long)
    capec_tensor = torch.tensor(capec_indices, dtype=torch.long)
    cwe_tensor = torch.tensor(cwe_indices, dtype=torch.long)
    cve_tensor = torch.tensor(cve_indices, dtype=torch.long)
    pulse_tensor = torch.tensor(pulse_indices, dtype=torch.long)
    
    # Generate query embedding using the trained model - note the parameter order
    with torch.no_grad():
        query_embedding = model(
            tech_tensor.unsqueeze(0), 
            soft_tensor.unsqueeze(0),
            capec_tensor.unsqueeze(0),
            cwe_tensor.unsqueeze(0),
            cve_tensor.unsqueeze(0),
            pulse_tensor.unsqueeze(0)
        )
    
    # Use dot product for similarity - as the model was trained
    similarities = torch.matmul(query_embedding, apt_embeddings.t()).squeeze(0)
    
    # Sort by similarity score (highest first)
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    
    # Load APT name mappings
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row['APTnameEncoded']
        original_name = row['APTnameOrigin']
        apt_name_mapping[encoded_id] = original_name
    
    # Prepare results
    results = []
    for i, (similarity, idx) in enumerate(zip(sorted_similarities, sorted_indices)):
        apt_id = apt_ids[idx.item()]
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        
        # Get profile for match details
        profile = saved_data['profiles'][apt_id]
        tech_matches = len(set(techniques) & profile['techniques'])
        soft_matches = len(set(software) & profile['software'])
        capec_matches = len(set(capec) & profile['capec'])
        cwe_matches = len(set(cwe) & profile['cwe'])
        cve_matches = len(set(cves) & profile['cve'])
        pulse_matches = len(set(pulse) & profile['pulse'])
        
        results.append({
            'rank': i + 1,
            'apt_name': apt_name,
            'similarity': similarity.item(),
            'tech_match': f"{tech_matches}/{len(techniques)}",
            'soft_match': f"{soft_matches}/{len(software)}", 
            'capec_match': f"{capec_matches}/{len(capec)}",
            'cwe_match': f"{cwe_matches}/{len(cwe)}",
            'cve_match': f"{cve_matches}/{len(cves)}",
            'pulse_match': f"{pulse_matches}/{len(pulse)}",
            'profile_size': f"{len(profile['techniques'])}T/{len(profile['software'])}S/{len(profile['capec'])}P/{len(profile['cwe'])}W/{len(profile['cve'])}C/{len(profile['pulse'])}L"
        })
    
    return results

def print_results_table(results, evidence):
    """Print the results in a formatted table"""
    print("🔍 Loading trained model...")
    print()
    print("📋 Query Evidence:")
    print(f"   Techniques: {len(evidence['techniques'])} - {evidence['techniques']}")
    print(f"   Software: {len(evidence['software'])} - {evidence['software']}")
    print(f"   CAPEC: {len(evidence['capec'])} - {evidence['capec']}")
    print(f"   CWE: {len(evidence['cwe'])} - {evidence['cwe']}")
    print(f"   CVEs: {len(evidence['cves'])} - {evidence['cves']}")
    print(f"   Pulse: {len(evidence['pulse'])} - {evidence['pulse']}")
    print()
    print("🎯 GNN Similarity Rankings")
    print("=" * 130)
    
    # Header - updated order to match the model
    header = f"{'Rank':<4} {'APT Name':<30} {'Similarity':<10} {'Tech':<6} {'Soft':<6} {'CAPEC':<6} {'CWE':<6} {'CVE':<6} {'Pulse':<6} {'Profile'}"
    print(header)
    print("-" * 130)
    
    for result in results:
        # Truncate long APT names
        apt_name = result['apt_name']
        if len(apt_name) > 29:
            apt_name = apt_name[:26] + "..."
        
        row = (f"{result['rank']:<4} {apt_name:<30} {result['similarity']:.4f}    "
               f"{result['tech_match']:<6} {result['soft_match']:<6} {result['capec_match']:<6} "
               f"{result['cwe_match']:<6} {result['cve_match']:<6} {result['pulse_match']:<6} "
               f"{result['profile_size']}")
        print(row)
    
    print("-" * 130)

def convert_evidence_to_encoded(evidence, mapping_file="../BaseData/comparison_Encoded_df.csv"):
    """
    Convert human-readable evidence to encoded IDs using the mapping file
    """
    print("🔍 Loading mapping data...")
    df_mapping = pd.read_csv(mapping_file)
    
    # Create mapping dictionaries for all feature types
    technique_mapping = {}
    software_mapping = {} 
    capec_mapping = {}
    cwe_mapping = {}
    cve_mapping = {}
    pulse_mapping = {}
    
    for _, row in df_mapping.iterrows():
        # Technique mapping
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            if row['techniques_Origin'] not in technique_mapping:
                technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        
        # Software mapping
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            if row['software_Origin'] not in software_mapping:
                software_mapping[row['software_Origin']] = row['software_Encoded']
        
        # CAPEC mapping
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            if row['CAPEC_ID_Origin'] not in capec_mapping:
                capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']
        
        # CWE mapping
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            if row['CWE_ID_Origin'] not in cwe_mapping:
                cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']
        
        # CVE mapping
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            if row['CVE_ID_Origin'] not in cve_mapping:
                cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
        
        # Pulse mapping
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            if row['Pulse_ID_Origin'] not in pulse_mapping:
                pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']
    
    print(f"✅ Mappings loaded: {len(technique_mapping)} techniques, {len(software_mapping)} software, "
          f"{len(capec_mapping)} CAPEC, {len(cwe_mapping)} CWE, {len(cve_mapping)} CVEs, {len(pulse_mapping)} Pulse")
    
    # Convert evidence
    encoded_evidence = {
        'techniques': [],
        'software': [],
        'capec': [],
        'cwe': [],
        'cves': [],
        'pulse': []
    }
    
    # Convert techniques
    for technique in evidence.get('techniques', []):
        if technique in technique_mapping:
            encoded_evidence['techniques'].append(technique_mapping[technique])
        else:
            print(f"⚠️  Warning: Technique '{technique}' not found in mapping")
    
    # Convert software
    for software in evidence.get('software', []):
        if software in software_mapping:
            encoded_evidence['software'].append(software_mapping[software])
        else:
            print(f"⚠️  Warning: Software '{software}' not found in mapping")
    
    # Convert CAPEC
    for capec in evidence.get('capec', []):
        if capec in capec_mapping:
            encoded_evidence['capec'].append(capec_mapping[capec])
        else:
            print(f"⚠️  Warning: CAPEC '{capec}' not found in mapping")
    
    # Convert CWE
    for cwe in evidence.get('cwe', []):
        if cwe in cwe_mapping:
            encoded_evidence['cwe'].append(cwe_mapping[cwe])
        else:
            print(f"⚠️  Warning: CWE '{cwe}' not found in mapping")
    
    # Convert CVEs
    for cve in evidence.get('cves', []):
        if cve in cve_mapping:
            encoded_evidence['cves'].append(cve_mapping[cve])
        else:
            print(f"⚠️  Warning: CVE '{cve}' not found in mapping")
    
    # Convert Pulse
    for pulse in evidence.get('pulse', []):
        if pulse in pulse_mapping:
            encoded_evidence['pulse'].append(pulse_mapping[pulse])
        else:
            print(f"⚠️  Warning: Pulse '{pulse}' not found in mapping")
    
    print(f"\n📊 Conversion Results:")
    print(f"   Input techniques: {evidence.get('techniques', [])}")
    print(f"   → Encoded techniques: {encoded_evidence['techniques']}")
    print(f"   Input software: {evidence.get('software', [])}")
    print(f"   → Encoded software: {encoded_evidence['software']}")
    print(f"   Input CAPEC: {evidence.get('capec', [])}")
    print(f"   → Encoded CAPEC: {encoded_evidence['capec']}")
    print(f"   Input CWE: {evidence.get('cwe', [])}")
    print(f"   → Encoded CWE: {encoded_evidence['cwe']}")
    print(f"   Input CVEs: {evidence.get('cves', [])}")
    print(f"   → Encoded CVEs: {encoded_evidence['cves']}")
    print(f"   Input Pulse: {evidence.get('pulse', [])}")
    print(f"   → Encoded Pulse: {encoded_evidence['pulse']}")
    
    return encoded_evidence

# Main execution
if __name__ == "__main__":
    # Example evidence with all feature types (use real values from your data)
    evidence_humanreadable = {
    'techniques_str': ['T1001.002', 'T1003.002', 'T1003.004', 'T1003.006', 'T1005', 'T1016.001', 'T1018', 'T1021.001', 
                      'T1021.002', 'T1021.006', 'T1021.007', 'T1027.001', 'T1027.002', 'T1027.003', 'T1027.006', 
                      'T1036.004', 'T1036.005', 'T1037', 'T1037.004', 'T1047', 'T1048.002', 'T1053.005', 'T1057', 
                      'T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1059.009', 'T1068', 'T1069.002', 
                      'T1070.004', 'T1070.006', 'T1070.008', 'T1071.001', 'T1074.002', 'T1078', 'T1078.002', 
                      'T1078.003', 'T1078.004', 'T1082', 'T1083', 'T1087.002', 'T1087.004', 'T1090.001', 'T1090.002', 
                      'T1090.003', 'T1090.004', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.005', 'T1102.002', 
                      'T1105', 'T1110.001', 'T1110.003', 'T1114.002', 'T1133', 'T1136.003', 'T1140', 'T1190', 
                      'T1195.002', 'T1199', 'T1203', 'T1204.001', 'T1204.002', 'T1213', 'T1213.003', 'T1218.005', 
                      'T1218.011', 'T1482', 'T1484.002', 'T1505.003', 'T1528', 'T1539', 'T1546.003', 'T1546.008', 
                      'T1547.001', 'T1548.002', 'T1550.001', 'T1550.003', 'T1550.004', 'T1552.004', 'T1553.002', 
                      'T1553.005', 'T1555.003', 'T1556.007', 'T1558.003', 'T1560.001', 'T1562.001', 'T1562.002', 
                      'T1562.004', 'T1562.008', 'T1566.001', 'T1566.002', 'T1566.003', 'T1568', 'T1573', 'T1583.001', 
                      'T1583.006', 'T1584.001', 'T1585.001', 'T1586.002', 'T1586.003', 'T1587.001', 'T1587.003', 
                      'T1588.002', 'T1589.001', 'T1595.002', 'T1606.001', 'T1606.002', 'T1621', 'T1649', 'T1651', 
                      'T1665'],
    'software_str': ['S0037', 'S0559', 'S0633', 'S0002', 'S0046', 'S0050', 'S0154', 'S0357', 'S0677', 'S0048', 
                    'S0512', 'S0514', 'S0515', 'S0634', 'S0661', 'S0597', 'S0039', 'S0521', 'S0552', 'S0684', 
                    'S0029', 'S0053', 'S0513', 'S0565', 'S0588', 'S0139', 'S0511', 'S0518', 'S0560', 'S0562', 
                    'S0589', 'S1084', 'S0516', 'S0049', 'S0057', 'S0150', 'S0195', 'S0051', 'S0052', 'S0054', 
                    'S0635', 'S0636', 'S0682', 'S0096', 'S0183', 'S0175', 'S0637', 'S0100'],
    'CAPEC_ID': ['CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-292', 'CAPEC-561', 'CAPEC-572', 'CAPEC-655', 
                'CAPEC-35', 'CAPEC-504', 'CAPEC-543', 'CAPEC-616', 'CAPEC-564', 'CAPEC-573', 'CAPEC-560', 'CAPEC-70', 
                'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-481', 'CAPEC-49', 
                'CAPEC-565', 'CAPEC-555', 'CAPEC-186', 'CAPEC-443', 'CAPEC-523', 'CAPEC-669', 'CAPEC-673', 'CAPEC-678', 
                'CAPEC-691', 'CAPEC-150', 'CAPEC-650', 'CAPEC-21', 'CAPEC-31', 'CAPEC-558', 'CAPEC-593', 'CAPEC-645', 
                'CAPEC-60', 'CAPEC-474', 'CAPEC-485', 'CAPEC-639', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-509', 
                'CAPEC-578', 'CAPEC-268', 'CAPEC-571', 'CAPEC-163', 'CAPEC-542', 'CAPEC-464'],
    'CWE_ID': ['CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 'CWE-1278', 'CWE-285', 'CWE-200', 
              'CWE-522', 'CWE-294', 'CWE-521', 'CWE-94', 'CWE-272', 'CWE-59', 'CWE-282', 'CWE-270', 'CWE-1021', 
              'CWE-284', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 
              'CWE-693', 'CWE-441', 'CWE-923', 'CWE-257', 'CWE-494', 'CWE-552', 'CWE-287', 'CWE-290', 'CWE-302', 
              'CWE-346', 'CWE-539', 'CWE-384', 'CWE-664', 'CWE-602', 'CWE-113', 'CWE-20', 'CWE-472', 'CWE-488', 
              'CWE-330', 'CWE-327', 'CWE-325', 'CWE-117', 'CWE-451', 'CWE-359'],
    'CVE_ID': ['CVE-2021-44228', 'toto'],  # Kept your example CVE
    'Pulse_ID': ['caca', '6608297f107807a45452733a']
    }

    evidence = convert_evidence_to_encoded(evidence_humanreadable)
    
    print("🚀 Running GNN APT Similarity Query")
    print("=" * 70)
    
    # Perform the query
    results = query_apt_similarity(evidence)
    
    # Print the results
    print_results_table(results, evidence)
    
    print(f"\n✅ Done! Ranked {len(results)} APTs from most similar to least similar")
    print(f"🏆 Most similar: {results[0]['apt_name']} (score: {results[0]['similarity']:.4f})")
    print(f"📊 Similarity range: {results[-1]['similarity']:.4f} to {results[0]['similarity']:.4f}")