import random

# Your data
Apt1 = {
    'techniques': ['T1003.001', 'T1557.002', 'T1585.001', 'T1587.001', 'T1588.002'],
    'software': ['S0002', 'S0004', 'S0029', 'S0056'],
    'capec': ['CAPEC-141', 'CAPEC-542'],
    'cwe': ['CWE-345', 'CWE-346', 'CWE-348', 'CWE-349'],
    'cve': ['CVE-2020-11985', 'CVE-2021-34572', 'CVE-2021-4122', 'CVE-2022-0031', 'CVE-2022-20774', 
            'CVE-2022-20829', 'CVE-2022-23491', 'CVE-2022-24889', 'CVE-2022-29818', 'CVE-2022-31024', 
            'CVE-2022-31800', 'CVE-2022-31801', 'CVE-2022-31813', 'CVE-2023-30996', 'CVE-2023-36858', 
            'CVE-2023-37920', 'CVE-2024-25638', 'CVE-2024-37968', 'CVE-2024-38198', 'CVE-2024-39689'],
    'pulse': ['61fa07500dac0edcaf418234', '6220da17f90c0344acc97b20', '622fd6e4d2592fc24c87e51d', 
              '62567f58c6ed12db3b5c481d', '62567f5d2e19b24473628e48', '6271815d5b43a4e4c1dbdd72', 
              '62a21c26ad29a430b9bbefc5', '62a21c44d0d5d87730e39790', '62ab365e312017ee94dae0dd', 
              '62b5ec2bbedb4d41f8dc7b53', '62badb2a0b18b533af313efe', '62bae377144bc36fdcac6cc3', 
              '62bae9bdfba0afd14c1df5a6', '635ee3d4be0a6941c2005ba2', '6393454b38c36548afd75fc7', 
              '63ce9b3fac3f789aa8273ac7', '64bace8d5d470a6a7baaa8a9', '64ef9a4ff737e32c25134189', 
              '6515d344918d6a84587478f5', '65da2e10a6bc9c007c0c0e32', '664cc3c1938ed0b9abf026a2', 
              '66a1221a4c2d00084ba3bcba', '66bf67dab92d1f0b87fd0417', '66e5b6d960be973dc05ca4d3', 
              '672f88f9190a6de9940c4e58', '6769dd2ac0f748c372512eef', '67c6bb5aa601e91b1314ff44', 
              '67d031cf1fa149f30e59fe81', '6807e1c45dd52410facbed8f']
}

Apt2 = {
    'techniques': ['T1102.002', 'T1203', 'T1204.002', 'T1566.001', 'T1568.003'],
    'software': ['S0003', 'S0015', 'S0040'],
    'capec': ['CAPEC-163'],
    'cwe': ['CWE-451'],
    'cve': ['CVE-2021-33593', 'CVE-2024-38112', 'CVE-2024-38197', 'CVE-2024-43461', 'CVE-2024-49040'],
    'pulse': ['664cc3c1938ed0b9abf026a2', '668e5353bb297021f2c812f9', '668e6bd1b7048a19975989d9', 
              '66911f5e1df065e410143bab', '6694a8b5c6e74d011d365eaf', '6694c387cdb8ddd8c1baad16', 
              '6696401c81cae41e046a6a6e', '6696439162b61a376502d01c', '669764c71040503ba11cefc3', 
              '66bf67dab92d1f0b87fd0417', '66e1a584c1f2be27f7d11527', '66e32107ddfe23ab0f47195f', 
              '66e42fc62878928200520cb6', '66e834ca12da67c3ffc6831f', '66ec1d053abb8d066f52e871', 
              '66f3d4dff860d18695313fd6', '672d552ee02b6697ba5e6474', '6734c7ed9f5c0b30bce7ad42', 
              '67380eff4b589d9ae37a68bd', '676652f6a7ad2f6c63bac5d6', '676652f7393d00df56cffdc0', 
              '676652f78bbff8f3a7dcb4c3', '676652f7a55e1301c9233e6c', '676652f7b1080740c101e9e9', 
              '6788b50e7be5db04f049def4', '67aa6fdc9d11e3813a040447', '67c6bb5aa601e91b1314ff44']
}




import random

# Set random seed for reproducibility
random.seed(42)

def combine_apts(apt1, apt2, apt1_ratio=0.3, apt2_ratio=0.7):
    combined = {}
    
    # Process each category
    for category in apt1.keys():
        combined[category] = []
        
        # Sample from APT1 (high → low - take from beginning)
        apt1_items = apt1[category]
        apt1_count = int(len(apt1_items) * apt1_ratio)
        # Take from start for high → low
        apt1_sample = apt1_items[:apt1_count]
        
        # Sample from APT2 (low → high - take from end)
        apt2_items = apt2[category]
        apt2_count = int(len(apt2_items) * apt2_ratio)
        # Take from end for low → high
        apt2_sample = apt2_items[-apt2_count:] if apt2_count > 0 else []
        
        # Combine and deduplicate while preserving order
        combined_category = []
        seen = set()
        
        for item in apt1_sample + apt2_sample:
            if item not in seen:
                seen.add(item)
                combined_category.append(item)
        
        combined[category] = combined_category
    
    return combined

# Combine the APTs
combined_apt = combine_apts(Apt1, Apt2)

