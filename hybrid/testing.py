import numpy as np
import numpy as np

def hybrid_rank_simple(lists):
    """
    Average ranks from dictionaries with 'apt_name' and 'rank' keys
    """
    # Collect all names and their ranks from each list
    rank_data = {}
    
    for list_idx, ranking in enumerate(lists):
        for item in ranking:  # Each item is a dictionary
            name = item['apt_name']
            rank = item['rank']
            if name not in rank_data:
                rank_data[name] = []
            rank_data[name].append(rank)
    
    # Calculate average and prepare for sorting
    name_avg = []
    for name, ranks in rank_data.items():
        avg_rank = sum(ranks) / len(ranks)
        name_avg.append((name, avg_rank))
    
    # Sort by average rank, then by name
    name_avg.sort(key=lambda x: (x[1], x[0]))
    
    return name_avg


def hybrid_rank_weighted(lists, weights=None):
    """
    Weighted average with confidence score from dictionaries
    """
    if weights is None:
        weights = [1] * len(lists)
    
    rank_data = {}
    
    for list_idx, ranking in enumerate(lists):
        weight = weights[list_idx]
        
        for item in ranking:  # Each item is a dictionary
            name = item['apt_name']
            rank = item['rank']
            
            if name not in rank_data:
                rank_data[name] = {'ranks': [], 'weights': []}
            
            rank_data[name]['ranks'].append(rank)
            rank_data[name]['weights'].append(weight)
    
    # Calculate weighted average and confidence
    results = []
    for name, data in rank_data.items():
        weighted_sum = sum(r * w for r, w in zip(data['ranks'], data['weights']))
        total_weight = sum(data['weights'])
        avg_rank = weighted_sum / total_weight
        
        # Confidence: lower std dev = higher confidence
        std_dev = np.std(data['ranks']) if len(data['ranks']) > 1 else 0
        
        results.append((name, avg_rank, std_dev))
    
    # Sort by avg rank, then by std dev (lower is better), then by name
    results.sort(key=lambda x: (x[1], x[2], x[0]))
    
    return [(name, avg) for name, avg, _ in results]


def hybrid_rank_borda(lists):
    """
    Borda count method - points based on position, not rank value
    """
    n_items = len(lists[0])
    borda_points = {}
    
    for ranking in lists:
        max_points = n_items
        for position, item in enumerate(ranking):  # position is 0-based
            name = item['apt_name']
            points = max_points - position  # Higher points for better rank
            if name not in borda_points:
                borda_points[name] = 0
            borda_points[name] += points
    
    # Convert to list and sort
    results = [(name, points) for name, points in borda_points.items()]
    results.sort(key=lambda x: (-x[1], x[0]))  # Descending points
    
    # Normalize to rank positions
    final_ranked = []
    for i, (name, _) in enumerate(results, 1):
        final_ranked.append((name, i))
    
    return final_ranked


def hybrid_rank_aggregate(lists):
    """
    Advanced rank aggregation with explicit tie-breaking
    """
    n_lists = len(lists)
    
    # Create rank matrix
    rank_matrix = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            if name not in rank_matrix:
                rank_matrix[name] = [None] * n_lists
            rank_matrix[name][list_idx] = rank
    
    # Calculate statistics
    results = []
    for name, ranks in rank_matrix.items():
        ranks = [r for r in ranks if r is not None]
        avg_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        min_rank = min(ranks)
        max_rank = max(ranks)
        rank_range = max_rank - min_rank
        
        results.append({
            'name': name,
            'avg': avg_rank,
            'median': median_rank,
            'min': min_rank,
            'range': rank_range
        })
    
    # Sort with multiple tie-breakers
    results.sort(key=lambda x: (
        x['avg'],           # Primary: average rank
        x['median'],        # Secondary: median
        x['min'],          # Tertiary: best rank
        x['range'],        # Quaternary: consistency
        x['name']          # Final: alphabetical
    ))
    
    return [(item['name'], i+1) for i, item in enumerate(results)]


def hybrid_rank_rrf(lists, k=60):
    """
    Reciprocal rank Fusion - handles ties elegantly
    """
    rrf_scores = {}
    
    for ranking in lists:
        for position, item in enumerate(ranking, 1):  # 1-based position
            name = item['apt_name']
            if name not in rrf_scores:
                rrf_scores[name] = 0
            rrf_scores[name] += 1 / (position + k)
    
    # Sort by RRF score (higher is better)
    results = [(name, score) for name, score in rrf_scores.items()]
    results.sort(key=lambda x: (-x[1], x[0]))
    
    # Convert to rank positions
    final_ranked = []
    current_rank = 1
    prev_score = None
    
    for i, (name, score) in enumerate(results, 1):
        if prev_score is not None and abs(score - prev_score) < 0.0001:
            final_ranked.append((name, current_rank))
        else:
            current_rank = i
            final_ranked.append((name, i))
        prev_score = score
    
    return final_ranked


# Example with your actual data structure:
list1 = [
    {'rank': 1, 'apt_name': 'APT-C-36', 'similarity': 1.0, 'tech_match': '7/114', 'soft_match': '0/48', 'capec_match': '6/54', 'cwe_match': '9/51', 'cve_match': '563/1060', 'pulse_match': '870/1359', 'match_score': 1455, 'profile_size': '9T/2S/9Ca/15Cw/610Cv/910P', 'apt_id': 14, 'is_comprehensive': True, 'total_features': 1836},
    {'rank': 2, 'apt_name': 'APT19', 'similarity': np.float64(0.9702719487784112), 'tech_match': '9/114', 'soft_match': '1/48', 'capec_match': '4/54', 'cwe_match': '5/51', 'cve_match': '288/1060', 'pulse_match': '448/1359', 'match_score': 755, 'profile_size': '21T/3S/9Ca/7Cw/290Cv/450P', 'apt_id': np.int64(20), 'is_comprehensive': True, 'total_features': 2565},
    {'rank': 3, 'apt_name': 'APT5', 'similarity': np.float64(0.9262400938799541), 'tech_match': '14/114', 'soft_match': '3/48', 'capec_match': '8/54', 'cwe_match': '10/51', 'cve_match': '343/1060', 'pulse_match': '579/1359', 'match_score': 957, 'profile_size': '29T/14S/15Ca/12Cw/349Cv/587P', 'apt_id': np.int64(31), 'is_comprehensive': True, 'total_features': 3534},
    {'rank': 4, 'apt_name': 'admin@338', 'similarity': np.float64(0.8788656294320595), 'tech_match': '7/114', 'soft_match': '3/48', 'capec_match': '8/54', 'cwe_match': '10/51', 'cve_match': '287/1060', 'pulse_match': '453/1359', 'match_score': 768, 'profile_size': '13T/8S/11Ca/11Cw/288Cv/454P', 'apt_id': np.int64(4), 'is_comprehensive': True, 'total_features': 3865},
    {'rank': 5, 'apt_name': 'Daggerfly', 'similarity': np.float64(0.8761604117845438), 'tech_match': '10/114', 'soft_match': '0/48', 'capec_match': '14/54', 'cwe_match': '10/51', 'cve_match': '530/1060', 'pulse_match': '716/1359', 'match_score': 1280, 'profile_size': '18T/7S/17Ca/12Cw/532Cv/718P', 'apt_id': np.int64(54), 'is_comprehensive': True, 'total_features': 2498}
    # ... more items
]
list2 = [
    {'rank': 1, 'apt_name': 'Kimsuky', 'similarity': 0.9997708797454834, 'tech_match': '43/114', 'soft_match': '2/48', 'capec_match': '25/54', 'cwe_match': '36/51', 'cve_match': '1013/1060', 'pulse_match': '1329/1359', 'weighted_match': 87270.0, 'profile_size': '92T/13S/49Ca/48Cw/1100Cv/1404P', 'apt_id': 95},
    {'rank': 2, 'apt_name': 'APT41', 'similarity': 0.9996299743652344, 'tech_match': '37/114', 'soft_match': '5/48', 'capec_match': '41/54', 'cwe_match': '45/51', 'cve_match': '1041/1060', 'pulse_match': '1350/1359', 'weighted_match': 90160.0, 'profile_size': '101T/31S/65Ca/56Cw/1131Cv/1414P', 'apt_id': np.int64(30)},
    {'rank': 3, 'apt_name': 'APT37', 'similarity': 0.9977949261665344, 'tech_match': '17/114', 'soft_match': '1/48', 'capec_match': '14/54', 'cwe_match': '21/51', 'cve_match': '779/1060', 'pulse_match': '1079/1359', 'weighted_match': 66850.0, 'profile_size': '29T/13S/21Ca/27Cw/830Cv/1125P', 'apt_id': np.int64(27)},
    {'rank': 4, 'apt_name': 'FIN13', 'similarity': 0.9977742433547974, 'tech_match': '27/114', 'soft_match': '2/48', 'capec_match': '21/54', 'cwe_match': '30/51', 'cve_match': '870/1060', 'pulse_match': '1145/1359', 'weighted_match': 74210.0, 'profile_size': '53T/4S/38Ca/34Cw/972Cv/1176P', 'apt_id': np.int64(72)},
    {'rank': 5, 'apt_name': 'Windshift', 'similarity': 0.9958516359329224, 'tech_match': '12/114', 'soft_match': '0/48', 'capec_match': '13/54', 'cwe_match': '14/51', 'cve_match': '738/1060', 'pulse_match': '1063/1359', 'weighted_match': 63850.0, 'profile_size': '35T/1S/22Ca/20Cw/785Cv/1103P', 'apt_id': np.int64(176)}
    # ... more items
]
list3 = [
    {'apt_id': np.int64(66), 'apt_name': 'Equation', 'similarity': np.float32(0.98807925), 'tech_matches': 0, 'soft_matches': 0, 'capec_matches': 0, 'cwe_matches': 2, 'cve_matches': 161, 'pulse_matches': 333, 'total_matches': 496, 'profile_size': '4T/1S/4Ca/6Cw/263Cv/364P', 'profile_total': 642, 'rank': 1},
    {'apt_id': np.int64(74), 'apt_name': 'FIN5', 'similarity': np.float32(0.9860136), 'tech_matches': 6, 'soft_matches': 2, 'capec_matches': 7, 'cwe_matches': 10, 'cve_matches': 224, 'pulse_matches': 280, 'total_matches': 529, 'profile_size': '12T/7S/11Ca/13Cw/235Cv/284P', 'profile_total': 562, 'rank': 2},
    {'apt_id': np.int64(159), 'apt_name': 'TeamTNT', 'similarity': np.float32(0.9839033), 'tech_matches': 20, 'soft_matches': 0, 'capec_matches': 17, 'cwe_matches': 26, 'cve_matches': 795, 'pulse_matches': 1063, 'total_matches': 1921, 'profile_size': '55T/5S/32Ca/28Cw/797Cv/1065P', 'profile_total': 1982, 'rank': 3},
    {'apt_id': np.int64(55), 'apt_name': 'Dark Caracal', 'similarity': np.float32(0.97965914), 'tech_matches': 8, 'soft_matches': 0, 'capec_matches': 7, 'cwe_matches': 15, 'cve_matches': 288, 'pulse_matches': 455, 'total_matches': 773, 'profile_size': '14T/5S/8Ca/16Cw/289Cv/456P', 'profile_total': 788, 'rank': 4},
    {'apt_id': np.int64(142), 'apt_name': 'SideCopy', 'similarity': np.float32(0.97935516), 'tech_matches': 8, 'soft_matches': 0, 'capec_matches': 6, 'cwe_matches': 4, 'cve_matches': 158, 'pulse_matches': 245, 'total_matches': 421, 'profile_size': '16T/3S/11Ca/7Cw/168Cv/250P', 'profile_total': 455, 'rank': 5}
    # ... more items
]
list4 = [
    {'rank': 1, 'apt_name': 'APT29', 'Score': 0.9966028928756714, 'Encoded_ID': np.int64(22), 'technique': 114, 'software': 48, 'capec': 54, 'cwe': 51, 'cve': 1060, 'pulse': 1359},
    {'rank': 2, 'apt_name': 'Sandworm Team', 'Score': 0.5038824081420898, 'Encoded_ID': np.int64(139), 'technique': 42, 'software': 6, 'capec': 39, 'cwe': 46, 'cve': 1041, 'pulse': 1351},
    {'rank': 3, 'apt_name': 'Thrip', 'Score': 0.502805233001709, 'Encoded_ID': np.int64(165), 'technique': 2, 'software': 2, 'capec': 0, 'cwe': 0, 'cve': 0, 'pulse': 0},
    {'rank': 4, 'apt_name': 'Cleaver', 'Score': 0.502507209777832, 'Encoded_ID': np.int64(45), 'technique': 3, 'software': 2, 'capec': 1, 'cwe': 1, 'cve': 3, 'pulse': 14},
    {'rank': 5, 'apt_name': 'Amateurs -Legitimately targeted individuals', 'Score': 0.5024663209915161, 'Encoded_ID': np.int64(9), 'technique': 6, 'software': 0, 'capec': 0, 'cwe': 0, 'cve': 7, 'pulse': 1}
    # ... more items
]
list5 = [
    {'rank': 1, 'apt_name': 'Kimsuky', 'similarity': 0.9997708797454834, 'tech_match': '43/114', 'soft_match': '2/48', 'capec_match': '25/54', 'cwe_match': '36/51', 'cve_match': '1013/1060', 'pulse_match': '1329/1359', 'weighted_match': 87270.0, 'profile_size': '92T/13S/49Ca/48Cw/1100Cv/1404P', 'apt_id': 95},
    {'rank': 2, 'apt_name': 'APT41', 'Score': np.float64(0.11), 'Encoded_ID': np.int64(30), 'techniques': 37, 'software': 5, 'capec': 41, 'cwe': 45, 'cve': 1041, 'pulse': 1350},
    {'rank': 3, 'apt_name': 'Sandworm Team', 'Score': np.float64(0.08), 'Encoded_ID': np.int64(139), 'techniques': 42, 'software': 6, 'capec': 39, 'cwe': 46, 'cve': 1041, 'pulse': 1351},
    {'rank': 4, 'apt_name': 'Scattered Spider', 'Score': np.float64(0.06), 'Encoded_ID': np.int64(141), 'techniques': 23, 'software': 2, 'capec': 16, 'cwe': 33, 'cve': 811, 'pulse': 1094},
    {'rank': 5, 'apt_name': 'FIN13', 'Score': np.float64(0.04), 'Encoded_ID': np.int64(72), 'techniques': 27, 'software': 2, 'capec': 21, 'cwe': 30, 'cve': 870, 'pulse': 1145}
    # ... more items
]

all_lists = [list1, list2, list3, list4, list5]

# # Method 1: Simple average with tie breaking
# simple_result = hybrid_rank_simple(all_lists)
# print("Method 1 - Simple Average:")
# for name, rank in simple_result:
#     print(f"  {name}: {rank:.2f}")

# # Method 2: Weighted average with confidence
# weighted_result = hybrid_rank_weighted(all_lists, weights=[1, 1, 1, 1, 1])  # Equal weights
# print("\nMethod 2 - Weighted Average with Confidence:")
# for name, rank in weighted_result:
#     print(f"  {name}: {rank:.2f}")

# # Method 3: Borda count
# borda_result = hybrid_rank_borda(all_lists)
# print("\nMethod 3 - Borda Count:")
# for name, rank in borda_result:
#     print(f"  {name}: {rank}")

# Method 4: rank aggregation with tie handling
aggregate_result = hybrid_rank_aggregate(all_lists)
print("\nMethod 4 - rank Aggregation:")
for name, rank in aggregate_result:
    print(f"  {name}: {rank}")

# # Method 5: Reciprocal rank Fusion
# rrf_result = hybrid_rank_rrf(all_lists, k=60)
# print("\nMethod 5 - Reciprocal rank Fusion (RRF):")
# for name, rank in rrf_result:
#     print(f"  {name}: {rank}")
