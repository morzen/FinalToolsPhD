import json

# Load the JSON data
with open('3_feature_CTI_Eval.json', 'r') as file:
    data = json.load(file)

# Function to calculate averages for a set of reports
def calculate_averages(reports):
    rank_sums = {}
    rank_counts = {}
    
    for report in reports:
        # Skip reports with "combination_of_evidence"
        if report['report_link'] == "combination_of_evidence":
            continue
            
        if 'results_ranks' in report:
            for rank_name, rank_value in report['results_ranks'].items():
                if rank_name not in rank_sums:
                    rank_sums[rank_name] = 0
                    rank_counts[rank_name] = 0
                rank_sums[rank_name] += rank_value
                rank_counts[rank_name] += 1
    
    # Calculate averages
    averages = {}
    for rank_name in rank_sums:
        averages[rank_name] = rank_sums[rank_name] / rank_counts[rank_name]
    
    # Sort by value (smallest to largest)
    return dict(sorted(averages.items(), key=lambda item: item[1]))

# Calculate and display averages for each APT group
print("=" * 60)
print("AVERAGE RANK RESULTS BY APT GROUP")
print("=" * 60)

for apt_group, reports in data.items():
    apt_averages = calculate_averages(reports)
    if apt_averages:  # Only print if there are valid reports
        print(f"\n{apt_group}:")
        for rank_name, avg_value in apt_averages.items():
            print(f"  {rank_name}: {avg_value:.2f}")

# Calculate total averages (all reports combined)
print("\n" + "=" * 60)
print("TOTAL AVERAGE RANK RESULTS (ALL APT GROUPS)")
print("=" * 60)

# Collect all reports for total calculation
all_reports = []
for reports in data.values():
    all_reports.extend(reports)

total_averages = calculate_averages(all_reports)
for rank_name, avg_value in total_averages.items():
    print(f"{rank_name}: {avg_value:.2f}")