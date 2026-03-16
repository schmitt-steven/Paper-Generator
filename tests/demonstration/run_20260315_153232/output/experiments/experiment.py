import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set headless environment for potential pygame usage (though not used here)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Use Agg backend to prevent UI windows
plt.switch_backend('Agg')

# Constants
DATA_FILE_PATH = 'datasets/1000A-0001_de_flat.csv'
RESULTS_FILE = 'results.json'
PLOTS_DIR = 'plots'
MAD_THRESHOLD_ACCEPTABLE = 0.05
MAD_THRESHOLD_SUSPECT = 0.012

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_PROBS = {d: math.log10(1 + 1/d) for d in range(1, 10)}
BENFORD_VALUES = [BENFORD_PROBS[d] for d in range(1, 10)]

# Helper Functions

def extract_first_digit(value):
    """Extract the first significant digit from a positive number."""
    if pd.isna(value) or value <= 0:
        return None
    # Convert to string to handle potential formatting issues easily
    s_val = str(float(value))
    for char in s_val:
        if char.isdigit():
            return int(char)
    return None

def calculate_mad(observed_probs, expected_probs):
    """Calculate Mean Absolute Deviation between observed and expected distributions."""
    digits = range(1, 10)
    deviations = [abs(observed_probs[d] - expected_probs[d]) for d in digits]
    return sum(deviations) / len(digits)

def calculate_chi_square(observed_counts, total_count, expected_probs):
    """Calculate Chi-Square statistic."""
    chi2 = 0.0
    for d in range(1, 10):
        expected_count = total_count * expected_probs[d]
        if expected_count > 0:
            observed_count = observed_counts.get(d, 0)
            chi2 += ((observed_count - expected_count) ** 2) / expected_count
    return chi2

def classify_conformity(mad_score):
    """Classify conformity based on MAD thresholds."""
    if mad_score < MAD_THRESHOLD_SUSPECT:
        return "Suspect (Too Perfect)"
    elif mad_score <= MAD_THRESHOLD_ACCEPTABLE:
        return "Natural / Acceptable"
    else:
        return "Anomalous / Deviant"

def generate_synthetic_uniform(n_samples):
    """Generate synthetic data with uniform digit distribution."""
    digits = np.random.choice(range(1, 10), size=n_samples)
    # Convert to actual numbers for simulation (e.g., random magnitude)
    return [int(d * 10**np.random.randint(2, 6)) for d in digits]

def generate_synthetic_biased(n_samples):
    """Generate synthetic data with bias towards digit '5'."""
    # 40% chance of being a number starting with 5, rest uniform
    digits = []
    for _ in range(n_samples):
        if np.random.rand() < 0.4:
            d = 5
        else:
            d = np.random.choice(range(1, 10))
        digits.append(int(d * 10**np.random.randint(2, 6)))
    return digits

def analyze_dataset(data_series, dataset_name):
    """Perform full Benford analysis on a dataset."""
    # Extract first digits
    first_digits = [extract_first_digit(v) for v in data_series]
    valid_digits = [d for d in first_digits if d is not None]
    
    if len(valid_digits) == 0:
        return {
            "name": dataset_name,
            "n_samples": 0,
            "mad": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
            "classification": "No Data",
            "observed_probs": {},
            "deviations": {}
        }

    # Count frequencies
    counts = {d: valid_digits.count(d) for d in range(1, 10)}
    total = len(valid_digits)
    
    # Calculate observed proportions
    observed_probs = {d: counts[d] / total for d in range(1, 10)}
    
    # Calculate metrics
    mad_score = calculate_mad(observed_probs, BENFORD_PROBS)
    chi2_stat = calculate_chi_square(counts, total, BENFORD_PROBS)
    
    # Approximate p-value (Chi-square with 8 degrees of freedom)
    # Using a simple approximation or lookup for large samples. 
    # For N > 1000, even small deviations yield tiny p-values.
    # We will use scipy if available, otherwise approximate.
    try:
        from scipy.stats import chi2
        p_val = 1 - chi2.cdf(chi2_stat, df=8)
    except ImportError:
        # Fallback for environments without scipy: just report Chi2 value
        p_val = "N/A (SciPy not available)"

    classification = classify_conformity(mad_score)
    
    # Calculate deviations for heatmap
    deviations = {d: abs(observed_probs[d] - BENFORD_PROBS[d]) for d in range(1, 10)}

    return {
        "name": dataset_name,
        "n_samples": total,
        "mad": mad_score,
        "chi2": chi2_stat,
        "p_value": p_val if isinstance(p_val, float) else str(p_val),
        "classification": classification,
        "observed_probs": observed_probs,
        "deviations": deviations
    }

# Main Execution Logic

def main():
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading dataset...")
    try:
        df = pd.read_csv(DATA_FILE_PATH, sep=';')
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {DATA_FILE_PATH}")
        return

    # Data Exploration: Check unique values for filtering columns
    print("\n--- Data Exploration ---")
    
    # Check value_variable_label column
    if 'value_variable_label' in df.columns:
        labels = df['value_variable_label'].unique()
        print(f"Available variable labels: {labels}")
        
        # Map German to English for analysis and output
        label_mapping = {}
        for lbl in labels:
            if pd.isna(lbl): continue
            lower_lbl = str(lbl).lower()
            if 'personen' in lower_lbl or 'bevölkerung' in lower_lbl:
                label_mapping[lbl] = "Population Count"
            elif 'fläche' in lower_lbl:
                label_mapping[lbl] = "Area (km²)"
            elif 'dichte' in lower_lbl:
                label_mapping[lbl] = "Density (Ew/km²)"
        
        print(f"Mapped labels: {label_mapping}")
    else:
        print("ERROR: Column 'value_variable_label' not found.")
        return

    # Check value_q column for quality flags
    if 'value_q' in df.columns:
        q_flags = df['value_q'].unique()
        print(f"Available quality flags (value_q): {q_flags}")
        
        # Filter for exact values ('e')
        target_flag = 'e'
        if target_flag not in [str(x) for x in q_flags]:
            print(f"WARNING: Target flag '{target_flag}' not found. Available: {q_flags}. Skipping quality filter.")
    else:
        print("ERROR: Column 'value_q' not found.")
        return

    # Filter Data
    print("\n--- Filtering Data ---")
    
    # Apply quality filter first
    df_filtered = df[df['value_q'].astype(str) == target_flag].copy()
    if len(df_filtered) < len(df):
        print(f"Filtered {len(df)} rows to {len(df_filtered)} rows based on quality flag '{target_flag}'.")

    # Extract variables using case-insensitive matching for robustness
    def get_subset(col_name, keyword):
        mask = df_filtered[col_name].str.contains(keyword, case=False, na=False)
        subset = df_filtered[mask]
        if len(subset) == 0:
            print(f"WARNING: No rows found containing '{keyword}' in column '{col_name}'.")
        return subset

    pop_subset = get_subset('value_variable_label', 'personen') # "Personen"
    area_subset = get_subset('value_variable_label', 'fläche')  # "Fläche"
    dens_subset = get_subset('value_variable_label', 'dichte')  # "Bevölkerungsdichte"

    # Extract values and clean them (handle commas in decimals if any, though CSV usually uses dots)
    def extract_values(subset):
        vals = subset['value'].astype(str).str.replace(',', '.', regex=False) # Handle potential European formatting
        try:
            return pd.to_numeric(vals, errors='coerce')
        except Exception as e:
            print(f"Error converting values: {e}")
            return pd.Series([np.nan] * len(subset))

    pop_values = extract_values(pop_subset)
    area_values = extract_values(area_subset)
    dens_values = extract_values(dens_subset)

    # Generate Synthetic Controls (Size matching the smallest real dataset to ensure fair comparison, or fixed size)
    n_synthetic = min(len(pop_values), len(area_values), len(dens_values)) if 0 < min(len(pop_values), len(area_values), len(dens_values)) else 3600
    # Ensure we have enough samples for meaningful stats
    n_synthetic = max(n_synthetic, 3600) 
    
    print(f"Generating synthetic controls with N={n_synthetic}...")
    
    synth_uniform_vals = generate_synthetic_uniform(n_synthetic)
    synth_biased_vals = generate_synthetic_biased(n_synthetic)

    # Run Analysis
    print("\n--- Running Benford Analysis ---")
    
    results = []
    
    # Analyze Real Data
    if len(pop_values.dropna()) > 0:
        res_pop = analyze_dataset(pop_values, "Population Count")
        results.append(res_pop)
        
    if len(area_values.dropna()) > 0:
        res_area = analyze_dataset(area_values, "Area (km²)")
        results.append(res_area)
        
    if len(dens_values.dropna()) > 0:
        res_dens = analyze_dataset(dens_values, "Density (Ew/km²)")
        results.append(res_dens)

    # Analyze Synthetic Data
    res_uniform = analyze_dataset(synth_uniform_vals, "Synthetic Uniform")
    results.append(res_uniform)
    
    res_biased = analyze_dataset(synth_biased_vals, "Synthetic Biased (Anchoring 5)")
    results.append(res_biased)

    # Print Summary Table
    print("\n--- Results Summary ---")
    print(f"{'Dataset':<25} | {'N':>6} | {'MAD':>8} | {'Chi-Sq':>10} | {'Classification'}")
    print("-" * 70)
    
    for r in results:
        name = r['name'][:24] if len(r['name']) > 24 else r['name']
        n = int(r['n_samples'])
        mad = f"{r['mad']:.5f}" if not np.isnan(r['mad']) else "N/A"
        chi2 = f"{r['chi2']:.1f}" if isinstance(r['chi2'], float) else str(r['chi2'])
        classif = r['classification']
        print(f"{name:<25} | {n:>6} | {mad:>8} | {chi2:>10} | {classif}")

    # Save Results to JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Visualization
    print("\n--- Generating Plots ---")
    
    # Prepare data for plotting
    datasets = [r['name'] for r in results]
    mad_scores = [r['mad'] if not np.isnan(r['mad']) else 0 for r in results]
    observed_probs_list = [r['observed_probs'] for r in results]
    
    # Plot 1: First-Digit Distribution Bar Chart
    plt.figure(figsize=(12, 7))
    x = np.arange(1, 10)
    width = 0.1
    
    # Colors from seaborn colorblind palette
    colors = sns.color_palette("colorblind", len(results))
    
    for i, probs in enumerate(observed_probs_list):
        values = [probs.get(d, 0) for d in range(1, 10)]
        plt.bar(x + i*width, values, width, label=datasets[i], color=colors[i])

    # Overlay Benford's Law
    plt.plot(x, BENFORD_VALUES, 'k-', linewidth=2.5, marker='o', label="Benford's Expected")
    
    plt.xlabel('First Digit')
    plt.ylabel('Proportion')
    plt.title('Observed vs. Expected First-Digit Distribution (German Zensus 2022)')
    plt.xticks(x + width * len(results)/2, range(1, 10))
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save Plot 1
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/benford_digit_distribution.pdf')
    print("[Plot Summary: benford_digit_distribution.pdf] Bar chart comparing observed first-digit frequencies for Population, Area, Density, and two synthetic datasets against Benford's Law curve.")
    plt.close()

    # Plot 2: Conformity Comparison Chart (MAD Scores)
    plt.figure(figsize=(10, 6))
    
    y_pos = np.arange(len(datasets))
    bars = plt.barh(y_pos, mad_scores, color=colors)
    
    # Add threshold lines
    plt.axvline(MAD_THRESHOLD_ACCEPTABLE, color='red', linestyle='--', label=f'Acceptable Threshold ({MAD_THRESHOLD_ACCEPTABLE})')
    plt.axvline(MAD_THRESHOLD_SUSPECT, color='green', linestyle=':', label=f'Suspect Threshold ({MAD_THRESHOLD_SUSPECT})')
    
    # Annotate bars with values
    for i, v in enumerate(mad_scores):
        if not np.isnan(v):
            plt.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=9)
            
    plt.xlabel('Mean Absolute Deviation (MAD)')
    plt.ylabel('Dataset')
    plt.title('Conformity Comparison: MAD Scores Across Datasets')
    plt.yticks(y_pos, datasets)
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save Plot 2
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/conformity_comparison.pdf')
    print("[Plot Summary: conformity_comparison.pdf] Horizontal bar chart comparing MAD scores for all datasets, with red dashed line indicating acceptable threshold and green dotted line for suspect threshold.")
    plt.close()

    # Plot 3: Per-Digit Deviation Heatmap
    plt.figure(figsize=(10, 8))
    
    deviation_matrix = []
    for r in results:
        row = [r['deviations'].get(d, 0) for d in range(1, 10)]
        deviation_matrix.append(row)
        
    deviation_matrix = np.array(deviation_matrix)
    
    sns.heatmap(deviation_matrix, annot=True, fmt=".3f", cmap="RdYlBu_r", 
                xticklabels=range(1, 10), yticklabels=datasets, ax=plt.gca())
    
    plt.xlabel('First Digit')
    plt.ylabel('Dataset')
    plt.title('Per-Digit Deviation Heatmap (|Observed - Expected|)')
    
    # Save Plot 3
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/deviation_heatmap.pdf')
    print("[Plot Summary: deviation_heatmap.pdf] Heatmap visualizing the magnitude of deviation for each digit (1-9) across all datasets, highlighting specific anomalies.")
    plt.close()

    print("\n--- Experiment Complete ---")

if __name__ == "__main__":
    main()