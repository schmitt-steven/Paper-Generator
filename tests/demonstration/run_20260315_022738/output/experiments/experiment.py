import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set headless environment
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Use Agg backend to prevent UI windows
plt.switch_backend('Agg')

# --- Configuration & Constants ---
DATASET_PATH = 'datasets/1000A-0001_de_flat.csv'
RESULTS_FILE = 'results.json'
PLOTS_DIR = 'plots'

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_PROBS = {d: np.log10(1 + 1/d) for d in range(1, 10)}
BENFORD_VALUES = [BENFORD_PROBS[d] for d in range(1, 10)]

# Ensure Benford values sum to exactly 1.0 to prevent scipy chisquare errors due to float precision
total_benford = sum(BENFORD_VALUES)
BENFORD_VALUES = [p / total_benford for p in BENFORD_VALUES]

# Conformity Thresholds (Schechter, 2015)
# Logic: Low MAD = Good Conformity. High MAD = Non-Conforming/Suspicious of fabrication.
THRESHOLD_SUSPECT = 0.009   # Below this is considered "Too Perfect" or highly suspicious in some forensic contexts, 
                            # but for general Benford validation, < 0.012 is usually "Good".
                            # We will use standard interpretation: Low MAD = Compliant.
THRESHOLD_ACCEPTABLE = 0.012
THRESHOLD_NON_CONFORMING = 0.025

# Colorblind-friendly palette
COLOR_PALETTE = sns.color_palette("colorblind")

# --- Helper Functions ---

def extract_leading_digit(value):
    """Extract the first non-zero digit from a number."""
    if pd.isna(value) or value <= 0:
        return None
    try:
        # Handle potential string inputs with commas (German format)
        if isinstance(value, str):
            value = float(value.replace(',', '.'))

        if value == 0:
            return None

        # Normalize to scientific notation to find leading digit
        exp = int(np.floor(np.log10(abs(value))))
        mantissa = abs(value) / (10 ** exp)

        # The leading digit is the integer part of the mantissa
        return int(mantissa)
    except (ValueError, ZeroDivisionError):
        return None

def calculate_benford_metrics(observed_counts, total_count):
    """Calculate MAD and Chi-Square statistics."""
    if total_count == 0:
        return {'mad': np.nan, 'chi2': np.nan, 'p_value': np.nan}

    # Observed proportions
    observed_props = [observed_counts.get(d, 0) / total_count for d in range(1, 10)]

    # Expected proportions (Benford) - already normalized to sum=1.0
    expected_props = BENFORD_VALUES

    # Mean Absolute Deviation (MAD)
    mad = np.mean([abs(o - e) for o, e in zip(observed_props, expected_props)])

    # Chi-Square Goodness-of-Fit
    observed_counts_list = [observed_counts.get(d, 0) for d in range(1, 10)]

    # Ensure sum of observed matches total_count exactly (integer arithmetic)
    if sum(observed_counts_list) != total_count:
        # Fallback to calculated total if mismatch due to logic error
        total_count = sum(observed_counts_list)

    expected_counts = [total_count * p for p in expected_props]

    try:
        chi2_stat, p_value = stats.chisquare(observed_counts_list, f_exp=expected_counts)
    except ValueError as e:
        # Handle cases where scipy fails due to precision or zero counts (though unlikely with Benford)
        return {'mad': mad, 'chi2': np.nan, 'p_value': np.nan}

    return {
        'mad': mad,
        'chi2': chi2_stat,
        'p_value': p_value,
        'observed_props': observed_props,
        'observed_counts': observed_counts_list
    }

def classify_conformity(mad):
    """Classify data based on MAD score.
    
    Interpretation:
    - Low MAD (< 0.012): Good conformity to Benford's Law (Natural Data).
    - Medium MAD (0.012 - 0.025): Suspect or borderline.
    - High MAD (> 0.025): Non-conforming (Likely Fabricated or Derived Ratios).
    
    Note: The original plan had inverted logic for "Suspicious" (<0.009), 
    but standard Benford analysis treats low deviation as compliance. 
    We align with the Success Criteria: "Population counts... exhibit low MAD values".
    """
    if pd.isna(mad):
        return "Unknown"
    
    # Re-evaluating thresholds based on Success Criteria (Low MAD = Good)
    if mad < THRESHOLD_ACCEPTABLE:
        return "Benford Compliant (Natural)"
    elif mad < THRESHOLD_NON_CONFORMING:
        return "Suspect / Borderline"
    else:
        return "Non-Conforming (Anomalous)"

def generate_synthetic_uniform(n_samples):
    """Generate synthetic data with uniform digit distribution."""
    if n_samples <= 0:
        return []
    # Ensure distinct random state for reproducibility within run, though not strictly required here
    digits = np.random.choice(range(1, 10), size=n_samples)
    # Convert to numbers that start with these digits (e.g., d * 10^k)
    return [int(d * np.power(10, np.random.randint(2, 6))) for d in digits]

def generate_synthetic_biased(n_samples):
    """Generate synthetic data simulating human bias (anchoring on 5)."""
    if n_samples <= 0:
        return []
    
    biased_count = int(n_samples * 0.4)
    uniform_count = n_samples - biased_count
    
    # Explicitly create the array to ensure bias is present
    biased_part = np.full(biased_count, 5)
    random_part = np.random.choice(range(1, 10), size=uniform_count)
    
    digits = np.concatenate([biased_part, random_part])
    
    # Convert to numbers that start with these digits
    return [int(d * np.power(10, np.random.randint(2, 6))) for d in digits]

# --- Data Loading & Exploration ---

def load_and_explore_data():
    """Load dataset and print unique values for filtering."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, sep=';')

    # Print unique labels to ensure we use correct English translations
    print("=== Data Exploration ===")
    print(f"Total rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check variable labels (German) - translate for internal logic but keep raw for debug
    if 'value_variable_label' in df.columns:
        unique_labels = df['value_variable_label'].unique()
        print(f"\nUnique value_variable_labels found:")
        for label in unique_labels:
            print(f"  - '{label}'")

    # Check quality flags
    if 'value_q' in df.columns:
        unique_flags = df['value_q'].unique()
        print(f"\nUnique value_q flags found:")
        for flag in unique_flags:
            print(f"  - '{flag}'")

    return df

def filter_and_extract_data(df):
    """Filter data by quality and variable type, extract leading digits."""

    # Define mappings based on exploration output (German keywords)
    target_labels = {
        'population': ['personen'],  
        'area': ['fläche'],          
        'density': ['bevölkerungsdichte'] 
    }

    results = {}

    for var_type, keywords in target_labels.items():
        # Filter rows where label contains any of the keywords (case-insensitive)
        mask = df['value_variable_label'].str.lower().apply(lambda x: any(kw in str(x).lower() for kw in keywords))

        # Also filter by quality flag 'e' (exact estimate)
        if 'value_q' in df.columns:
            q_mask = df['value_q'] == 'e'
            mask = mask & q_mask

        subset_df = df[mask]

        if len(subset_df) == 0:
            print(f"WARNING: No data found for {var_type} with keywords {keywords}. Skipping.")
            continue

        # Extract values and convert to numeric
        def safe_convert(val):
            try:
                if isinstance(val, str):
                    return float(val.replace(',', '.'))
                return float(val)
            except (ValueError, TypeError):
                return np.nan

        values = subset_df['value'].apply(safe_convert).dropna()

        # Extract leading digits
        leading_digits = []
        for v in values:
            d = extract_leading_digit(v)
            if d is not None and 1 <= d <= 9:
                leading_digits.append(d)

        results[var_type] = {
            'label': var_type.capitalize().replace('_', ' '), 
            'count': len(leading_digits),
            'digits': leading_digits,
            'raw_subset_size': len(subset_df)
        }

        print(f"[Data Filtered] {var_type}: Found {len(leading_digits)} valid entries (from {len(subset_df)} raw rows).")

    return results

# --- Experiment Execution ---

def run_experiment():
    """Main execution pipeline."""

    # 1. Load Data
    df = load_and_explore_data()

    # 2. Filter and Extract
    real_data = filter_and_extract_data(df)

    if not real_data:
        print("ERROR: No valid data extracted. Aborting.")
        return []

    # 3. Generate Synthetic Controls
    n_samples = max([d['count'] for d in real_data.values()]) + 1000 

    synthetic_uniform_digits = generate_synthetic_uniform(n_samples)
    synthetic_biased_digits = generate_synthetic_biased(n_samples)

    real_data['synthetic_uniform'] = {
        'label': 'Synthetic Uniform',
        'count': len(synthetic_uniform_digits),
        'digits': synthetic_uniform_digits,
        'raw_subset_size': n_samples
    }

    real_data['synthetic_biased'] = {
        'label': 'Synthetic Biased (Anchored)',
        'count': len(synthetic_biased_digits),
        'digits': synthetic_biased_digits,
        'raw_subset_size': n_samples
    }

    # 4. Calculate Metrics for all datasets
    analysis_results = []

    for key, data in real_data.items():
        digits = data['digits']

        # Ensure we have valid digits before counting
        if not digits:
            print(f"WARNING: No valid digits found for {data['label']} after filtering. Skipping metrics.")
            continue

        counts = {d: digits.count(d) for d in range(1, 10)}

        metrics = calculate_benford_metrics(counts, len(digits))

        # Skip if calculation failed (NaN returned)
        if np.isnan(metrics['mad']):
             print(f"WARNING: Metrics calculation failed for {data['label']}. Skipping.")
             continue

        classification = classify_conformity(metrics['mad'])

        analysis_results.append({
            'dataset': data['label'],
            'sample_size': len(digits),
            'mad': round(float(metrics['mad']), 6),
            'chi2': round(float(metrics['chi2']), 2) if not np.isnan(metrics['chi2']) else None,
            'p_value': round(float(metrics['p_value']), 10) if not np.isnan(metrics['p_value']) else None,
            'classification': classification,
            'observed_props': metrics['observed_props']
        })

        print(f"[Metric] {data['label']}: MAD={metrics['mad']:.6f}, Chi2={metrics['chi2']:.2f}, P={metrics['p_value']:.10e}")

    # 5. Save Results to JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    return analysis_results

# --- Visualization Functions ---

def create_plots(results):
    """Generate publication-quality plots."""

    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not results:
        print("No valid results to plot.")
        return

    # Prepare data for plotting
    datasets = [r['dataset'] for r in results]
    mad_scores = [r['mad'] for r in results]
    classifications = [r['classification'] for r in results]
    observed_props_list = [r['observed_props'] for r in results]

    # Plot 1: First-Digit Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(1, 10)
    width = 0.08 # Reduced width to fit more bars comfortably

    # Colors for datasets (cycle through palette)
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(datasets))]

    for i, props in enumerate(observed_props_list):
        # Ensure no NaN/Inf in data before plotting
        if np.any(np.isnan(props)) or np.any(np.isinf(props)):
            continue

        plt.bar(x + i * width - (len(datasets)-1)*width/2, props, width, label=datasets[i], color=colors[i])

    # Benford Expected Line
    plt.plot(x, BENFORD_VALUES, 'k-', linewidth=2.5, marker='o', markersize=6, label="Benford's Law (Expected)")

    plt.xlabel('Leading Digit')
    plt.ylabel('Proportion')
    plt.title('First-Digit Distribution: Observed vs Benford\'s Expected')
    plt.xticks(x)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'benford_digit_distribution.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Plot Summary: benford_digit_distribution.pdf] Bar chart comparing observed leading digit frequencies for {len(datasets)} datasets against Benford's Law baseline.")

    # Plot 2: Conformity Comparison Chart (MAD Scores)
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(datasets))

    # Color based on classification
    plot_colors = []
    for cls in classifications:
        if "Non-Conforming" in cls or "Anomalous" in cls:
            plot_colors.append('#d9534f') # Red
        elif "Suspect" in cls:
            plot_colors.append('#f0ad4e') # Orange
        elif "Compliant" in cls:
            plot_colors.append('#5cb85c') # Green
        else:
            plot_colors.append('#5bc0de') # Blue

    bars = plt.barh(y_pos, mad_scores, color=plot_colors)

    # Threshold lines
    plt.axvline(THRESHOLD_ACCEPTABLE, color='gray', linestyle='--', label=f'Acceptable Threshold ({THRESHOLD_ACCEPTABLE})')
    plt.axvline(THRESHOLD_NON_CONFORMING, color='red', linestyle=':', label=f'Non-Conforming Threshold ({THRESHOLD_NON_CONFORMING})')

    # Add value labels on bars (only if valid)
    for i, v in enumerate(mad_scores):
        if not np.isnan(v):
            plt.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=9)

    plt.xlabel('Mean Absolute Deviation (MAD)')
    plt.ylabel('Dataset')
    plt.title('Conformity Comparison: MAD Scores Across Datasets')
    plt.yticks(y_pos, datasets)
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Invert y-axis to match list order (top-down) if desired, or keep default
    plt.gca().invert_yaxis() 

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'conformity_comparison.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Plot Summary: conformity_comparison.pdf] Horizontal bar chart showing MAD scores for {len(datasets)} datasets. Green indicates acceptable conformity, Red/Orange indicates suspicion.")

    # Plot 3: Per-Digit Deviation Heatmap
    deviation_matrix = []
    valid_datasets_heatmap = []

    for i, props in enumerate(observed_props_list):
        if np.any(np.isnan(props)) or np.any(np.isinf(props)):
            continue
        deviations = [abs(p - e) for p, e in zip(props, BENFORD_VALUES)]
        deviation_matrix.append(deviations)
        valid_datasets_heatmap.append(datasets[i])

    if deviation_matrix:
        plt.figure(figsize=(10, 6))
        sns.heatmap(deviation_matrix, 
                    annot=True, fmt='.3f', 
                    cmap='RdYlBu_r', # Red-Yellow-Blue reversed (Red=High Deviation)
                    xticklabels=['Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9'],
                    yticklabels=valid_datasets_heatmap,
                    cbar_kws={'label': 'Absolute Deviation from Benford'})

        plt.title('Per-Digit Deviation Heatmap')
        plt.xlabel('Leading Digit')
        plt.ylabel('Dataset')
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, 'deviation_heatmap.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Plot Summary: deviation_heatmap.pdf] Heatmap visualizing absolute deviations per digit for {len(valid_datasets_heatmap)} datasets. Red indicates high deviation from Benford's Law.")
    else:
        print("WARNING: No valid data available to generate heatmap.")

# --- Main Execution Block ---

if __name__ == "__main__":
    try:
        # Run the experiment logic
        results = run_experiment()

        if results:
            # Generate visualizations
            create_plots(results)

            # Print Final Summary Table to Console
            print("\n" + "="*80)
            print("FINAL SUMMARY TABLE")
            print("="*80)
            print(f"{'Dataset':<25} | {'N':>6} | {'MAD':>10} | {'Chi-Sq':>10} | {'P-Value':>12} | Classification")
            print("-"*80)

            for r in results:
                p_val_str = f"{r['p_value']:.2e}" if r['p_value'] is not None and r['p_value'] < 0.001 else (f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A")
                chi_str = f"{r['chi2']:.2f}" if r['chi2'] is not None else "N/A"

                print(f"{r['dataset']:<25} | {r['sample_size']:>6} | {r['mad']:>10.6f} | {chi_str:>10} | {p_val_str:>12} | {r['classification']}")

            print("="*80)
            print("Analysis Complete.")
        else:
            print("No results to display.")

    except Exception as e:
        print(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()