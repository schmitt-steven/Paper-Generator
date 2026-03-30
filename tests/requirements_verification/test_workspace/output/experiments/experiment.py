import os
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, chisquare

# Set environment for headless execution
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Use Agg backend to prevent GUI windows
plt.switch_backend('Agg')

# Constants
DATASET_PATH = Path("datasets/1000A-0001_de_flat.csv")
RESULTS_FILE = "results.json"
PLOTS_DIR = Path("plots")
MAD_THRESHOLD_HIGH = 0.012
MAD_THRESHOLD_MODERATE = 0.025

# Variable mappings (Internal Code -> English Label)
VARIABLE_MAP = {
    'PRS018': 'Population Count',
    'FLC001': 'Area (km²)',
    'PRS017': 'Population Density'
}

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_PROBS = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

def load_and_filter_data(path: Path):
    """Load CSV and filter for specific variables with quality flag 'e'."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Load data
    df = pd.read_csv(path, delimiter=';', encoding='utf-8-sig')

    # Data Exploration: Print unique values before filtering
    print("Available value_variable_codes:", df['value_variable_code'].unique())
    print("Available value_q flags:", df['value_q'].unique())

    # Filter for target variables and quality flag 'e'
    valid_codes = ['PRS018', 'FLC001', 'PRS017']
    
    # Use case-insensitive matching to be safe, though data seems consistent
    mask_var = df['value_variable_code'].isin(valid_codes)
    mask_q = (df['value_q'] == 'e') | (df['value_q'].str.lower() == 'e')

    filtered_df = df[mask_var & mask_q].copy()

    if filtered_df.empty:
        print("WARNING: Filtered dataset is empty. Check variable codes and quality flags.")
        return pd.DataFrame()

    # Convert value column to numeric, handling comma decimals
    def safe_convert(val):
        try:
            return float(str(val).replace(',', '.'))
        except (ValueError, TypeError):
            return np.nan

    filtered_df['value_float'] = filtered_df['value'].apply(safe_convert)
    
    # Remove non-positive values for Benford analysis (Benford applies to positive numbers)
    valid_data = filtered_df[filtered_df['value_float'] > 0].copy()

    if valid_data.empty:
        print("WARNING: No positive numeric values found after filtering.")
        return pd.DataFrame()

    # Map variable codes to labels for clarity
    valid_data['variable_label'] = valid_data['value_variable_code'].map(VARIABLE_MAP)

    return valid_data

def extract_first_digit(value):
    """Extract the first significant digit from a positive number."""
    if value <= 0:
        return None
    # Use log10 to find magnitude, then divide to get leading digit
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = abs(value) / (10 ** exponent)
    return int(mantissa)

def compute_benford_metrics(values):
    """Compute observed frequencies, MAD, and Chi-square for a list of values."""
    digits = [extract_first_digit(v) for v in values]
    valid_digits = [d for d in digits if d is not None]

    if len(valid_digits) == 0:
        return {
            'count': 0,
            'observed_freqs': np.zeros(9),
            'mad': np.nan,
            'chi2_stat': np.nan,
            'p_value': np.nan,
            'classification': 'N/A'
        }

    # Compute observed frequencies (proportions)
    counts = np.bincount(valid_digits, minlength=10)[1:]  # Ignore index 0
    total = len(valid_digits)
    observed_freqs = counts / total

    # Calculate MAD
    deviation = np.abs(observed_freqs - BENFORD_PROBS)
    mad = np.mean(deviation)

    # Calculate Chi-square statistic and p-value
    expected_counts = total * BENFORD_PROBS
    chi2_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts)

    # Classification based on MAD thresholds (Schechter's criteria adapted)
    if mad < MAD_THRESHOLD_HIGH:
        classification = "Highly Conformant"
    elif mad < MAD_THRESHOLD_MODERATE:
        classification = "Conformant"
    else:
        classification = "Non-Conformant / Suspicious"

    return {
        'count': total,
        'observed_freqs': observed_freqs,
        'mad': mad,
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'classification': classification
    }

def generate_synthetic_data(n_samples: int, seed: int = 42):
    """Generate synthetic datasets for control analysis."""
    np.random.seed(seed)

    # Uniform Control: Each digit 1-9 has equal probability (1/9)
    uniform_digits = np.random.choice(np.arange(1, 10), size=n_samples, p=[1/9]*9)
    
    # Fabricated Control: Human bias towards '5' and '6', underrepresentation of '4'
    # Simulating psychological anchoring to round numbers or specific digits
    biased_probs = np.array([0.08, 0.08, 0.08, 0.02, 0.30, 0.30, 0.05, 0.05, 0.04]) # Sum ~1.0
    biased_digits = np.random.choice(np.arange(1, 10), size=n_samples, p=biased_probs)

    return uniform_digits, biased_digits

def create_plots(results_data):
    """Generate publication-quality plots."""
    PLOTS_DIR.mkdir(exist_ok=True)

    # Set style for professional look
    sns.set_theme(style="whitegrid", palette="colorblind")

    datasets = list(results_data.keys())
    labels = [VARIABLE_MAP.get(d, d) if d in VARIABLE_MAP else d for d in datasets]
    
    # Prepare data for plotting
    observed_freqs_list = [results_data[d]['observed_freqs'] for d in datasets]
    mad_scores = [results_data[d]['mad'] for d in datasets]

    # --- Plot 1: First-Digit Distribution Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(1, 10)
    width = 0.1
    
    # Plot observed frequencies for each dataset
    for i, freqs in enumerate(observed_freqs_list):
        ax.bar(x + i * width - (len(datasets)-1)*width/2, freqs, width, label=labels[i])

    # Overlay Benford's expected distribution
    ax.plot(x, BENFORD_PROBS, 'k-', linewidth=2.5, marker='o', label="Benford's Expected")

    ax.set_xlabel('First Digit')
    ax.set_ylabel('Proportion')
    ax.set_title('Observed vs. Benford\'s Law First-Digit Distribution')
    ax.set_xticks(x)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "benford_digit_distribution.pdf", dpi=150)
    print(f"[Plot Summary: benford_digit_distribution.pdf] Bar chart comparing observed first-digit frequencies for {len(datasets)} datasets against Benford's Law (black line).")

    # --- Plot 2: Conformity Comparison Chart (MAD Scores) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("colorblind", len(datasets))
    bars = ax.bar(labels, mad_scores, color=colors, edgecolor='black')

    # Add threshold lines
    ax.axhline(y=MAD_THRESHOLD_HIGH, color='green', linestyle='--', label=f'Highly Conformant (< {MAD_THRESHOLD_HIGH:.3f})')
    ax.axhline(y=MAD_THRESHOLD_MODERATE, color='orange', linestyle='--', label=f'Conformant (< {MAD_THRESHOLD_MODERATE:.3f})')

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom' if height > 0.05 else 'top', fontsize=9)

    ax.set_ylabel('Mean Absolute Deviation (MAD)')
    ax.set_title('Conformity Comparison: MAD Scores Across Datasets')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "conformity_comparison_mad.pdf", dpi=150)
    print(f"[Plot Summary: conformity_comparison_mad.pdf] Bar chart showing MAD scores for {len(datasets)} datasets with threshold lines.")

    # --- Plot 3: Per-Digit Deviation Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    deviation_matrix = np.array([results_data[d]['observed_freqs'] - BENFORD_PROBS for d in datasets])
    
    # Create heatmap
    sns.heatmap(deviation_matrix.T, annot=True, fmt=".3f", cmap="RdBu_r", 
                xticklabels=datasets, yticklabels=[str(d) for d in range(1, 10)],
                ax=ax, center=0, cbar_kws={'label': 'Deviation from Benford'})

    ax.set_xlabel('Dataset')
    ax.set_ylabel('First Digit')
    ax.set_title('Per-Digit Deviation Heatmap (Observed - Expected)')
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "deviation_heatmap.pdf", dpi=150)
    print(f"[Plot Summary: deviation_heatmap.pdf] Heatmap showing magnitude and direction of deviation for each digit across datasets.")

def main():
    """Main execution pipeline."""
    print("Starting Benford's Law Conformity Analysis...")

    # 1. Load and Filter Data
    try:
        df = load_and_filter_data(DATASET_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("No valid data to analyze.")
        return

    # 2. Analyze Real Data Variables
    results = {}
    
    for var_code, label in VARIABLE_MAP.items():
        subset = df[df['value_variable_code'] == var_code]['value_float'].values
        
        if len(subset) > 0:
            metrics = compute_benford_metrics(subset)
            results[var_code] = {
                'label': label,
                **metrics
            }
            print(f"Variable [{var_code} - {label}]: N={metrics['count']}, MAD={metrics['mad']:.4f}, Class={metrics['classification']}")
        else:
            results[var_code] = {'label': label, 'count': 0, 'mad': np.nan, 'chi2_stat': np.nan, 'p_value': np.nan, 'classification': 'No Data'}

    # 3. Generate and Analyze Synthetic Controls
    n_samples = int(df['value_float'].notna().sum()) # Use total valid count as reference size
    
    uniform_digits, biased_digits = generate_synthetic_data(n_samples)
    
    results['Synth-Uniform'] = {
        'label': 'Synthetic Uniform',
        **compute_benford_metrics(uniform_digits)
    }
    print(f"Variable [Synth-Uniform]: N={results['Synth-Uniform']['count']}, MAD={results['Synth-Uniform']['mad']:.4f}, Class={results['Synth-Uniform']['classification']}")

    results['Synth-Biased'] = {
        'label': 'Synthetic Biased',
        **compute_benford_metrics(biased_digits)
    }
    print(f"Variable [Synth-Biased]: N={results['Synth-Biased']['count']}, MAD={results['Synth-Biased']['mad']:.4f}, Class={results['Synth-Biased']['classification']}")

    # 4. Generate Plots
    create_plots(results)

    # 5. Save Results to JSON
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            'label': v['label'],
            'count': int(v['count']),
            'mad': float(v['mad']) if not np.isnan(v['mad']) else None,
            'chi2_stat': float(v['chi2_stat']) if not np.isnan(v['chi2_stat']) else None,
            'p_value': float(v['p_value']) if not np.isnan(v['p_value']) else None,
            'classification': v['classification']
        }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    print("Analysis complete.")

if __name__ == "__main__":
    main()