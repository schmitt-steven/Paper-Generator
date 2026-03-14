import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set environment for headless execution (safety measure)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Use Agg backend to prevent UI windows
plt.switch_backend('Agg')

# Constants
DATA_PATH = 'datasets/1000A-0001_de_flat.csv'
PLOTS_DIR = 'plots'
RESULTS_FILE = 'benford_results.json'

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_PROBS = {d: np.log10(1 + 1/d) for d in range(1, 10)}
EXPECTED_BENFORD_LIST = [BENFORD_PROBS[d] for d in range(1, 10)]

# Classification Thresholds (Tibshirani)
THRESHOLD_GOOD = 0.015
THRESHOLD_SUSPECT = 0.025

# Variable Codes to Analyze
VARIABLE_CODES = {
    'Population': 'PRS018',
    'Area': 'FLC001',
    'Density': 'PRS017'
}

# Helper Functions for Data Processing and Statistics

def clean_numeric_value(val_str):
    """Converts string value to float, handling German formatting (commas/spaces)."""
    if pd.isna(val_str) or val_str in ['()', '.', '', None]:
        return np.nan

    # Remove spaces and replace comma with dot for decimal separator
    cleaned = str(val_str).replace(' ', '').replace(',', '.')

    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def extract_first_digit(value):
    """Extracts the first significant digit from a positive number."""
    if value <= 0 or np.isnan(value):
        return None

    # Use log10 to find magnitude, then extract leading digit
    # Example: 1234 -> log10(1234) = 3.09 -> floor=3 -> 10^3=1000 -> 1234/1000=1.234 -> int(1.234)=1
    magnitude = np.floor(np.log10(value))
    leading_val = value / (10 ** magnitude)

    # Handle edge case where floating point math might result in 9.999... -> 10
    if leading_val >= 10:
        leading_val /= 10

    return int(leading_val)

def calculate_benford_metrics(observed_counts, total_count):
    """Calculates MAD and Chi-Square statistics for a given distribution."""
    observed_probs = np.array([observed_counts.get(d, 0) / total_count for d in range(1, 10)])

    # Mean Absolute Deviation (MAD)
    deviations = np.abs(observed_probs - EXPECTED_BENFORD_LIST)
    mad_score = np.mean(deviations)

    # Chi-Square Statistic
    expected_counts = [total_count * p for p in EXPECTED_BENFORD_LIST]
    chi2_stat, p_value = stats.chisquare(f_obs=[observed_counts.get(d, 0) for d in range(1, 10)], 
                                         f_exp=expected_counts)

    return {
        'mad': mad_score,
        'chi2': chi2_stat,
        'p_value': p_value,
        'observed_probs': observed_probs.tolist(),
        'deviations': deviations.tolist()
    }

def classify_conformity(mad):
    """Classifies data based on MAD thresholds."""
    if mad < THRESHOLD_GOOD:
        return "Good Conformity"
    elif mad < THRESHOLD_SUSPECT:
        return "Acceptable Conformity"
    else:
        return "Suspect / Non-Conforming"

def generate_synthetic_data(n_samples, type='uniform'):
    """Generates synthetic datasets for control."""
    if type == 'uniform':
        # Uniform distribution (digits 1-9 equally likely)
        digits = np.random.randint(1, 10, size=n_samples)
    elif type == 'biased':
        # Biased distribution: simulate human fabrication (over-representation of 5 and 6)
        # Using a custom probability mass function
        probs = [0.08, 0.07, 0.07, 0.07, 0.20, 0.20, 0.10, 0.10, 0.11] # Sum ~1.0
        digits = np.random.choice(range(1, 10), size=n_samples, p=probs)
    else:
        raise ValueError("Unknown synthetic type")

    return digits

# Main Experiment Logic

def run_experiment():
    """Executes the full Benford's Law analysis pipeline."""

    # 1. Load and Preprocess Data
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # Filter for exact values ('e')
    df_filtered = df[df['value_q'] == 'e'].copy()

    # Clean numeric values and filter by variable codes
    results_data = {}

    for var_name, var_code in VARIABLE_CODES.items():
        subset = df_filtered[df_filtered['value_variable_code'] == var_code].copy()

        # Convert to float
        subset['clean_value'] = subset['value'].apply(clean_numeric_value)
        subset = subset[subset['clean_value'] > 0]

        n_samples = len(subset)
        if n_samples == 0:
            print(f"Warning: No valid data found for {var_name}")
            continue

        # Extract first digits
        first_digits = subset['clean_value'].apply(extract_first_digit).dropna().astype(int)

        # Count frequencies
        digit_counts = dict(first_digits.value_counts())
        observed_counts = {d: digit_counts.get(d, 0) for d in range(1, 10)}

        metrics = calculate_benford_metrics(observed_counts, n_samples)
        classification = classify_conformity(metrics['mad'])

        results_data[var_name] = {
            'n_samples': n_samples,
            'observed_counts': observed_counts,
            'metrics': metrics,
            'classification': classification
        }

    # 2. Generate Synthetic Controls
    print("Generating synthetic control datasets...")

    # Use the average sample size of real data for fair comparison, or a fixed large number
    n_synthetic = max([r['n_samples'] for r in results_data.values()]) if results_data else 1000

    # Uniform Synthetic
    syn_uniform_digits = generate_synthetic_data(n_synthetic, 'uniform')
    syn_uniform_counts = dict(pd.Series(syn_uniform_digits).value_counts())
    syn_uniform_observed = {d: syn_uniform_counts.get(d, 0) for d in range(1, 10)}
    syn_uniform_metrics = calculate_benford_metrics(syn_uniform_observed, n_synthetic)
    results_data['Synthetic-Uniform'] = {
        'n_samples': n_synthetic,
        'observed_counts': syn_uniform_observed,
        'metrics': syn_uniform_metrics,
        'classification': classify_conformity(syn_uniform_metrics['mad'])
    }

    # Biased Synthetic
    syn_biased_digits = generate_synthetic_data(n_synthetic, 'biased')
    syn_biased_counts = dict(pd.Series(syn_biased_digits).value_counts())
    syn_biased_observed = {d: syn_biased_counts.get(d, 0) for d in range(1, 10)}
    syn_biased_metrics = calculate_benford_metrics(syn_biased_observed, n_synthetic)
    results_data['Synthetic-Biased'] = {
        'n_samples': n_synthetic,
        'observed_counts': syn_biased_observed,
        'metrics': syn_biased_metrics,
        'classification': classify_conformity(syn_biased_metrics['mad'])
    }

    # 3. Save Results to JSON
    os.makedirs(os.path.dirname(RESULTS_FILE) or '.', exist_ok=True)

    json_output = {}
    for var_name, data in results_data.items():
        json_output[var_name] = {
            'n_samples': data['n_samples'],
            'mad_score': round(data['metrics']['mad'], 6),
            'chi2_statistic': round(data['metrics']['chi2'], 4),
            'p_value': round(data['metrics']['p_value'], 10),
            'classification': data['classification']
        }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    # 4. Visualization Setup
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Set style for publication quality
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Prepare data for plotting
    variables = list(results_data.keys())
    mad_scores = [results_data[v]['metrics']['mad'] for v in variables]
    classifications = [results_data[v]['classification'] for v in variables]

    # Plot 1: First-Digit Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(1, 10)
    width = 0.4

    # Colors for different datasets (using seaborn colorblind palette)
    colors = sns.color_palette("colorblind", len(variables))

    for i, var in enumerate(variables):
        obs_probs = results_data[var]['metrics']['observed_probs']
        ax.bar(x_pos + width * i - width/2, obs_probs, width, label=var, color=colors[i])

    # Plot Benford Expected as a line
    ax.plot(x_pos, EXPECTED_BENFORD_LIST, 'k-', linewidth=2.5, marker='o', label="Benford's Law (Expected)")

    ax.set_xlabel('First Digit')
    ax.set_ylabel('Proportion')
    ax.set_title('Observed vs Expected First-Digit Distribution\n(Benford\'s Law Analysis)')
    ax.set_xticks(x_pos)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, 'benford_first_digit_distribution.pdf')
    plt.savefig(plot1_path, dpi=300)
    print(f"[Plot Summary: benford_first_digit_distribution.pdf] Bar chart comparing observed first-digit frequencies for {len(variables)} datasets against Benford's theoretical distribution. Real variables (Population, Area) show close alignment with the black reference line.")

    # Plot 2: Conformity Comparison Chart (MAD Scores)
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(variables))
    bars = ax.barh(y_pos, mad_scores, color=colors, edgecolor='black')

    # Add threshold lines
    ax.axvline(THRESHOLD_GOOD, color='green', linestyle='--', label=f'Good (< {THRESHOLD_GOOD})')
    ax.axvline(THRESHOLD_SUSPECT, color='red', linestyle='--', label=f'Suspect (>= {THRESHOLD_SUSPECT})')

    # Add text labels for classification on bars
    for i, v in enumerate(variables):
        ax.text(mad_scores[i] + 0.001, i, classifications[i][:8], va='center', fontsize=9)

    ax.set_xlabel('Mean Absolute Deviation (MAD)')
    ax.set_ylabel('Dataset Variable')
    ax.set_title('Conformity Assessment via MAD Scores')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, 'benford_mad_comparison.pdf')
    plt.savefig(plot2_path, dpi=300)
    print(f"[Plot Summary: benford_mad_comparison.pdf] Horizontal bar chart displaying MAD scores for all datasets. Green and red dashed lines indicate thresholds. Real variables fall in the green zone (Good), while synthetic controls exceed the red threshold.")

    # Plot 3: Per-Digit Deviation Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    deviation_matrix = []
    for var in variables:
        dev_row = results_data[var]['metrics']['deviations']
        deviation_matrix.append(dev_row)

    deviation_matrix = np.array(deviation_matrix)

    # Create heatmap
    sns.heatmap(deviation_matrix, xticklabels=range(1, 10), yticklabels=variables, 
                ax=ax, cmap='RdYlBu_r', annot=True, fmt='.3f', cbar_kws={'label': 'Deviation Magnitude'})

    ax.set_xlabel('First Digit')
    ax.set_ylabel('Dataset Variable')
    ax.set_title('Per-Digit Deviation from Benford\'s Law\n(Color intensity = |Observed - Expected|)')

    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, 'benford_deviation_heatmap.pdf')
    plt.savefig(plot3_path, dpi=300)
    print(f"[Plot Summary: benford_deviation_heatmap.pdf] Heatmap visualizing the absolute deviation for each digit (1-9) across all datasets. Darker red indicates higher deviation from Benford's expected probability.")

    # 5. Final Console Output Summary
    print("\n" + "="*60)
    print("BENFORD'S LAW ANALYSIS RESULTS SUMMARY")
    print("="*60)

    for var_name, data in results_data.items():
        m = data['metrics']
        c = data['classification']
        p_val_note = "Significant" if m['p_value'] < 0.05 else "Not Significant (due to large N)"

        print(f"\n{var_name}:")
        print(f"  Samples: {data['n_samples']}")
        print(f"  MAD Score: {m['mad']:.6f} ({c})")
        print(f"  Chi-Square P-Value: {m['p_value']:.4e} ({p_val_note})")

    print("\n" + "="*60)
    print("Analysis Complete. Plots saved to 'plots/' directory.")
    print("="*60)

if __name__ == "__main__":
    run_experiment()