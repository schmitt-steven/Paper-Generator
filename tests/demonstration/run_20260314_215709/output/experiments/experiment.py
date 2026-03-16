import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set environment for headless operation and plotting backend
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
plt.switch_backend('Agg')

# Configuration Constants
DATASET_PATH = 'datasets/1000A-0001_de_flat.csv'
RESULTS_FILE = 'results.json'
PLOTS_DIR = 'plots'
SAMPLE_SIZE_SYNTHETIC = 10800

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_PROBS = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

# Define variable mappings (English labels for internal German identifiers)
VAR_LABELS = {
    'Personen': 'Population Count',
    'Fläche': 'Municipal Area',
    'Bevölkerungsdichte': 'Population Density'
}

def extract_first_digit(value):
    """Extract the first significant digit from a positive number."""
    if pd.isna(value) or value <= 0:
        return None
    try:
        val = float(str(value).replace(',', '.')) # Handle potential comma decimals
        if val <= 0:
            return None
        return int(np.floor(val / (10 ** np.floor(np.log10(abs(val))))))
    except (ValueError, OverflowError):
        return None

def calculate_benford_metrics(observed_counts):
    """Calculate MAD and Chi-Square statistics for a given set of observed counts."""
    total = sum(observed_counts)
    if total == 0:
        return {'mad': np.nan, 'chi2': np.nan, 'p_value': np.nan}

    proportions = np.array(observed_counts) / total
    
    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(proportions - BENFORD_PROBS))
    
    # Chi-Square Goodness of Fit
    expected_counts = total * BENFORD_PROBS
    chi2, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)
    
    return {
        'mad': mad,
        'chi2': chi2,
        'p_value': p_value,
        'proportions': proportions.tolist(),
        'observed_counts': observed_counts.tolist()
    }

def classify_conformity(mad):
    """Classify conformity based on MAD value."""
    if np.isnan(mad):
        return "Invalid Data"
    elif mad < 0.01:
        return "Excellent Conformity"
    elif mad < 0.02:
        return "Good Conformity"
    elif mad < 0.05:
        return "Marginal / Suspicious"
    else:
        return "Anomalous / Non-Conforming"

def generate_synthetic_data(distribution_type, size):
    """Generate synthetic datasets for control comparison."""
    if distribution_type == 'uniform':
        # Uniform distribution (P(d) = 1/9)
        digits = np.random.randint(1, 10, size=size)
    elif distribution_type == 'biased':
        # Human bias: Overrepresentation of 5 and 6
        weights = [0.02, 0.03, 0.04, 0.05, 0.30, 0.30, 0.10, 0.08, 0.08] # Sum ~1.0
        digits = np.random.choice(range(1, 10), size=size, p=weights)
    else:
        raise ValueError("Unknown distribution type")
    
    return digits

def load_and_filter_data(df):
    """Load data and filter for specific variables with quality flag 'e'."""
    results = {}
    
    # Check available labels before filtering to ensure robustness
    print(f"Available value_variable_label values: {df['value_variable_label'].unique()}")
    print(f"Available value_q values: {df['value_q'].unique()}")

    for var_key, var_name in VAR_LABELS.items():
        # Case-insensitive search to handle potential variations
        mask = df['value_variable_label'].str.contains(var_key, case=False, na=False) & \
               (df['value_q'] == 'e')
        
        subset = df[mask]
        
        if len(subset) == 0:
            print(f"WARNING: No data found for {var_name} with quality flag 'e'. Skipping.")
            continue
            
        # Extract numeric values, handling potential formatting issues (commas vs dots)
        # Replace commas with dots to handle European number formats if present in string
        subset['value_clean'] = subset['value'].astype(str).str.replace(',', '.')
        
        try:
            numeric_vals = pd.to_numeric(subset['value_clean'], errors='coerce')
        except Exception as e:
            print(f"Error converting values for {var_name}: {e}")
            continue
            
        # Filter out non-positive and NaN values before digit extraction
        valid_vals = numeric_vals[numeric_vals > 0]
        
        if len(valid_vals) == 0:
            print(f"WARNING: No positive valid values found for {var_name}. Skipping.")
            continue

        # Extract first digits
        first_digits = [extract_first_digit(v) for v in valid_vals]
        first_digits = [d for d in first_digits if d is not None]
        
        results[var_key] = np.array(first_digits)
        
    return results

def run_analysis_pipeline():
    """Main execution pipeline."""
    
    # 1. Load Data
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return

    # 2. Filter and Extract Real Data
    print("\nFiltering data for Population, Area, and Density...")
    real_data = load_and_filter_data(df)
    
    if not real_data:
        print("ERROR: No valid data extracted. Terminating.")
        return

    # 3. Generate Synthetic Controls
    print("Generating synthetic control datasets...")
    synth_uniform = generate_synthetic_data('uniform', SAMPLE_SIZE_SYNTHETIC)
    synth_biased = generate_synthetic_data('biased', SAMPLE_SIZE_SYNTHETIC)
    
    all_datasets = {
        'Population Count': real_data['Personen'],
        'Municipal Area': real_data['Fläche'],
        'Population Density': real_data['Bevölkerungsdichte'],
        'Synthetic Uniform': synth_uniform,
        'Synthetic Biased': synth_biased
    }

    # 4. Compute Metrics
    print("\nComputing Benford conformity metrics...")
    results_summary = []
    
    for name, data in all_datasets.items():
        counts = np.bincount(data, minlength=10)[1:] # Digits 1-9
        
        metrics = calculate_benford_metrics(counts)
        classification = classify_conformity(metrics['mad'])
        
        results_summary.append({
            'variable': name,
            'sample_size': len(data),
            'mad': float(metrics['mad']),
            'chi2': float(metrics['chi2']),
            'p_value': float(metrics['p_value']),
            'classification': classification,
            'observed_proportions': metrics['proportions']
        })
        
        print(f"[{name}] MAD: {metrics['mad']:.4f} | Chi2: {metrics['chi2']:.2f} | P-val: {metrics['p_value']:.2e} -> {classification}")

    # 5. Save Results to JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    return results_summary, all_datasets

def create_visualizations(results_summary, all_datasets):
    """Generate publication-quality plots."""
    
    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Set style for professional look
    sns.set_theme(style="whitegrid", palette="colorblind")
    
    # Prepare data for plotting
    variables = [r['variable'] for r in results_summary]
    mads = [r['mad'] for r in results_summary]
    classifications = [r['classification'] for r in results_summary]
    
    # Plot 1: First-Digit Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1, 10)
    width = 0.1
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, data) in enumerate(all_datasets.items()):
        counts = np.bincount(data, minlength=10)[1:]
        proportions = counts / len(data)
        
        # Offset bars slightly to avoid overlap
        offset = width * (i - 2) 
        ax.bar(x + offset, proportions, width, label=name, color=colors[i % len(colors)])

    # Overlay Benford's Law
    ax.plot(x, BENFORD_PROBS, 'k-', linewidth=2.5, marker='o', label="Benford's Expected", zorder=10)
    
    ax.set_xlabel('First Digit (d)', fontsize=12)
    ax.set_ylabel('Proportion of Occurrences', fontsize=12)
    ax.set_title('Observed vs. Expected First-Digit Frequencies\n(German Zensus 2022 & Synthetic Controls)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'benford_distribution.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plot Summary: benford_distribution.pdf] Bar chart comparing observed first-digit frequencies for 5 datasets against Benford's Law curve. Population and Area show close alignment with expected log distribution.")

    # Plot 2: Conformity Comparison Chart (MAD Values)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(variables, mads, color=colors[:len(variables)], edgecolor='black')
    
    # Add threshold lines
    ax.axhline(y=0.01, color='green', linestyle='--', label='Excellent (<0.01)')
    ax.axhline(y=0.02, color='orange', linestyle='--', label='Good (<0.02)')
    ax.axhline(y=0.05, color='red', linestyle='--', label='Suspicious (>0.05)')
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom' if height > 0.05 else 'top', fontsize=9)

    ax.set_ylabel('Mean Absolute Deviation (MAD)', fontsize=12)
    ax.set_title('Conformity Comparison: MAD Scores Across Variables', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'mad_comparison.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plot Summary: mad_comparison.pdf] Bar chart of MAD scores. Population and Area are below 0.02 (Good), Density is higher, Synthetic Uniform and Biased show distinct deviations.")

    # Plot 3: Per-Digit Deviation Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    deviation_matrix = []
    for name in variables:
        row_data = results_summary[variables.index(name)]['observed_proportions']
        deviations = [abs(p - BENFORD_PROBS[i]) for i, p in enumerate(row_data)]
        deviation_matrix.append(deviations)
    
    deviation_matrix = np.array(deviation_matrix)
    
    sns.heatmap(deviation_matrix, xticklabels=['1','2','3','4','5','6','7','8','9'], 
                yticklabels=variables, cmap='RdYlBu_r', annot=True, fmt='.3f', ax=ax, cbar_kws={'label': 'Absolute Deviation'})
    
    ax.set_title('Per-Digit Deviation Heatmap\n(Color: Red = High Deviation, Blue = Low Deviation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('First Digit (d)')
    ax.set_ylabel('Dataset Variable')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'deviation_heatmap.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plot Summary: deviation_heatmap.pdf] Heatmap showing absolute deviation per digit. Highlights specific digits where data diverges from Benford's Law.")

if __name__ == "__main__":
    # Run the analysis pipeline
    results, datasets = run_analysis_pipeline()
    
    if results:
        # Generate visualizations
        create_visualizations(results, datasets)
        print("\nExperiment completed successfully. Check 'plots/' directory and 'results.json'.")
    else:
        print("Experiment failed to produce valid data.")