import os
import csv
import json
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure headless execution for plotting and potential pygame usage
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Use Agg backend to prevent GUI windows
plt.switch_backend('Agg')

# Set professional aesthetic style
sns.set_theme(style="whitegrid", palette="colorblind")

# Constants
DATASET_PATH = Path("datasets/1000A-0001_de_flat.csv")
PLOTS_DIR = Path("plots")
RESULTS_FILE = "results.json"

# Benford's Law Expected Probabilities for digits 1-9
BENFORD_EXPECTED = {d: math.log10(1 + 1/d) for d in range(1, 10)}
EXPECTED_VALUES = [BENFORD_EXPECTED[d] for d in range(1, 10)]

# Classification Thresholds (MAD)
THRESHOLD_CONFORMING = 0.015
THRESHOLD_SUSPICIOUS = 0.030

# Variable Mappings (English labels for output)
VARIABLE_LABELS = {
    "PRS018": "Population Count",
    "FLC001": "Area (km²)",
    "PRS017": "Population Density"
}

def load_and_filter_data(path: Path):
    """Loads CSV, filters by quality flag 'e', and extracts numeric values."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    # Load data
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    
    # Safety check: Print unique labels to ensure we are filtering correctly
    print("Available value_variable_codes:", sorted(df['value_variable_code'].unique()))
    print("Available value_q flags:", sorted(df['value_q'].unique()))
    
    # Filter by quality flag 'e' (exact data) and positive values
    df = df[df['value_q'] == 'e'].copy()
    
    if df.empty:
        raise ValueError("No records found with quality flag 'e'. Check dataset content.")

    # Parse numeric values, handling German comma decimals
    def parse_value(val):
        try:
            return float(str(val).replace(',', '.'))
        except (ValueError, TypeError):
            return None
    
    df['value_float'] = df['value'].apply(parse_value)
    
    # Filter out non-positive values and NaNs
    df = df[df['value_float'].notna() & (df['value_float'] > 0)]
    
    if df.empty:
        raise ValueError("No valid positive numeric records found after filtering.")

    return df

def extract_first_digit(value):
    """Extracts the leading digit of a positive number."""
    if value <= 0:
        return None
    # Use log10 to find magnitude, then divide to get first digit
    d = int(str(int(np.floor(math.log10(abs(value)))))) 
    # Alternative robust method for floating point:
    s = f"{value:.15g}" # Convert to string without scientific notation if possible
    if 'e' in s.lower():
        # Handle scientific notation manually if needed, though log10 is safer for large numbers
        exp = int(np.floor(math.log10(abs(value))))
        mantissa = abs(value) / (10 ** exp)
        return int(mantissa)
    
    # Fallback: string parsing
    s_clean = s.lstrip('0')
    if not s_clean or s_clean[0] == '.':
        return None
    return int(s_clean[0])

def compute_digit_frequencies(values):
    """Computes observed frequency of leading digits 1-9."""
    digits = []
    for v in values:
        d = extract_first_digit(v)
        if d is not None and 1 <= d <= 9:
            digits.append(d)
    
    counts = Counter(digits)
    total = len(digits)
    frequencies = {d: counts.get(d, 0) / total for d in range(1, 10)}
    return frequencies

def calculate_mad(frequencies):
    """Calculates Mean Absolute Deviation from Benford's Law."""
    deviations = [abs(frequencies[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)]
    return sum(deviations) / 9.0

def classify_conformity(mad):
    """Classifies conformity based on MAD thresholds."""
    if mad < THRESHOLD_CONFORMING:
        return "Conforming"
    elif mad < THRESHOLD_SUSPICIOUS:
        return "Acceptable Deviation"
    else:
        return "Non-Conforming / Suspicious"

def generate_synthetic_data(n_samples, distribution_type):
    """Generates synthetic datasets for control."""
    if distribution_type == 'uniform':
        # Uniform distribution over digits 1-9
        digits = [random.randint(1, 9) for _ in range(n_samples)]
    elif distribution_type == 'biased':
        # Biased towards digit 5 (psychological anchoring)
        # P(5) ~ 0.3, others distributed equally among remaining 8 digits (~0.0875 each)
        digits = []
        for _ in range(n_samples):
            if random.random() < 0.3:
                digits.append(5)
            else:
                # Pick from {1,2,3,4,6,7,8,9} uniformly
                other_digits = [d for d in range(1, 10) if d != 5]
                digits.append(random.choice(other_digits))
    else:
        raise ValueError("Unknown distribution type")
    
    # Convert to float values (just the digit itself for simplicity, or scaled)
    # For Benford analysis, the magnitude doesn't matter as much as the leading digit distribution.
    # We'll use the digits directly as values 1-9.
    return [float(d) for d in digits]

def run_analysis():
    """Main execution pipeline."""
    
    # 1. Load Data
    print("Loading dataset...")
    df = load_and_filter_data(DATASET_PATH)
    
    # 2. Extract Variables
    datasets = {}
    
    # Population Count (PRS018)
    pop_df = df[df['value_variable_code'] == 'PRS018']
    if pop_df.empty:
        print("WARNING: No data found for Population Count (PRS018). Skipping.")
    else:
        datasets["Population Count"] = pop_df['value_float'].tolist()

    # Area (FLC001)
    area_df = df[df['value_variable_code'] == 'FLC001']
    if area_df.empty:
        print("WARNING: No data found for Area (FLC001). Skipping.")
    else:
        datasets["Area (km²)"] = area_df['value_float'].tolist()

    # Population Density (PRS017)
    density_df = df[df['value_variable_code'] == 'PRS017']
    if density_df.empty:
        print("WARNING: No data found for Population Density (PRS017). Skipping.")
    else:
        datasets["Population Density"] = density_df['value_float'].tolist()

    # 3. Generate Synthetic Controls
    n_samples = max(len(v) for v in datasets.values()) if datasets else 10800
    print(f"Generating synthetic controls (N={n_samples})...")
    
    datasets["Synthetic Uniform"] = generate_synthetic_data(n_samples, 'uniform')
    datasets["Synthetic Biased"] = generate_synthetic_data(n_samples, 'biased')

    # 4. Compute Metrics
    results = {}
    all_frequencies = {}
    
    for name, values in datasets.items():
        freqs = compute_digit_frequencies(values)
        mad = calculate_mad(freqs)
        classification = classify_conformity(mad)
        
        results[name] = {
            "sample_size": len(values),
            "mad": round(mad, 6),
            "classification": classification,
            "frequencies": freqs
        }
        all_frequencies[name] = freqs
        
        print(f"{name}: MAD={mad:.5f} ({classification})")

    # 5. Save Results to JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    return results, all_frequencies

def create_plots(all_frequencies):
    """Generates and saves the three required plots."""
    
    # Ensure plots directory exists
    PLOTS_DIR.mkdir(exist_ok=True)
    
    datasets = list(all_frequencies.keys())
    digits = range(1, 10)
    
    # --- Plot 1: First-Digit Distribution Bar Chart ---
    plt.figure(figsize=(12, 6))
    width = 0.8 / len(datasets)
    x = np.arange(1, 10)
    
    for i, dataset in enumerate(datasets):
        freqs = all_frequencies[dataset]
        observed = [freqs[d] for d in digits]
        
        # Offset bars slightly for visibility if many datasets, or use transparency
        offset = (i - len(datasets)/2) * width
        plt.bar(x + offset, observed, width=width, label=VARIABLE_LABELS.get(dataset, dataset), alpha=0.8)

    # Overlay Benford's Expected Distribution
    plt.plot(x, EXPECTED_VALUES, 'k-', linewidth=2.5, marker='o', label="Benford's Law (Expected)")
    
    plt.xlabel("Leading Digit", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("First-Digit Distribution: Observed vs Benford's Law", fontsize=14)
    plt.xticks(digits)
    plt.legend(loc='best', framealpha=0.9)
    
    # Save Plot 1
    plot_path = PLOTS_DIR / "benford_distribution.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Summary: benford_distribution.pdf] Bar chart comparing observed leading digit frequencies for all datasets against Benford's Law expected distribution.")
    plt.close()

    # --- Plot 2: Conformity Comparison Chart (MAD) ---
    mad_values = [results[ds]['mad'] for ds in datasets]
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6'] # Colorblind friendly
    
    bars = plt.bar(datasets, mad_values, color=colors[:len(datasets)], edgecolor='black')
    
    # Add threshold lines
    plt.axhline(y=THRESHOLD_CONFORMING, color='green', linestyle='--', label=f"Conforming Threshold (MAD < {THRESHOLD_CONFORMING})")
    plt.axhline(y=THRESHOLD_SUSPICIOUS, color='red', linestyle='--', label=f"Suspicious Threshold (MAD > {THRESHOLD_SUSPICIOUS})")
    
    # Annotate bars with values
    for bar, val in zip(bars, mad_values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001, 
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Mean Absolute Deviation (MAD)", fontsize=12)
    plt.title("Conformity Comparison: MAD Scores Across Datasets", fontsize=14)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc='upper right', framealpha=0.9)
    
    # Save Plot 2
    plot_path = PLOTS_DIR / "mad_comparison.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Summary: mad_comparison.pdf] Bar chart showing MAD scores for all datasets with threshold lines indicating conformity levels.")
    plt.close()

    # --- Plot 3: Per-Digit Deviation Heatmap ---
    deviation_matrix = []
    for dataset in datasets:
        row = [abs(all_frequencies[dataset][d] - BENFORD_EXPECTED[d]) for d in digits]
        deviation_matrix.append(row)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(deviation_matrix, xticklabels=digits, yticklabels=[VARIABLE_LABELS.get(ds, ds) for ds in datasets], 
                cmap="YlOrRd", annot=True, fmt=".3f", cbar_kws={'label': 'Absolute Deviation'})
    
    plt.xlabel("Leading Digit", fontsize=12)
    plt.ylabel("Dataset", fontsize=12)
    plt.title("Per-Digit Deviation Heatmap: |Observed - Expected|", fontsize=14)
    
    # Save Plot 3
    plot_path = PLOTS_DIR / "deviation_heatmap.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Summary: deviation_heatmap.pdf] Heatmap visualizing the magnitude of deviation for each digit across all datasets.")
    plt.close()

if __name__ == "__main__":
    try:
        results, frequencies = run_analysis()
        create_plots(frequencies)
        
        # Final Summary Output
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for name, data in results.items():
            print(f"{name:20} | MAD: {data['mad']:.6f} | Class: {data['classification']}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise