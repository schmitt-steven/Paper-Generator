# Experiment Plan: Benford's Law Conformity Analysis of Zensus 2022 Data

## 1. Objective and Success Criteria
**Objective:** To validate the hypothesis that naturally occurring population counts and physical area measurements in German municipal data (Zensus 2022) conform to Benford's Law, while derived ratios (population density) and synthetic controls exhibit significant deviations. The experiment aims to demonstrate that Mean Absolute Deviation (MAD) is a more robust metric for large-sample anomaly detection than Chi-square tests, which suffer from inflated power in this context.

**Success Criteria:**
1.  **Conformity:** Population counts (`PRS018`) and Area measurements (`FLC001`) must show low MAD values (typically < 0.02) and first-digit distributions visually overlapping the theoretical Benford curve.
2.  **Deviation:** Derived density ratios (`PRS017`) and synthetic datasets (Uniform, Fabricated) must exhibit significantly higher MAD values and distinct visual deviations from the expected distribution.
3.  **Metric Robustness:** The experiment will demonstrate that Chi-square p-values for real data are likely significant due to sample size ($N \approx 10,800$), whereas MAD provides a scale-invariant measure of practical conformity.

## 2. Mathematical Formulas & Technical Details
*   **Benford's Expected Probability:** $P(d) = \log_{10}(1 + \frac{1}{d})$ for $d \in \{1, ..., 9\}$.
    *   Values: $d=1 (30.1\%), d=2 (17.6\%), d=3 (12.5\%), ..., d=9 (4.6\%)$.
*   **Mean Absolute Deviation (MAD):**
    $$ \text{MAD} = \frac{1}{9} \sum_{d=1}^{9} | P_{\text{observed}}(d) - P_{\text{expected}}(d) | $$
    *   *Note:* Calculations must use proportions (counts / total valid count), not raw counts.
*   **Chi-Square Statistic:**
    $$ \chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d} $$
    *   Where $O_d$ is observed count and $E_d$ is expected count ($N \times P(d)$).
*   **First-Digit Extraction:** For a value $x > 0$, the first digit $d = \lfloor x / 10^{\lfloor \log_{10} x \rfloor} \rfloor$.

## 3. Experiment Setup & Data Handling
**Environment Configuration (Headless):**
*   Initialize `os.environ['SDL_VIDEODRIVER']` and `SDL_AUDIODRIVER` to 'dummy' to prevent display errors if any libraries attempt GUI initialization.
*   Disable interactive plotting: All figures must be saved via `plt.savefig('filename.pdf')` without calling `plt.show()`.

**Data Loading & Preprocessing:**
1.  **Pathing:** Load data from `datasets/1000A-0001_de_flat.csv`.
2.  **Parsing:** Use semicolon delimiter. Handle locale-specific formatting (replace `,` with `.` in numeric strings).
3.  **Filtering:**
    *   Select rows where `value_variable_code` is one of: `'PRS018'` (Population), `'FLC001'` (Area), `'PRS017'` (Density).
    *   Filter strictly for quality flag `value_q == 'e'` (exact values) to exclude suppressed or cell-key altered data.
    *   Discard non-numeric entries and zero/negative values before digit extraction.
4.  **Variable Mapping:** Map internal codes to English labels:
    *   `PRS018` $\rightarrow$ "Population Count"
    *   `FLC001` $\rightarrow$ "Area (km²)"
    *   `PRS017` $\rightarrow$ "Population Density"

**Synthetic Data Generation:**
*   **Uniform Control:** Generate $N=10,800$ random integers where each digit 1-9 has probability $1/9$.
*   **Fabricated Control:** Generate data with a bias toward specific digits (e.g., overrepresenting '5' and underrepresenting '4') to simulate human fabrication.

## 4. Metrics to Measure
For each dataset (Real: Pop, Area, Density; Synthetic: Uniform, Fabricated):
1.  **First-Digit Frequency:** Observed percentage for digits 1-9.
2.  **MAD Score:** The calculated Mean Absolute Deviation from Benford's Law.
3.  **Chi-Square Statistic & P-value:** To demonstrate the "over-sensitivity" of this test in large samples.
4.  **Conformity Classification:** Based on MAD thresholds (e.g., < 0.012 = "Highly Conformant", > 0.05 = "Non-Conformant").

## 5. Implementation Approach
**Step-by-Step Logic:**
1.  **Initialize:** Set environment variables for headless execution. Import `pandas`, `numpy`, `matplotlib`, `scipy.stats`.
2.  **Load & Clean:** Read CSV, filter by variable codes and quality flag 'e', convert strings to floats.
3.  **Digit Extraction:** Create a helper function to extract the first significant digit from positive numbers. Apply to all valid values in each category.
4.  **Synthetic Generation:** Programmatically create two additional datasets (Uniform, Biased) with matching sample sizes.
5.  **Statistical Computation:**
    *   Compute observed frequencies for digits 1-9.
    *   Calculate MAD using the formula above.
    *   Calculate Chi-square statistic and p-value.
6.  **Visualization Generation:**
    *   *Plot 1 (Bar Chart):* Grouped bar chart comparing Observed vs. Expected Benford distribution for all 5 datasets.
    *   *Plot 2 (Conformity Comparison):* Bar chart of MAD scores with threshold lines indicating "Acceptable" ranges.
    *   *Plot 3 (Heatmap):* Matrix showing deviation magnitude per digit per dataset.
7.  **Output:** Print a summary table of statistics and save plots as `.pdf` files in the current directory.

**Pseudocode Algorithm:**
```text
1. Initialize environment for headless execution.
2. Load CSV from 'datasets/1000A-0001_de_flat.csv'.
3. Filter rows where value_variable_code IN {PRS018, FLC001, PRS017} AND value_q == 'e'.
4. For each variable group:
    a. Convert 'value' to float (handle comma decimals).
    b. Extract first digit d for all positive values.
    c. Compute observed proportion P_obs(d) = count(d) / total_count.
5. Generate Synthetic Uniform and Biased datasets with same N.
6. For each dataset (Real + Synthetic):
    a. Calculate MAD = (1/9) * sum(|P_obs(d) - Benford(d)|).
    b. Calculate Chi-square statistic and p-value.
7. Classify conformity based on MAD thresholds.
8. Generate three plots: Digit Distribution, MAD Comparison, Deviation Heatmap.
9. Save plots to .pdf files; print summary table to stdout.
```

## 6. Output Requirements
*   **Stdout:** A concise text report containing the sample size, calculated MAD scores, Chi-square p-values, and a final conclusion on which datasets conform to Benford's Law.
*   **Files:** Three PDF plots saved as:
    *   `benford_digit_distribution.pdf` (Bar chart of frequencies).
    *   `conformity_comparison_mad.pdf` (MAD scores comparison).
    *   `deviation_heatmap.pdf` (Heatmap of per-digit deviations).

## 7. Constraints & Optimization
*   **Time Limit:** The entire pipeline must execute in under 5 minutes.
*   **Optimization:** Use vectorized NumPy operations for digit extraction and frequency counting to avoid slow Python loops over the ~10,800 rows.
*   **Safety:** Ensure no `plt.show()` calls are present; all figures are saved immediately after creation.