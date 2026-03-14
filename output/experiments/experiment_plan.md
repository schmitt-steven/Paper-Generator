# Experiment Plan: Benford's Law Conformity Analysis for German Municipal Census Data (Zensus 2022)

## 1. Objective and Success Criteria
**Objective:** To validate the efficacy of Benford's Law as an anomaly detection tool for official statistics by analyzing first-digit frequencies in German municipal census data. The experiment aims to demonstrate that natural variables (Population, Area) conform to the logarithmic distribution, while derived ratios (Density) and synthetic fabricated datasets deviate significantly, thereby establishing a robust forensic framework using Mean Absolute Deviation (MAD).

**Success Criteria:**
1.  **Conformity of Natural Data:** Population counts and Area measurements must yield MAD values below the "acceptable" threshold (< 0.015), indicating strong adherence to Benford's Law.
2.  **Deviation of Derived Ratios:** Population Density (a derived ratio) must exhibit a significantly higher MAD value, confirming that mathematical constraints disrupt scale invariance.
3.  **Detection of Fabrication:** Synthetic Uniform and Biased datasets must show the highest MAD values, clearly distinguishable from real data, validating the method's sensitivity to non-natural patterns.
4.  **Statistical Robustness:** The experiment must successfully mitigate "excess statistical power" issues in large samples ($N \approx 10,800$) by prioritizing proportion-based MAD over raw Chi-square p-values for classification.

## 2. Mathematical Formulas and Technical Details
*   **Benford's Expected Probability:** $P(d) = \log_{10}(1 + \frac{1}{d})$ for $d \in \{1, ..., 9\}$.
    *   Values: $d=1 (30.1\%), d=2 (17.6\%), d=3 (12.5\%), d=4 (9.7\%), d=5 (7.9\%), d=6 (6.7\%), d=7 (5.8\%), d=8 (5.1\%), d=9 (4.6\%)$.
*   **Mean Absolute Deviation (MAD):** $MAD = \frac{1}{9} \sum_{d=1}^{9} |P_{observed}(d) - P_{expected}(d)|$.
    *   *Note:* Calculations must use proportions ($count / N$), not raw counts.
*   **Chi-Square Statistic:** $\chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d}$, where $O_d$ is observed count and $E_d$ is expected count ($N \times P(d)$).
    *   *Usage:* Computed for completeness but interpreted with caution due to large $N$.
*   **Classification Thresholds (Tibshirani):**
    *   MAD < 0.015: Good conformity.
    *   0.015 ≤ MAD < 0.025: Acceptable conformity.
    *   MAD ≥ 0.025: Suspect/Non-conforming.

## 3. Experiment Setup
*   **Environment:** Python script running in headless mode (no GUI, no interactive plots).
*   **Data Loading:** Load `datasets/1000A-0001_de_flat.csv` using pandas with semicolon delimiter.
*   **Filtering Logic:**
    *   Parse the `value_q` column (quality flag).
    *   Retain only rows where `value_q == 'e'` (exact values).
    *   Exclude rows with `()` or `.` flags to ensure data integrity.
*   **Variable Extraction:**
    *   Filter for specific variables: `PRS018` (Population), `FLC001` (Area), `PRS017` (Density).
    *   Extract the numeric value from the `value` column, handling potential formatting issues (e.g., commas in German numbers like "1.234,56" or spaces). Convert to float.
    *   Filter out non-positive values ($\le 0$) as Benford's Law applies to positive integers/decimals.
*   **Synthetic Data Generation:**
    *   Generate a Uniform distribution (random digits 1-9 with equal probability).
    *   Generate a Biased distribution (simulate human fabrication, e.g., over-representing digit '5' or '0').

## 4. Metrics to Measure
*   **MAD Score:** Primary metric for conformity assessment.
*   **Chi-Square Statistic & P-value:** Secondary metric for statistical significance (to demonstrate the "excess power" issue).
*   **First-Digit Frequencies:** Observed proportions for each digit 1-9 per dataset.
*   **Conformity Classification:** Categorical label based on MAD thresholds ("Good", "Acceptable", "Suspect").

## 5. Implementation Approach
1.  **Initialization:** Set environment variables `SDL_VIDEODRIVER` and `SDL_AUDIODRIVER` to 'dummy' (safety measure). Import necessary libraries (`pandas`, `numpy`, `matplotlib`).
2.  **Data Ingestion & Cleaning:**
    *   Read CSV from `datasets/1000A-0001_de_flat.csv`.
    *   Clean numeric columns: Remove commas/spaces, convert to float.
    *   Filter rows where `value_q == 'e'` and variable code matches the target list.
3.  **First-Digit Extraction:**
    *   For each valid positive number $x$, compute $d = \lfloor x / 10^{\lfloor \log_{10}(x) \rfloor} \rfloor$.
    *   Aggregate counts for digits 1-9.
4.  **Synthetic Data Creation:**
    *   Create two synthetic datasets of comparable size ($N$) with uniform and biased digit distributions.
5.  **Statistical Computation:**
    *   Calculate observed proportions $P_{obs}(d)$ for all 5 datasets (3 real, 2 synthetic).
    *   Compute MAD using the formula above.
    *   Compute Chi-square statistics.
6.  **Visualization Generation:**
    *   **Plot 1 (Bar Chart):** Grouped bar chart comparing Observed vs. Expected Benford frequencies for all datasets.
    *   **Plot 2 (Conformity Comparison):** Bar chart of MAD scores with horizontal threshold lines at 0.015 and 0.025.
    *   **Plot 3 (Heatmap):** Matrix showing deviation magnitude ($|P_{obs} - P_{exp}|$) for digits 1-9 across datasets.
7.  **Output:** Save plots as `.pdf` files in the current directory. Print a summary table of statistics and conclusions to stdout.

## 6. Output Requirements
*   **Console Output:** A concise text report containing:
    *   Sample sizes after filtering.
    *   MAD scores for all variables.
    *   Chi-square p-values (with a note on their interpretation).
    *   Final classification (Conforming/Suspect) for each variable.
*   **Files Generated:**
    *   `benford_first_digit_distribution.pdf`
    *   `benford_mad_comparison.pdf`
    *   `benford_deviation_heatmap.pdf`

## 7. Pseudocode Algorithm
```text
1. Initialize environment for headless execution (SDL dummy drivers).
2. Load dataset from 'datasets/1000A-0001_de_flat.csv'.
3. Filter rows: Keep only where value_q == 'e' AND variable in [Population, Area, Density].
4. Clean numeric values: Remove non-numeric chars, convert to float, discard <= 0.
5. Define Benford Expected Probs P_exp = log10(1 + 1/d) for d=1..9.
6. For each variable (Pop, Area, Density):
    a. Extract leading digit d for every value.
    b. Compute observed proportions P_obs(d).
    c. Calculate MAD = mean(|P_obs - P_exp|).
    d. Calculate Chi-square statistic.
7. Generate Synthetic Uniform and Biased datasets; repeat step 6.
8. Classify results using MAD thresholds (<0.015 Good, <0.025 Acceptable, else Suspect).
9. Generate three plots: (a) Observed vs Expected Bars, (b) MAD Comparison with thresholds, (c) Deviation Heatmap.
10. Save plots as .pdf files and print summary table to stdout.
```

## 8. Execution Constraints
*   **Time Limit:** Must complete within 5 minutes. The dataset size (~32k rows) is small enough for direct pandas operations; no sampling required, but ensure vectorized operations are used.
*   **Headless Mode:** No `plt.show()` calls. All figures must be saved via `fig.savefig()`.
*   **Pathing:** Strictly use `datasets/` prefix for file loading.