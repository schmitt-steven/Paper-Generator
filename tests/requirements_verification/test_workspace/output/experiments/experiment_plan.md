# Experiment Plan: Benford's Law Conformity Analysis of German Municipal Census Data (Zensus 2022)

## 1. Objective and Success Criteria
**Objective:** To empirically validate the hypothesis that primary municipal census variables (Population Count, Area) conform to Benford's Law, while derived ratios (Population Density) and synthetic datasets exhibit significant deviations, using Mean Absolute Deviation (MAD) as a robust metric insensitive to large sample sizes.

**Success Criteria:**
1.  **Conformity Ranking:** The calculated MAD for "Population Count" and "Area" must be significantly lower than the MAD for "Population Density," "Synthetic Uniform," and "Synthetic Biased."
2.  **Statistical Distinction:** The deviation of derived/synthetic data from Benford's expected distribution must be visually and numerically distinct from primary variables, confirming the method's sensitivity to non-natural scaling.
3.  **Execution Constraints:** The entire pipeline (loading, filtering, synthetic generation, calculation, plotting) must complete in under 5 minutes on standard hardware without interactive windows.

## 2. Required Mathematical Formulas & Technical Details
*   **Benford's Expected Probability ($P_{expected}$):**
    $$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$
    Where $d \in \{1, 2, ..., 9\}$.
    *Expected values:* $d=1: 30.1\%$, $d=2: 17.6\%$, $d=3: 12.5\%$, $d=4: 9.7\%$, $d=5: 7.9\%$, $d=6: 6.7\%$, $d=7: 5.8\%$, $d=8: 5.1\%$, $d=9: 4.6\%$.

*   **Mean Absolute Deviation (MAD):**
    $$MAD = \frac{1}{9} \sum_{d=1}^{9} |P_{observed}(d) - P_{expected}(d)|$$
    *Note:* $P_{observed}$ is the proportion of leading digits in the dataset, not raw counts.

*   **First Digit Extraction:**
    For a value $x > 0$:
    $$d = \lfloor x / 10^{\lfloor \log_{10}(x) \rfloor} \rfloor$$
    (Extracts the leading digit of the absolute value).

*   **Synthetic Data Generation:**
    *   **Uniform:** Random integer $d \in \{1..9\}$ with probability $1/9$.
    *   **Biased:** Distribution favoring digit 5 (e.g., $P(5) = 0.3$, others distributed equally among remaining).

## 3. Experiment Setup
*   **Data Source:** `datasets/1000A-0001_de_flat.csv` (Semicolon-delimited, UTF-8-sig).
*   **Variables to Analyze:**
    1.  **Population Count:** Filter `value_variable_code == 'PRS018'` and `value_unit == 'Anzahl'`.
    2.  **Area:** Filter `value_variable_code == 'FLC001'` and `value_unit == 'qkm'`.
    3.  **Population Density:** Filter `value_variable_code == 'PRS017'` and `value_unit == 'Ew/qkm'`.
*   **Data Preprocessing:**
    *   Parse numeric values, handling German comma decimals (replace `,` with `.`).
    *   **Filtering:** Exclude records where `value_q != 'e'` (exclude suppressed or altered data per quality flags).
    *   **Zero Handling:** Discard non-positive values ($x \le 0$) as Benford's Law applies to positive numbers.
*   **Synthetic Controls:** Generate two datasets of size $N=10,800$ (matching the real sample size) for Uniform and Biased distributions.

## 4. Metrics to Measure
*   **Primary Metric:** Mean Absolute Deviation (MAD). Lower MAD indicates better conformity.
*   **Secondary Metric:** Chi-Square Statistic ($\chi^2$) calculated on observed vs. expected counts, primarily for illustrative comparison of "excess power" in large samples (though not used as the primary decision criterion).
*   **Classification:** Based on thresholds:
    *   MAD < 0.015: Suspected conformity (Benford-compliant).
    *   0.015 ≤ MAD < 0.030: Acceptable deviation.
    *   MAD ≥ 0.030: Suspicious/Non-conforming.

## 5. Implementation Approach
1.  **Environment Setup:** Set `SDL_VIDEODRIVER` and `SDL_AUDIODRIVER` to 'dummy' to ensure headless execution if any underlying libraries attempt GUI initialization. Import `matplotlib.pyplot` with non-interactive backend.
2.  **Data Loading & Parsing:**
    *   Load CSV using `pandas` or `csv` module (leveraging existing `analyze_population.py` logic for parsing).
    *   Filter rows based on `value_variable_code` and `value_q`.
    *   Convert `value` strings to floats, handling locale-specific formatting.
3.  **Digit Extraction & Frequency Calculation:**
    *   For each valid dataset (Real: Pop, Area, Density; Synthetic: Uniform, Biased):
        *   Extract leading digit for every positive value.
        *   Compute frequency distribution $P_{observed}(d)$.
4.  **Statistical Computation:**
    *   Calculate MAD for each dataset using the formula above.
    *   (Optional) Calculate $\chi^2$ for comparison purposes only.
5.  **Visualization Generation:**
    *   **Plot 1 (Bar Chart):** Grouped bar chart showing Observed vs. Expected frequencies for all 5 datasets side-by-side or overlaid with transparency. X-axis: Digits 1-9, Y-axis: Frequency (%).
    *   **Plot 2 (Conformity Comparison):** Bar chart of MAD values for all 5 datasets. Add horizontal lines marking the "Acceptable" and "Suspicious" thresholds.
    *   **Plot 3 (Heatmap):** Matrix with Digits (1-9) on Y-axis, Datasets on X-axis. Color intensity represents $|P_{observed} - P_{expected}|$.
6.  **Output:**
    *   Save all plots as `.pdf` files in the current directory.
    *   Print a concise summary table to stdout containing: Dataset Name, Sample Size, MAD Value, and Conformity Classification.

## 6. Pseudocode Algorithm
```text
1. Define Benford Expected Proportions E[d] = log10(1 + 1/d) for d=1..9
2. Load CSV from 'datasets/1000A-0001_de_flat.csv'
3. Filter records where value_q == 'e' and value > 0
4. Extract three subsets: Population (PRS018), Area (FLC001), Density (PRS017)
5. Generate Synthetic Uniform (S_U) and Biased (S_B) datasets of size N=10,800
6. For each dataset D in [Pop, Area, Density, S_U, S_B]:
    a. Extract leading digit for all values -> list L
    b. Compute observed proportions O[d] = count(L==d) / len(L)
    c. Calculate MAD_D = (1/9) * sum(|O[d] - E[d]| for d=1..9)
7. Classify each D based on MAD_D thresholds
8. Generate Plot 1: Bar chart of O vs E for all datasets
9. Generate Plot 2: Bar chart of MAD values with threshold lines
10. Generate Plot 3: Heatmap of |O[d] - E[d]| across digits and datasets
11. Save plots as .pdf files
12. Print summary table to stdout
```

## 7. Output Requirements
*   **Console:** A clean, formatted text table summarizing the MAD values and classification for all five datasets.
*   **Files:** Three PDF plots (`benford_distribution.pdf`, `mad_comparison.pdf`, `deviation_heatmap.pdf`) saved in the working directory.
*   **Execution Mode:** Strictly headless (no windows), no interactive prompts, no `plt.show()`.