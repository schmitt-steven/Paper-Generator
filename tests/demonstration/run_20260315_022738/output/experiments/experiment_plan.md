# Experiment Plan: Benford's Law Forensic Analysis of German Zensus 2022 Data

## 1. Objective and Success Criteria
**Objective:** To empirically validate the applicability of Benford's Law as a forensic tool for detecting anomalies in official municipal statistics (Zensus 2022) by comparing first-digit frequency distributions across population counts, area measurements, derived density ratios, and synthetic control datasets.

**Success Criteria:**
1.  **Population Counts & Area:** Observed leading digit frequencies must exhibit low Mean Absolute Deviation (MAD), consistent with the theoretical Benford distribution ($P(d) = \log_{10}(1 + 1/d)$).
2.  **Derived Density:** Must show significantly higher MAD values compared to raw counts, reflecting the statistical properties of ratios and potential rounding artifacts.
3.  **Synthetic Controls:** Uniform synthetic data must yield a high MAD (approx. 0.167), while biased synthetic data (anchored on digit 5) must show distinct deviation patterns.
4.  **Robustness:** The analysis must demonstrate that for large sample sizes ($N \approx 32,000$), the Chi-square test yields significant p-values even for trivial deviations, justifying the reliance on MAD as the primary conformity metric.

## 2. Mathematical Formulas & Technical Details
*   **Benford's Expected Probability:**
    $$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$
    Where $d \in \{1, 2, ..., 9\}$.
    *Expected values:* $d=1: 30.1\%$, $d=2: 17.6\%$, $d=3: 12.5\%$, $d=4: 9.7\%$, $d=5: 7.9\%$, $d=6: 6.7\%$, $d=7: 5.8\%$, $d=8: 5.1\%$, $d=9: 4.6\%$.

*   **Mean Absolute Deviation (MAD):**
    $$MAD = \frac{1}{9} \sum_{d=1}^{9} |P_{observed}(d) - P_{expected}(d)|$$
    *Note:* $P_{observed}$ is the proportion of leading digits in the dataset, not raw counts.

*   **Chi-Square Goodness-of-Fit:**
    $$\chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d}$$
    Where $O_d$ is observed count and $E_d$ is expected count ($N \times P(d)$).

*   **Conformity Thresholds (Schechter, 2015):**
    *   MAD < 0.009: Highly suspicious (likely fabricated)
    *   0.009 ≤ MAD < 0.012: Suspect
    *   0.012 ≤ MAD < 0.025: Acceptable conformity
    *   MAD ≥ 0.025: Non-conforming (likely uniform or random)

## 3. Experiment Setup & Data Handling
*   **Data Source:** `datasets/1000A-0001_de_flat.csv` (Semicolon-delimited).
*   **Column Mapping:**
    *   Population Count: Filter rows where `value_variable_label` is "Personen" and `value_unit` is "Anzahl". Map to label "Population Count".
    *   Area: Filter rows where `value_variable_label` is "Fläche" and `value_unit` is "qkm". Map to label "Municipal Area (sq km)".
    *   Density: Filter rows where `value_variable_label` is "Bevölkerungsdichte" and `value_unit` is "Ew/qkm". Map to label "Population Density".
*   **Quality Filtering:** Strictly filter for `value_q == 'e'`. Exclude all other flags (e.g., cell-keyed or suppressed data).
*   **Preprocessing:**
    *   Convert `value` column to numeric, handling German decimal commas if present.
    *   Extract leading digit: For a number $x$, compute $\lfloor x / 10^{\lfloor \log_{10} |x| \rfloor} \rfloor$. Ignore values with no leading digit (e.g., 0 or negative).

## 4. Implementation Approach
1.  **Load & Filter:** Read CSV, filter for exact quality flags (`'e'`), and split into three subsets based on variable labels (Population, Area, Density).
2.  **Synthetic Generation:**
    *   Create a "Uniform" dataset: Random integers 1-9 with equal probability ($P=1/9$).
    *   Create a "Biased" dataset: Simulate human anchoring by forcing ~40% of digits to be '5' and distributing the rest randomly.
3.  **Digit Extraction:** For all datasets, compute the leading digit for every valid value.
4.  **Metric Calculation:** Compute observed proportions ($P_{obs}$), expected Benford proportions ($P_{exp}$), MAD, and Chi-square statistics for each dataset.
5.  **Visualization:** Generate three specific plots (saved as PDFs) without interactive windows:
    *   Bar chart comparing Observed vs. Expected for all datasets.
    *   Bar chart comparing MAD scores across all five datasets with threshold lines.
    *   Heatmap of deviations per digit for each dataset.
6.  **Output:** Print a summary table and conclusions to stdout.

## 5. Metrics to Measure
*   **MAD Score:** Primary indicator of conformity (scale-invariant).
*   **Chi-Square Statistic & P-value:** Secondary indicator; expected to show significance for all real datasets due to large $N$, illustrating the "power problem."
*   **Sample Size ($N$):** Number of valid entries per variable.

## 6. Output Requirements
*   **Console Output:** A concise table listing Dataset Name, Sample Size, MAD Score, Chi-Square Value, P-value, and Conformity Classification (e.g., "Benford-Compliant", "Non-Conforming").
*   **Plots (Saved as PDFs):**
    1.  `benford_digit_distribution.pdf`: Bar chart of digit frequencies.
    2.  `conformity_comparison.pdf`: MAD scores comparison with threshold markers.
    3.  `deviation_heatmap.pdf`: Heatmap of $|P_{obs} - P_{exp}|$ for digits 1-9 across datasets.

## 7. Headless Execution Constraints
*   Initialize SDL drivers to dummy mode before any plotting or potential game logic (though not used here, ensures robustness).
*   Use `plt.savefig()` for all figures; **never** call `plt.show()`.
*   Ensure all plot labels and titles are in English (e.g., "Population Count", "MAD Score").

## 8. Pseudocode Algorithm
```text
1. Load data from 'datasets/1000A-0001_de_flat.csv'
2. Filter rows where value_q == 'e' and variable_label in {Personen, Fläche, Bevölkerungsdichte}
3. For each variable:
    a. Extract leading digit for all numeric values > 0
    b. Compute observed frequency distribution (proportions)
    c. Calculate MAD = mean(|observed - benford_expected|)
    d. Calculate Chi-Square statistic and p-value
4. Generate Synthetic Uniform Data (random digits, P=1/9)
5. Generate Synthetic Biased Data (high frequency of digit 5)
6. Compute metrics for synthetic datasets using steps 3a-3d
7. Plot: Bar chart of Observed vs Expected Benford frequencies for all 5 datasets
8. Plot: MAD comparison bar chart with threshold lines at 0.012 and 0.025
9. Plot: Heatmap of deviations (digits x datasets)
10. Print summary table with classifications based on MAD thresholds
```