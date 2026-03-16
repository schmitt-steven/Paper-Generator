# Experiment Plan: Benford's Law Forensic Analysis of German Zensus 2022 Data

## 1. Objective and Success Criteria
**Objective:** To empirically validate the applicability of Benford's Law as a forensic tool for detecting anomalies in official German municipal statistics (Zensus 2022). The experiment will compare the first-digit frequency distributions of three real-world variables against theoretical expectations and synthetic baselines to distinguish natural data patterns from potential fabrication or derived-ratio artifacts.

**Success Criteria:**
1.  **Conformity Confirmation:** Population counts (`Personen`) and Municipal Area (`Fläche`) must exhibit low Mean Absolute Deviation (MAD) values, indicating strong conformity with Benford's Law ($P(d) = \log_{10}(1 + 1/d)$).
2.  **Anomaly Detection:** Derived variables (Population Density `Bevölkerungsdichte`) and synthetic controls (Uniform distribution, Human Bias) must exhibit significantly higher MAD values, demonstrating the method's ability to flag non-natural distributions.
3.  **Robustness:** The analysis must successfully filter out data points with quality flags other than 'e' (exact), ensuring only high-integrity records are analyzed.
4.  **Metric Validity:** The experiment must demonstrate that MAD is a more interpretable metric for large samples ($N \approx 10,800$) than Chi-square tests, which may yield false positives due to excessive statistical power.

## 2. Required Mathematical Formulas & Technical Details
*   **Benford's Expected Probability:** $P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$ for $d \in \{1, ..., 9\}$.
    *   Values: $d=1 (30.1\%), d=2 (17.6\%), d=3 (12.5\%), d=4 (9.7\%), d=5 (7.9\%), d=6 (6.7\%), d=7 (5.8\%), d=8 (5.1\%), d=9 (4.6\%)$.
*   **Mean Absolute Deviation (MAD):** $MAD = \frac{1}{9} \sum_{d=1}^{9} |P_{observed}(d) - P_{expected}(d)|$, where $P_{observed}$ is the proportion of observations starting with digit $d$.
*   **Chi-Square Goodness-of-Fit (Supplementary):** $\chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d}$, where $O_d$ is observed count and $E_d$ is expected count ($N \times P(d)$).
*   **First-Digit Extraction:** For a value $x$, the first digit $d = \lfloor x / 10^{\lfloor \log_{10}|x| \rfloor} \rfloor$. Values $\le 0$ or non-numeric are excluded.

## 3. Experiment Setup
*   **Data Source:** `datasets/1000A-0001_de_flat.csv` (Zensus 2022, Gemeinde-level).
*   **Variables to Analyze:**
    1.  **Population Count:** Filter rows where `value_variable_label` == 'Personen' and `value_q` == 'e'. Extract numeric values from `value`.
    2.  **Municipal Area:** Filter rows where `value_variable_label` == 'Fläche' and `value_q` == 'e'. Extract numeric values from `value`.
    3.  **Population Density:** Filter rows where `value_variable_label` == 'Bevölkerungsdichte' and `value_q` == 'e'. Extract numeric values from `value`.
*   **Synthetic Controls (Generated In-Memory):**
    1.  **Uniform:** Randomly generate integers 1-9 with equal probability ($P(d)=1/9$). Size $N=10,800$.
    2.  **Human Bias:** Generate integers favoring '5' and '6' (psychological anchoring) to simulate fabrication. Size $N=10,800$.
*   **Environment Configuration:**
    *   Initialize headless mode for plotting: `os.environ['SDL_VIDEODRIVER'] = 'dummy'`, `os.environ['SDL_AUDIODRIVER'] = 'dummy'`.
    *   Ensure no interactive plots are shown; all figures saved to `.pdf`.

## 4. Metrics to Measure
*   **Primary Metric:** Mean Absolute Deviation (MAD). Interpretation: $<0.01$ indicates excellent conformity, $0.01-0.02$ good, $>0.05$ suspicious.
*   **Secondary Metric:** Chi-Square Statistic and p-value (for comparison purposes only).
*   **Sample Size ($N$):** Total valid records per variable to contextualize the statistical power.

## 5. Implementation Approach
1.  **Data Loading & Filtering:** Load CSV using `pd.read_csv('datasets/1000A-0001_de_flat.csv', sep=';')`. Filter for `value_q == 'e'` and specific variable labels ('Personen', 'Fläche', 'Bevölkerungsdichte'). Convert `value` column to numeric, coercing errors to NaN.
2.  **First-Digit Extraction:** Create a function to extract the leading digit from valid positive numbers. Apply this to all three real variables.
3.  **Synthetic Generation:** Generate two synthetic datasets (Uniform and Biased) with $N=10,800$ using `numpy.random`.
4.  **Statistical Computation:** For each of the 5 datasets:
    *   Compute digit frequencies ($d=1..9$).
    *   Convert to proportions.
    *   Calculate MAD against Benford's expected probabilities.
    *   Calculate Chi-Square statistic and p-value (for reference).
5.  **Visualization Generation:**
    *   **Plot 1 (Bar Chart):** Grouped bar chart comparing Observed vs. Expected frequencies for all datasets, with Benford curve overlaid.
    *   **Plot 2 (Conformity Comparison):** Bar chart of MAD values for all 5 datasets with threshold lines ($0.01$, $0.02$, $0.05$).
    *   **Plot 3 (Heatmap):** Matrix showing deviation magnitude per digit vs. dataset, color-coded by absolute difference.
6.  **Output:** Print a summary table of MAD and Chi-Square results. Save all plots as `benford_analysis.pdf`.

## 6. Pseudocode Algorithm
```text
1. LOAD data from 'datasets/1000A-0001_de_flat.csv' (semicolon-delimited).
2. FILTER rows where value_q == 'e'.
3. EXTRACT numeric values for variables: Population, Area, Density.
4. GENERATE synthetic datasets: Uniform(1-9) and Biased(5-heavy), size N=10800.
5. FOR each dataset (Real + Synthetic):
    a. COMPUTE first digit d for every value > 0.
    b. CALCULATE observed proportion P_obs(d) for d in 1..9.
    c. DEFINE expected proportion P_exp(d) = log10(1 + 1/d).
    d. CALCULATE MAD = (1/9) * sum(|P_obs - P_exp|).
    e. CALCULATE Chi-Square statistic and p-value.
6. GENERATE Plot 1: Bar chart of Observed vs Expected frequencies.
7. GENERATE Plot 2: Bar chart of MAD values with threshold markers.
8. GENERATE Plot 3: Heatmap of |P_obs - P_exp| per digit/dataset.
9. PRINT summary table of all metrics and conformity classification.
10. SAVE plots to 'benford_analysis.pdf' (headless mode).
```

## 7. Output Requirements
*   **Console Output:** A concise text block displaying the MAD, Chi-Square value, p-value, and a qualitative conclusion ("Conforms", "Suspicious", or "Anomalous") for each of the five datasets.
*   **Files Generated:** `benford_analysis.pdf` containing three distinct visualizations as described in Section 5.
*   **Constraints:** All text (labels, titles) must be in English. No interactive windows will appear. Execution time must remain under 5 minutes.