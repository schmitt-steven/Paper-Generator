# Experiment Plan: Benford's Law Forensic Analysis of German Zensus 2022 Data

## 1. Objective and Success Criteria
**Objective:** To empirically validate the applicability of Benford's Law as a forensic tool for detecting anomalies in official German municipal statistics (Zensus 2022). The experiment will compare the first-digit frequency distributions of three distinct variables—Population Count, Municipal Area, and Population Density—against theoretical expectations and synthetic control datasets.

**Success Criteria:**
*   **Conformity:** Population counts and Area measurements must exhibit low Mean Absolute Deviation (MAD) values, indicating strong alignment with Benford's Law ($P(d) = \log_{10}(1 + 1/d)$).
*   **Differentiation:** Derived variables (Density) and synthetic controls (Uniform distribution, Human-fabricated bias) must show statistically significant deviations or distinct MAD profiles compared to the natural aggregates.
*   **Robustness:** The analysis must demonstrate that MAD is a superior metric for large samples ($N \approx 10,800$) compared to Chi-square tests, which are prone to flagging trivial deviations as significant due to high statistical power.

## 2. Mathematical Formulas and Technical Details
*   **Benford's Expected Probability:** $P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$ for $d \in \{1, \dots, 9\}$.
    *   Values: $d=1 (30.1\%), d=2 (17.6\%), \dots, d=9 (4.6\%)$.
*   **Mean Absolute Deviation (MAD):**
    $$ \text{MAD} = \frac{1}{9} \sum_{d=1}^{9} | P_{observed}(d) - P_{expected}(d) | $$
    *   *Note:* Calculations must be performed on **proportions** (frequencies normalized to sum to 1), not raw counts, to ensure scale invariance.
*   **Chi-Square Goodness-of-Fit:**
    $$ \chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d} $$
    *   Used for supplementary analysis but interpreted with caution due to sample size sensitivity ($N > 10,000$).
*   **Synthetic Distributions:**
    *   *Uniform:* $P(d) = 1/9 \approx 11.1\%$ for all digits.
    *   *Biased (Anchoring):* Simulated overrepresentation of digit '5' to mimic human rounding/fabrication biases.

## 3. Experiment Setup
**Data Source:** `datasets/1000A-0001_de_flat.csv` (Semicolon-delimited).
**Variables to Extract:**
1.  **Population Count:** Filter rows where `value_variable_label` is 'Personen' and `value_unit` is 'Anzahl'. Map internal column `value` to "Population".
2.  **Area:** Filter rows where `value_variable_label` is 'Fläche' and `value_unit` is 'qkm'. Map internal column `value` to "Area".
3.  **Density:** Filter rows where `value_variable_label` is 'Bevölkerungsdichte' and `value_unit` is 'Ew/qkm'. Map internal column `value` to "Density".

**Preprocessing Steps:**
1.  Load CSV using `pd.read_csv('datasets/1000A-0001_de_flat.csv', sep=';')`.
2.  **Filter Quality Flags:** Retain only rows where `value_q` equals `'e'` (exact). Exclude values marked with parentheses or dots to ensure data integrity.
3.  **Data Cleaning:** Convert the `value` column to numeric, handling potential formatting issues (e.g., commas in decimals if present), and drop non-numeric entries.
4.  **First-Digit Extraction:** For each valid positive number $x$, compute the leading digit: $d = \lfloor x / 10^{\lfloor \log_{10}(x) \rfloor} \rfloor$. Exclude zeros or negative numbers if any remain after filtering.
5.  **Synthetic Generation:** Programmatically generate two synthetic datasets of comparable size ($N \approx 3,600$ each):
    *   One with digits uniformly distributed (1-9).
    *   One with a bias toward digit '5' (e.g., 40% probability for '5', rest uniform).

## 4. Metrics to Measure
*   **Primary Metric:** Mean Absolute Deviation (MAD) calculated on observed vs. expected proportions.
*   **Secondary Metric:** Chi-Square statistic and associated p-value (for comparative discussion).
*   **Classification Thresholds:** Define standard MAD thresholds:
    *   $< 0.012$: Suspect (High conformity, likely natural or fabricated perfectly).
    *   $\leq 0.05$: Acceptable (Natural variation).
    *   $> 0.05$: Unacceptable (Significant deviation).

## 5. Implementation Approach
**Algorithmic Pipeline:**
1.  **Load & Filter:** Read CSV, filter for `value_q == 'e'`, and separate into three subsets based on variable labels.
2.  **Digit Extraction:** Apply logarithmic extraction to get leading digits (1-9) for all valid entries in each subset.
3.  **Frequency Calculation:** Compute the proportion of each digit $d$ for Population, Area, Density, and both synthetic sets.
4.  **Metric Computation:** Calculate MAD and Chi-Square for each set against Benford's theoretical distribution.
5.  **Comparison & Classification:** Compare observed MADs against thresholds and classify conformity status (Natural vs. Anomalous).
6.  **Visualization Generation:** Create three specific plots:
    *   Bar chart of digit frequencies with Benford overlay.
    *   Bar chart comparing MAD scores across all five datasets.
    *   Heatmap of per-digit deviations for all datasets.

**Critical Constraints:**
*   **Headless Execution:** Set `os.environ['SDL_VIDEODRIVER'] = 'dummy'` and `os.environ['SDL_AUDIODRIVER'] = 'dummy'` at startup.
*   **No Interactive Displays:** Use `plt.savefig('filename.pdf')` for all plots; do not call `plt.show()`.
*   **Language:** All plot labels, titles, legends, and console output must be in English (translate German terms like "Fläche" to "Area", etc.).
*   **Performance:** Ensure the script completes within 5 minutes by avoiding unnecessary iterations or complex simulations.

## 6. Output Requirements
**Console Output:**
*   Summary table displaying: Dataset Name, Sample Size ($N$), MAD Score, Chi-Square Value, P-Value, and Conformity Classification (e.g., "Natural", "Anomalous").
*   Brief conclusion on whether Benford's Law holds for the specific variables tested.

**Visualizations (Saved as PDFs):**
1.  **`benford_digit_distribution.pdf`:** Bar chart showing observed vs. expected frequencies for all five datasets, with Benford's curve overlaid.
2.  **`conformity_comparison.pdf`:** Bar chart comparing the MAD metric across Population, Area, Density, Synthetic-Uniform, and Synthetic-Biased, with threshold lines marked.
3.  **`deviation_heatmap.pdf`:** Heatmap visualizing $|P_{observed} - P_{expected}|$ for digits 1-9 across all datasets to highlight specific digit anomalies.

## 7. Pseudocode Algorithm
```text
BEGIN Benford_Analysis_Pipeline
    LOAD data from 'datasets/1000A-0001_de_flat.csv' (semicolon delimiter)
    
    FILTER rows where value_q == 'e' AND value is numeric
    
    SEPARATE into subsets: Pop, Area, Density based on variable_label
    
    FOR each subset in [Pop, Area, Density]:
        EXTRACT leading digit d for every valid positive value
        COMPUTE observed proportion P_obs(d) = count(d) / total_count
        
    GENERATE Synthetic_Uniform: Random digits 1-9 with equal probability (P=1/9)
    GENERATE Synthetic_Biased: Digits biased towards '5' (e.g., P(5)=0.4, others uniform)
    
    DEFINE Benford_Expected(d) = log10(1 + 1/d) for d in 1..9
    
    FOR each dataset in [Pop, Area, Density, Synthetic_Uniform, Synthetic_Biased]:
        COMPUTE MAD = (1/9) * sum(|P_obs(d) - Benford_Expected(d)|)
        COMPUTE Chi2 = sum((O_d - E_d)^2 / E_d) where E_d = N * Benford_Expected(d)
        CLASSIFY conformity based on MAD thresholds
        
    GENERATE Plot 1: Bar chart of P_obs vs. Benford_Expected for all datasets
    GENERATE Plot 2: Bar chart comparing MAD scores across all datasets
    GENERATE Plot 3: Heatmap of |P_obs - Benford_Expected| per digit
    
    PRINT Summary Table with metrics and classifications
    SAVE plots as .pdf files (no interactive display)
END Benford_Analysis_Pipeline
```