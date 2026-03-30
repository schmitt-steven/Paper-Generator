# Paper Specification

## General Information

### Topic
Empirical Validation of Benford's Law as an Anomaly Detection Tool: An Analysis of German Municipal Data from the Zensus 2022.

Benford's Law predicts the probability of leading digit d (1-9):
P(d) = log₁₀(1 + 1/d)

Yielding: d=1: 30.1%, d=2: 17.6%, d=3: 12.5%, ..., d=9: 4.6%

The Mean Absolute Deviation (MAD) for Benford conformity is defined as: MAD = (1/9) × Σ|P_observed(d) - P_expected(d)| for d = 1..9

When computing deviation metrics such as MAD, calculations must operate on proportions (observed proportion vs. expected proportion), not raw counts. Raw count comparisons scale with sample size and render fixed conformity thresholds meaningless. Additionally, for large samples (N > 10,000), chi-square tests suffer from excess statistical power — even trivially small, scientifically irrelevant deviations will produce significant p-values. The analysis should account for this and not rely on chi-square as the sole conformity criterion.

Dataset:
- **Source:** Statistisches Bundesamt — Zensus 2022, Zensusdatenbank
- **Table:** Bevölkerung kompakt (Gebietsstand 15.05.2022), Gemeinde-level
- **Format:** CSV (semicolon-delimited), flat structure with quality flags
- **Variables:**
  - Population Count (`Einwohnerzahl` ) — primary variable
  - Area, qkm (`Fläche`) — secondary variable
  - Population Density, Ew/qkm (`Bevölkerungsdichte`) — tertiary variable (a derived ratio)
- **Sample size:** ~10,800 German municipalities
- **Quality flags:** `e` = exact, `()` = possibly altered by Cell-Key method, `.` = suppressed. Exclude non-exact values. Note: population counts are always unaltered per Destatis policy.

### Hypothesis
Population counts and municipal area measurements from the German Zensus 2022 conform to Benford's Law, while population density (a derived ratio) and synthetically generated data deviate significantly. This demonstrates Benford's Law's effectiveness as an anomaly detection tool for official statistics.

## Section Requirements

### Abstract
3-4 sentences: problem, approach, key results, implications.

### Introduction
Introduce Benford's Law, its mathematical basis (include the core formula), and why it applies to naturally occurring data spanning multiple orders of magnitude. Motivate the study: official statistics underpin policy and financial decisions, so integrity validation matters. State the research question and contributions.

### Related Work
Cover: Newcomb (1881) and Benford (1938), mathematical foundations (scale invariance), empirical applications to population/census data, use in fraud detection (financial auditing), the controversy around election data analysis, and relevant statistical testing approaches.

### Methods
The pipeline should determine and justify appropriate statistical conformity tests. At minimum, the methodology must include:
- Data preprocessing (parsing, filtering by quality flags, first-digit extraction)
- At least two complementary statistical tests for Benford conformity (the system should identify and justify which tests are suitable, and discuss why relying on a single test may be problematic for large samples)
- Generation of synthetic control datasets: one with uniform digit distribution (P(d) = 1/9), and one simulating human fabrication bias (e.g., overrepresentation of a specific digit like 5 to simulate psychological anchoring toward round numbers)
- Comparative analysis across all real and synthetic variables

**Required Listing:**
Include a concise pseudocode algorithm block (10-20 lines max) describing the high-level Benford conformity analysis pipeline. Must be appropriate for a methodology section. The pseudocode should cover the key stages (data filtering, first-digit extraction, frequency computation, MAD/Chi-square calculation, classification) at an abstracted level.

### Results

**Variables tested:** Population, Area, Density, Synthetic-Uniform, Synthetic-Biased

**Required Plots:**
1. **First-Digit Distribution Bar Chart:** Observed vs. expected Benford frequencies for each dataset, with Benford's expected distribution overlaid as reference.
2. **Conformity Comparison Chart:** Visual comparison of the primary conformity metric across all five datasets, with accepted threshold levels clearly marked.
3. **Per-Digit Deviation Heatmap:** Digits 1-9 vs. datasets, color-coded by deviation magnitude, highlighting statistically significant deviations.

**Required Tables:**
- Summary table of all test statistics across all five datasets with conformity classification.

### Discussion
Analyze why real variables conform or deviate. Address the challenge of statistical testing on large samples (N ≈ 10,800). Discuss limitations (single country, single census year, simplistic synthetic controls). Suggest extensions (longitudinal comparison with Zensus 2011, cross-country analysis, application to financial data).

### Conclusion
2-3 sentences summarizing key findings and practical implications for anomaly detection in official statistics.