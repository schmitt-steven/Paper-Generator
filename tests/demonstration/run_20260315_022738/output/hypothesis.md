# Research Hypothesis

## Description
Validating Benford's Law conformity in German Zensus 2022 municipal data by comparing first-digit frequencies of population counts and area measurements against derived population density and synthetic control datasets.

## Rationale
Naturally occurring large-scale demographic aggregates typically exhibit logarithmic leading digit distributions, whereas derived ratios and human-fabricated data often display uniform or biased patterns. Standard Chi-square tests lack specificity for large sample sizes due to excessive sensitivity to trivial deviations; therefore, Mean Absolute Deviation (MAD) on proportions provides a scale-invariant metric to distinguish natural variation from systematic anomalies.

## Success Criteria
Population counts and area measurements exhibit low MAD values consistent with Benford's expected distribution. Derived population density and synthetic datasets demonstrate significantly higher MAD values indicating deviation from the logarithmic baseline.
