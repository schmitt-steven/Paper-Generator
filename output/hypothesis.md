            # Research Hypothesis

            ## Description
            Validating Benford's Law conformity for German municipal census variables by comparing observed first-digit frequencies against theoretical logarithmic distributions using Mean Absolute Deviation (MAD) and Chi-square statistics, while contrasting real population counts and area measurements with derived density ratios and synthetic control datasets.

            ## Rationale
            Standard conformity tests like Chi-square exhibit excess statistical power in large samples ($N 
eq 10,800$), where trivial deviations yield significant p-values regardless of practical relevance. Proportion-based metrics such as Mean Absolute Deviation (MAD) provide scale-invariant thresholds that distinguish between natural scale-invariant data and structurally anomalous entries. Derived ratios like population density are expected to violate the logarithmic distribution due to mathematical constraints, while synthetic uniform or biased distributions serve as known non-conforming baselines for calibrating anomaly detection sensitivity.

            ## Success Criteria
            Real variables (population counts and area measurements) exhibit first-digit frequency patterns closely aligned with the theoretical Benford distribution. Derived ratios (population density) and synthetic control datasets demonstrate distinct deviations from the expected logarithmic profile, confirming the method's ability to differentiate between naturally occurring data and structurally anomalous or fabricated entries.
