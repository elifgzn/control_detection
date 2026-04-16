# Power Analysis Results

## Parameters

- Simulations per cell: 1000
- Alpha level: 0.05
- Target trials per condition: 60
- Foil trials total: 240
- Uncontrolled old items total: 120 (supplementary analyses only)
- Baseline hit rate: 0.62 (Wu et al., 2025, Exp. 4)
- Baseline false alarm rate: 0.2 (assumed)
- Baseline d-prime: 0.74 (avg of Schreiner et al. conditions)
- D-prime pooled SD: 0.75 (Schreiner et al.)
- Hit rate SD: 0.2 (Schreiner et al.)
- Random intercept SD: 1.36 (converted from d-prime SD via log-odds)

## Primary Analysis Descriptions

1. **D-prime paired t-test**: D-prime per participant per condition (loglinear correction), paired t-test. Standard SDT approach (Schreiner et al., 2024).
2. **Hit rate paired t-test**: Mean hit rate per participant per condition, paired t-test. Used in Wu et al. (2025).
3. **GEE hits only**: Trial-level GEE on target trials only. Predictor: control_level (+-0.5). Binomial family, logit link, exchangeable working correlation.
4. **GEE with interaction**: Trial-level GEE on all trials. item_type x control interaction captures d-prime difference. Foils dummy-assigned equally across conditions.

## Supplementary Analysis Descriptions

These analyses incorporate the uncontrolled items (seen simultaneously with controlled items but never under participant control) as a third condition.
**Theoretical note**: Uncontrolled items were viewed in the same context as controlled items, so they do not constitute a clean 'no encoding' baseline. Interpret with caution.
**Conservative assumption**: Uncontrolled item hit rate is set equal to the low control hit rate (worst case for detecting an effect of motor control per se).

S1. **D-prime one-way ANOVA**: Three d-primes per participant (high, low, uncontrolled), one-way repeated measures ANOVA. Tests omnibus effect.
S2. **Hit rate one-way ANOVA**: Three hit rates per participant, one-way repeated measures ANOVA.
S3. **GEE all old items**: Trial-level GEE on all old items. control_type as 3-level predictor (uncontrolled = reference). Tests high control vs uncontrolled contrast.
S4. **GEE interaction (3-level)**: Trial-level GEE on all trials. item_is_old x control_type interaction. Foils require no dummy assignment — cleanest version of Analysis 4.

## Effect Size Conversions

- Cohen's d = 0.2: delta d-prime = 0.150 (0.665 vs 0.815), delta hit rate = 0.040 (0.600 vs 0.640)
- Cohen's d = 0.5: delta d-prime = 0.375 (0.552 vs 0.927), delta hit rate = 0.100 (0.570 vs 0.670)

## Primary Analysis Results

### Small (d=0.2)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 1: d-prime paired t-test | 0.137 | 0.128 | 0.165 | 0.209 |
| Analysis 2: Hit rate paired t-test | 0.109 | 0.159 | 0.138 | 0.176 |
| Analysis 3: GEE hits only | 0.575 | 0.712 | 0.814 [80%+] | 0.853 [80%+] |
| Analysis 4: GEE with interaction | 0.433 | 0.492 | 0.596 | 0.684 |

[80%+] = meets 80% power threshold

### Medium (d=0.5)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 1: d-prime paired t-test | 0.439 | 0.574 | 0.681 | 0.771 |
| Analysis 2: Hit rate paired t-test | 0.450 | 0.586 | 0.675 | 0.784 |
| Analysis 3: GEE hits only | 0.999 [80%+] | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] |
| Analysis 4: GEE with interaction | 0.985 [80%+] | 0.995 [80%+] | 1.000 [80%+] | 1.000 [80%+] |

[80%+] = meets 80% power threshold

## Supplementary Analysis Results

### Small (d=0.2)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Supp 1: d-prime one-way ANOVA (3 conditions) | 0.121 | 0.128 | 0.145 | 0.172 |
| Supp 2: Hit rate one-way ANOVA (3 conditions) | 0.103 | 0.121 | 0.145 | 0.175 |
| Supp 3: GEE all old items (3-level control_type) | 0.721 | 0.817 [80%+] | 0.899 [80%+] | 0.949 [80%+] |
| Supp 4: GEE interaction (3-level, clean foils) | 0.951 [80%+] | 0.980 [80%+] | 0.942 [80%+] | 0.951 [80%+] |

[80%+] = meets 80% power threshold

### Medium (d=0.5)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Supp 1: d-prime one-way ANOVA (3 conditions) | 0.491 | 0.632 | 0.730 | 0.809 [80%+] |
| Supp 2: Hit rate one-way ANOVA (3 conditions) | 0.498 | 0.596 | 0.735 | 0.819 [80%+] |
| Supp 3: GEE all old items (3-level control_type) | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] |
| Supp 4: GEE interaction (3-level, clean foils) | 0.962 [80%+] | 0.981 [80%+] | 0.946 [80%+] | 0.954 [80%+] |

[80%+] = meets 80% power threshold

## Notes and Limitations

- **GEE**: Estimates population-average effects. Robust to misspecification of working correlation. Does not require specifying random slopes (advantageous with small N).
- **Random intercept SD**: Estimated indirectly from d-prime SD. Actual between-participant variability is unknown.
- **False alarm rate**: Assumed 0.20. Does not directly affect Analyses 1, 2, 3, S1, S2, or S3.
- **Foil dummy assignment**: Only relevant for primary Analysis 4 — resolved in Supplementary Analysis S4.
- **Effect size uncertainty**: Parameters from a choice paradigm (Schreiner et al.) — true effect in a control paradigm may differ.
- **Supplementary conservative assumption**: Uncontrolled hit rate = low control hit rate. Power for S1-S4 would be higher if uncontrolled items are remembered less well than low control items.
- **Recommendation**: For small effects (d=0.2), N=30-60 is likely underpowered regardless of analysis. Frame as exploratory. For medium effects (d=0.5), GEE analyses offer more power by leveraging trial-level variance.