# CDmem Power Analysis Report (Matched to Final Analyses)

**Iterations per simulation:** 1000
**Alpha level:** 0.05

## Methods

Power was estimated via Monte Carlo simulation (1,000 iterations per cell). Each of the seven planned analyses (1A–3G) was simulated separately. Baseline parameters (hit rates, FA rates, log-RT distributions, detection accuracy by condition) were derived from pilot data. The random intercept SD for binomial models (RI_SD = 0.656) was extracted from the pilot binomial GLMM (model 1B). For Gaussian LMM models, between-person SD (0.149) and within-person SD (0.241) were extracted directly from the pilot LMM random effects (model 1C). Effect sizes for RT models are scaled by TOTAL_SD_RT = sqrt(between² + within²) ≈ 0.284.

## Effect Size Notes

- **H1 & H2 analyses (1A–2E):** Effect sizes are Cohen's *d*, applied as mean differences scaled by the relevant outcome SD.
- **H3 analysis 3F (binomial):** Effect size is specified as a log-odds slope because agency_z is a continuous z-scored within-participant predictor. Approximate Cohen's *d* equivalents are provided via the conversion log-odds / 1.81 (Borenstein et al., 2009). These are smaller than the d = 0.20 / 0.50 used elsewhere because the effect operates at the trial level on a continuous predictor, not as a condition mean difference.
- **H3 analysis 3G (Gaussian):** Effect size is Cohen's *d*, consistent with H1/H2.

## GEE Approximation Note

The planned analyses use `lme4`-based mixed models (via pymer4). For computational efficiency, power simulations use Generalized Estimating Equations (GEE) with an exchangeable correlation structure as an approximation. GEE tends to yield slightly conservative power estimates for binary outcomes at small N; the results below should therefore be interpreted as a lower bound on the power of the planned mixed models.

## ANOVA Simulation Note

For analysis 1A, the control-level effect is applied equally to both controlled and uncontrolled items, consistent with H1 (a main effect of motor control on recognition). No interaction with item type is assumed; any such interaction is exploratory.

## Power Estimates

Columns: `Effect_Size_d_H1_H2` = Cohen's d for analyses 1A–2E and 3G. `Effect_Size_H3_logodds` = log-odds slope for 3F. `Effect_Size_d_H3_approx` = approximate Cohen's d equivalent for 3F.

|   N |   Effect_Size_d_H1_H2 |   Effect_Size_H3_logodds |   Effect_Size_d_H3_approx |   1A_ANOVA_dprime |   1B_GLMM_HitRate |   1C_LMM_RT |   2D_Detect_HitRate |   2E_Detect_RT |   3F_Agency_HitRate |   3G_Agency_RT |
|----:|----------------------:|-------------------------:|--------------------------:|------------------:|------------------:|------------:|--------------------:|---------------:|--------------------:|---------------:|
|  30 |                   0.2 |                      0.2 |                      0.11 |             0.349 |             0.402 |       0.996 |               0.425 |          0.998 |                   1 |              1 |
|  30 |                   0.5 |                      0.5 |                      0.28 |             0.966 |             0.99  |       1     |               0.993 |          1     |                   1 |              1 |
|  40 |                   0.2 |                      0.2 |                      0.11 |             0.45  |             0.519 |       1     |               0.564 |          1     |                   1 |              1 |
|  40 |                   0.5 |                      0.5 |                      0.28 |             0.99  |             0.996 |       1     |               0.998 |          1     |                   1 |              1 |
|  50 |                   0.2 |                      0.2 |                      0.11 |             0.471 |             0.585 |       1     |               0.646 |          1     |                   1 |              1 |
|  50 |                   0.5 |                      0.5 |                      0.28 |             1     |             0.999 |       1     |               0.999 |          1     |                   1 |              1 |
|  60 |                   0.2 |                      0.2 |                      0.11 |             0.593 |             0.662 |       1     |               0.708 |          1     |                   1 |              1 |
|  60 |                   0.5 |                      0.5 |                      0.28 |             1     |             1     |       1     |               1     |          1     |                   1 |              1 |