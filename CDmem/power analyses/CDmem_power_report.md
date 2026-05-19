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

---

## Preregistration Write-Up (Extensive Version)

### Sample Size Justification

Sample size was determined via Monte Carlo simulation-based power analysis. We simulated each of the seven planned analyses (1A–3G) using parameters derived from pilot data (*N* = 10). Baseline parameters included: *d*′ = 0.74 (*SD* = 0.75), hit rate = .62 (*SD* = .20), false alarm rate = .20, and log-transformed reaction time *M* = −0.192, with between-person *SD* = 0.149 and within-person *SD* = 0.241 (extracted from pilot LMM random effects). The random intercept *SD* for binomial GLMMs was set to 0.656 (extracted from pilot GLMM model 1B). Detection accuracy probabilities for the high- and low-control conditions were .892 and .577, respectively. Each participant contributed 60 trials per condition. Effect sizes for H1 and H2 analyses (1A–2E) and for the H3 RT model (3G) were specified as Cohen's *d* = 0.20 (small) and *d* = 0.50 (medium). For the H3 agency–memory binomial model (3F), effect sizes were specified as log-odds slopes of 0.20 and 0.50 (approximately *d* = 0.11 and *d* = 0.28, respectively; Borenstein et al., 2009), because the predictor (z-scored sense of agency) is a continuous within-trial variable rather than a condition-level contrast.

Power was estimated over 1,000 simulation iterations per cell using Generalized Estimating Equations (GEE) with an exchangeable correlation structure as a computationally efficient approximation to the planned `lme4`-based mixed models. GEE is expected to yield slightly conservative power estimates for binary outcomes at small *N*, making these results a defensible lower bound.

Results indicated that at a medium effect size (*d* = 0.50), all seven analyses achieved power ≥ .97 even at *N* = 30. At a small effect size (*d* = 0.20), the Gaussian LMM models (1C, 2E, 3G) and the agency binomial model (3F) achieved power ≥ .99 at all sample sizes, reflecting the high statistical leverage of trial-level continuous outcomes. The bottleneck analyses were the *d*′ ANOVA (1A) and the binomial GLMMs for hit rate (1B, 2D), which at *N* = 50 achieved power of .47, .59, and .65, respectively, for a small effect — below the conventional .80 threshold. For medium effects at *N* = 50, these same analyses reached power ≥ .99.

We therefore plan to collect data from *N* = 50 participants (with an initial recruitment target of approximately 55–60 to allow for exclusions). This sample provides ≥ .99 power for all analyses at a medium effect size and adequate (though below .80) sensitivity to small effects in the most demanding binomial and ANOVA-based analyses, which we consider acceptable given that effects of *d* = 0.20 are unlikely to be theoretically meaningful in this paradigm. Also considering logistic and financial dependencies imposed upon many EEG studies, we determine a lower cutoff on our sample size of 30 participants, in case reaching 50 is impossible.

---

## Preregistration Write-Up (AsPredicted — Concise Version)

Sample size was determined by Monte Carlo simulation (1,000 iterations per cell) using pilot-derived parameters (*N* = 10). We simulated each of the seven planned analyses (1A–3G) at small (*d* = 0.20) and medium (*d* = 0.50) effect sizes for sample sizes of 30, 40, 50, and 60. Baseline hit rate (.62), *d*′ (0.74), log-RT variance components (between-person *SD* = 0.149, within-person *SD* = 0.241), and random intercept *SD* (0.656) were extracted from pilot mixed models. Power was approximated using GEE, providing conservative estimates relative to the planned `lme4`-based models.

At a medium effect size, all seven analyses achieved power ≥ .97 at *N* = 30. At a small effect size, reaction time and agency models reached ceiling power (≥ .99) at all sample sizes, while the *d*′ ANOVA and binomial hit-rate models remained below .80 even at *N* = 60. We target *N* = 50 (recruiting ~55–60 for exclusion buffer), which provides ≥ .99 power for all analyses at medium effect sizes. Also considering logistic and financial dependencies imposed upon many EEG studies, we determine a lower cutoff on our sample size of 30 participants, in case reaching 50 is impossible.