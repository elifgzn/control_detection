# CDmem Power Analysis Overview

**File:** `CDmem_power.py`  
Monte Carlo simulation-based power analysis for the Control Detection + Memory (CDmem) experiment.

---

## Purpose

Estimates statistical power for detecting behavioral effects on recognition memory and reaction times across seven primary analyses. Analyses are crossed with two effect sizes (small and medium) and four sample sizes (30, 40, 50, 60).

All analysis functions in this script are designed to **mirror the exact seven analyses in `CDmem_analyses_final.py`** — using the same design, same predictors, and same contrast coding. The only difference is the estimator: **GEE** is used instead of GLMM for simulation stability (see below).

---

## Simulation Settings

| Parameter | Value | Notes |
|---|---|---|
| `N_SIMULATIONS` | 1000 | Monte Carlo iterations per cell. |
| `SAMPLE_SIZES` | [30, 40, 50, 60] | Target range for an EEG study with practical constraints |
| `ALPHA` | 0.05 | Two-tailed significance threshold |
| `N_TRIALS_PER_CONDITION` | 60 | Target (old) trials per condition (High / Low control) |
| `np.random.seed` | 42 | Fixed for reproducibility |

---

## Parameters from the Literature

| Parameter | Value | Source |
|---|---|---|
| `BASELINE_DPRIME` | 0.74 | Average of congruent (0.78) and incongruent (0.70) conditions — Schreiner et al. (2024) |
| `DPRIME_SD` | 0.75 | Pooled SD across conditions — Schreiner et al. (2024) |
| `BASELINE_HIT_RATE` | 0.62 | Schreiner et al. (2024), congruent condition; confirmed by Wu et al. (2025), Exp. 4 |
| `HIT_RATE_SD` | 0.20 | Schreiner et al. (2024) |
| `BASELINE_LOG_RT` | -0.20 | Assumed baseline log-reaction time (~820ms) |
| `LOG_RT_SD` | 0.20 | Assumed within-person SD of log-reaction times |
| `DETECT_HIGH` | 0.892 | Pilot mean for detection accuracy in high control |
| `DETECT_LOW` | 0.577 | Pilot mean for detection accuracy in low control |
| `RI_SD` | 1.36 | Random intercept SD estimated by converting d′ SD to log-odds: `DPRIME_SD × (π / √3) ≈ 1.36` |
| `WITHIN_PERSON_CORR` | 0.50 | Within-participant correlation across conditions |
| `AGENCY_SLOPE_SMALL` | 0.20 | Log-odds per +1 SD agency (small effect, ~+5% hit rate at baseline) |
| `AGENCY_SLOPE_MEDIUM` | 0.50 | Log-odds per +1 SD agency (medium effect, ~+12% hit rate at baseline) |

---

## Why GEE (Not GLMM) for Trial-Level Simulations?

GEE is used for trial-level power simulations instead of GLMM because:

1. **Computational stability** — GEE almost never fails to converge across 1000+ iterations; GLMM convergence failures require complex handling and reduce effective simulation count.
2. **Conservative estimates** — GEE coefficients are attenuated relative to GLMM subject-specific coefficients due to the **non-collapsibility of the logit link** (Skrondal & Rabe-Hesketh, 2004; Diggle et al., 2002). Therefore: **actual GLMM power ≥ GEE-estimated power**. Using GEE is conservative and safe for planning.
3. **Established precedent** — Rochon (1998) and Liu & Liang (1997) explicitly recommend GEE-based simulation for power analysis with correlated binary outcomes.

> **Implication for methods section:** Note that GEE was used for power estimation (conservative approximation), while the confirmatory analyses use GLMM (`pymer4`/`lme4`).

---

## The Seven Main Analyses

### 1) Does higher motor control lead to better memory?

**1A) 2x2 RM ANOVA on d'**
* **Type:** Repeated-Measures ANOVA (`AnovaRM`)
* **Model:** `dprime ~ control_level * item_type`
* **DGP:** D-prime generated with between- and within-subject variance decomposition (rho=0.50). Only controlled items exhibit the effect size difference.

**1B) Binomial GLMM on All Old Items (Hit Rate)**
* **Type:** GEE, Binomial / logit / Exchangeable
* **Model:** `said_old_int ~ control_level_c * item_type_c`
* **DGP:** Binary hit/miss sampled from log-odds. Uncontrolled items sit at baseline; controlled items differ by the effect size. 

**1C) Gaussian LMM on All Old Items (RT)**
* **Type:** GEE, Gaussian / identity / Exchangeable
* **Model:** `log_mem_rt ~ control_level_c * item_type_c`
* **DGP:** Continuous log-RTs generated with random intercepts. Faster (lower) RTs mapped to the high control condition.

### 2) Does this memory effect depend on conscious detection?

**2D) Binomial GLMM on Old Controlled Items (Detection → Hit Rate)**
* **Type:** GEE, Binomial / logit / Exchangeable
* **Model:** `said_old_int ~ detection_accuracy_c * control_level_c`
* **DGP:** Detection is binary sampled from pilot probabilities (89% high, 58% low). Log-odds of memory hit increases with correct detection.

**2E) Gaussian LMM on Old Controlled Items (Detection → RT)**
* **Type:** GEE, Gaussian / identity / Exchangeable
* **Model:** `log_mem_rt ~ detection_accuracy_c * control_level_c`
* **DGP:** Correct detection predicts faster (more negative) log-reaction times.

### 3) Can we predict this effect from subjective agency ratings?

**3F) Binomial GLMM on Old Controlled Items (Agency → Hit Rate)**
* **Type:** GEE, Binomial / logit / Exchangeable
* **Model:** `said_old_int ~ agency_z * control_level_c`
* **DGP:** Agency ratings drawn from N(0,1). Memory hit probability scales via logistic link with `agency_z`.

**3G) Gaussian LMM on Old Controlled Items (Agency → RT)**
* **Type:** GEE, Gaussian / identity / Exchangeable
* **Model:** `log_mem_rt ~ agency_z * control_level_c`
* **DGP:** Higher agency predicts faster log-reaction times.

---

## Power Estimates (Probability of finding p < .05)

|   N |   Effect_Size |   1A_ANOVA_dprime |   1B_GLMM_HitRate |   1C_LMM_RT |   2D_Detect_HitRate |   2E_Detect_RT |   3F_Agency_HitRate |   3G_Agency_RT |
|----:|--------------:|------------------:|------------------:|------------:|--------------------:|---------------:|--------------------:|---------------:|
|  30 |           0.2 |             0.127 |             0.365 |           1 |               0.366 |              1 |               0.999 |              1 |
|  30 |           0.5 |             0.477 |             0.971 |           1 |               0.984 |              1 |               1     |              1 |
|  40 |           0.2 |             0.151 |             0.451 |           1 |               0.475 |              1 |               1     |              1 |
|  40 |           0.5 |             0.614 |             0.987 |           1 |               0.997 |              1 |               1     |              1 |
|  50 |           0.2 |             0.159 |             0.508 |           1 |               0.532 |              1 |               1     |              1 |
|  50 |           0.5 |             0.687 |             0.998 |           1 |               0.999 |              1 |               1     |              1 |
|  60 |           0.2 |             0.191 |             0.584 |           1 |               0.601 |              1 |               1     |              1 |
|  60 |           0.5 |             0.758 |             1     |           1 |               1     |              1 |               1     |              1 |

*Note: Power = 1 for continuous reaction time models (1C, 2E, 3G) and Agency models (3F) reflects very high sensitivity due to trial-level data and small assumed residual SDs. Actual empirical power may be lower if true RT variability is substantially higher.*