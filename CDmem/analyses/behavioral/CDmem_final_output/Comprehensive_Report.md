# CDmem Comprehensive Analysis Report

## Data Loading & Preprocessing

- Excluded for timeout: []
- Excluded for accuracy outliers: []
- Excluded for calibration failure: []
- Total recognition trials excluded for RT outliers: 65
- Final N participants: 8

## Sanity Checks

**Sanity Check: Agency Ratings (High vs Low)**
We expect agency ratings to be significantly higher in the 'high' control condition.
- High Mean (SD): 4.856 (1.125)
- Low Mean (SD): 3.092 (0.834)
- Result: *t*(7) = 4.099, *p* = 0.0046, Cohen's *d* = 1.782

**Sanity Check: Detection Accuracy (High vs Low)**
We expect detection accuracy to be significantly higher in the 'high' control condition.
- High Mean (SD): 0.892 (0.115)
- Low Mean (SD): 0.577 (0.067)
- Result: *t*(7) = 6.274, *p* = 0.0004, Cohen's *d* = 3.352

## ANALYSES

### 1) Does higher motor control lead to better memory?

**1A) 2 (Control Level) x 2 (Item Type) ANOVA on d'**
| Source                    |        SS |   ddof1 |   ddof2 |        MS |         F |       p-unc |   p-GG-corr |        ng2 |   eps |
|:--------------------------|----------:|--------:|--------:|----------:|----------:|------------:|------------:|-----------:|------:|
| control_level             | 0.163654  |       1 |       7 | 0.163654  |  6.32233  | 0.0401364   | 0.0401364   | 0.0183227  |     1 |
| item_type                 | 0.392774  |       1 |       7 | 0.392774  | 39.4737   | 0.000410916 | 0.000410916 | 0.0428751  |     1 |
| control_level * item_type | 0.0292205 |       1 |       7 | 0.0292205 |  0.574856 | 0.473084    | 0.473084    | 0.00332152 |     1 |

> **APA 7 Reporting Example:**
> A 2 (Control Level: High vs. Low) x 2 (Item Type: Controlled vs. Uncontrolled) repeated-measures ANOVA on *d'* revealed a main effect of Control Level, *F*(1, 7) = 6.32, *p* = 0.040, $\eta_p^2$ = 0.018. The main effect of Item Type was *F*(1, 7) = 39.47, *p* = 0.000, $\eta_p^2$ = 0.043. The interaction effect was *F*(1, 7) = 0.57, *p* = 0.473, $\eta_p^2$ = 0.003.

**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**
shape: (4, 8)
┌─────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                        ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                         ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                         ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)                 ┆ -0.387506 ┆ 0.237063  ┆ -0.852142 ┆ 0.077129  ┆ -1.634612 ┆ inf ┆ 0.10213  │
│ control_level_c             ┆ 0.242877  ┆ 0.098865  ┆ 0.049105  ┆ 0.436648  ┆ 2.456652  ┆ inf ┆ 0.014024 │
│ item_type_c                 ┆ 0.362796  ┆ 0.09897   ┆ 0.168818  ┆ 0.556773  ┆ 3.665712  ┆ inf ┆ 0.000247 │
│ control_level_c:item_type_c ┆ 0.180154  ┆ 0.197636  ┆ -0.207206 ┆ 0.567513  ┆ 0.911543  ┆ inf ┆ 0.362009 │
└─────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

> **APA 7 Reporting Example:**
> A binomial generalized linear mixed model on target trials revealed that Control Level predicted memory hits, $\beta$ = 0.24, *SE* = 0.10, *z* = 2.46, *p* = 0.014.

**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**
shape: (4, 8)
┌─────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term                        ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---                         ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str                         ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞═════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)                 ┆ -0.192278 ┆ 0.053548  ┆ -0.319062 ┆ -0.065495 ┆ -3.590767 ┆ 6.955894   ┆ 0.008943 │
│ control_level_c             ┆ -0.009196 ┆ 0.017481  ┆ -0.043513 ┆ 0.025121  ┆ -0.526036 ┆ 765.128196 ┆ 0.599016 │
│ item_type_c                 ┆ -0.024021 ┆ 0.017454  ┆ -0.058284 ┆ 0.010242  ┆ -1.376282 ┆ 764.979317 ┆ 0.169137 │
│ control_level_c:item_type_c ┆ 0.047866  ┆ 0.035025  ┆ -0.020891 ┆ 0.116622  ┆ 1.366617  ┆ 765.201939 ┆ 0.172147 │
└─────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

> **APA 7 Reporting Example:**
> A linear mixed model on log-transformed reaction times for correct old items showed an effect of Control Level, $\beta$ = -0.01, *SE* = 0.02, *t* = -0.53, *p* = 0.599.

### 2) Does this memory effect depend on the conscious detection of control?

*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*

**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
shape: (4, 8)
┌─────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                            ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                             ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                             ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)                     ┆ -0.334566 ┆ 0.274171  ┆ -0.871932 ┆ 0.202799  ┆ -1.220284 ┆ inf ┆ 0.222357 │
│ detection_accuracy_c            ┆ 0.612657  ┆ 0.194498  ┆ 0.231448  ┆ 0.993866  ┆ 3.149938  ┆ inf ┆ 0.001633 │
│ control_level_c                 ┆ 0.222802  ┆ 0.19221   ┆ -0.153923 ┆ 0.599528  ┆ 1.159157  ┆ inf ┆ 0.246392 │
│ detection_accuracy_c:control_l… ┆ -0.277484 ┆ 0.388075  ┆ -1.038097 ┆ 0.48313   ┆ -0.715025 ┆ inf ┆ 0.474593 │
└─────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

> **APA 7 Reporting Example:**
> Detection accuracy significantly predicted subsequent recognition of controlled items, $\beta$ = 0.61, *SE* = 0.19, *z* = 3.15, *p* = 0.002.

**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
shape: (4, 8)
┌─────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term                            ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---                             ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str                             ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞═════════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)                     ┆ -0.195599 ┆ 0.051051  ┆ -0.31402  ┆ -0.077179 ┆ -3.831482 ┆ 7.737922   ┆ 0.005331 │
│ detection_accuracy_c            ┆ -0.036606 ┆ 0.032396  ┆ -0.100285 ┆ 0.027073  ┆ -1.129952 ┆ 418.406636 ┆ 0.259144 │
│ control_level_c                 ┆ 0.027804  ┆ 0.031509  ┆ -0.034132 ┆ 0.089741  ┆ 0.882435  ┆ 416.38326  ┆ 0.378051 │
│ detection_accuracy_c:control_l… ┆ -0.013912 ┆ 0.064173  ┆ -0.140055 ┆ 0.11223   ┆ -0.216792 ┆ 417.919416 ┆ 0.828476 │
└─────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

> **APA 7 Reporting Example:**
> Detection accuracy was associated with reaction times for correctly remembered controlled items, $\beta$ = -0.04, *SE* = 0.03, *t* = -1.13, *p* = 0.259.

### 3) Can we predict this effect from participants' subjective agency ratings?

*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*

**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
shape: (4, 8)
┌──────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                     ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                      ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                      ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)              ┆ -0.245043 ┆ 0.262585  ┆ -0.759699 ┆ 0.269614  ┆ -0.933194 ┆ inf ┆ 0.35072  │
│ agency_z                 ┆ 0.086396  ┆ 0.082475  ┆ -0.075251 ┆ 0.248043  ┆ 1.047546  ┆ inf ┆ 0.294848 │
│ control_level_c          ┆ 0.245103  ┆ 0.164523  ┆ -0.077357 ┆ 0.567564  ┆ 1.489778  ┆ inf ┆ 0.136283 │
│ agency_z:control_level_c ┆ 0.139145  ┆ 0.174579  ┆ -0.203024 ┆ 0.481314  ┆ 0.79703   ┆ inf ┆ 0.425433 │
└──────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

> **APA 7 Reporting Example:**
> Within-participant variation in subjective agency ratings significantly predicted recognition hits for controlled items, $\beta$ = 0.09, *SE* = 0.08, *z* = 1.05, *p* = 0.295.

**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
shape: (4, 8)
┌──────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term                     ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---                      ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str                      ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)              ┆ -0.200462 ┆ 0.050262  ┆ -0.318057 ┆ -0.082866 ┆ -3.988355 ┆ 7.388429   ┆ 0.004719 │
│ agency_z                 ┆ -0.005604 ┆ 0.01354   ┆ -0.032219 ┆ 0.021011  ┆ -0.413881 ┆ 415.260002 ┆ 0.679175 │
│ control_level_c          ┆ 0.021678  ┆ 0.026775  ┆ -0.030952 ┆ 0.074309  ┆ 0.809661  ┆ 415.146905 ┆ 0.418599 │
│ agency_z:control_level_c ┆ -0.023255 ┆ 0.029867  ┆ -0.081963 ┆ 0.035453  ┆ -0.778607 ┆ 420.557527 ┆ 0.436649 │
└──────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

> **APA 7 Reporting Example:**
> Subjective agency ratings predicted recognition reaction times for controlled items, $\beta$ = -0.01, *SE* = 0.01, *t* = -0.41, *p* = 0.679.

