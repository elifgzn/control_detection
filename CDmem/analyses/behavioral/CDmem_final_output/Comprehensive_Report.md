# CDmem Comprehensive Analysis Report

## Data Loading & Preprocessing

[WARNING] Could not load CDmem_1_1.csv: Error tokenizing data. C error: Expected 1 fields in line 129, saw 6

- Excluded for timeout: []
- Excluded for accuracy outliers: []
- Excluded for calibration failure: []
- Total recognition trials excluded for RT outliers: 34
- Final N participants: 3

## Sanity Checks

**Sanity Check: Agency Ratings (High vs Low)**
We expect agency ratings to be significantly higher in the 'high' control condition.
- High Mean (SD): 4.872 (0.187)
- Low Mean (SD): 4.011 (1.435)
- Result: *t*(2) = 0.997, *p* = 0.4238, Cohen's *d* = 0.842

**Sanity Check: Detection Accuracy (High vs Low)**
We expect detection accuracy to be significantly higher in the 'high' control condition.
- High Mean (SD): 0.867 (0.120)
- Low Mean (SD): 0.761 (0.228)
- Result: *t*(2) = 0.984, *p* = 0.4289, Cohen's *d* = 0.580

## ANALYSES

### 1) Does higher motor control lead to better memory?

**1A) Descriptive d' Summary (above-chance check)**
**Participant-level d' values:**
  - P2: d' = 0.882
  - P3: d' = 1.027
  - P4: d' = 0.752

- Mean d' = 0.887 (SD = 0.137)
- All above chance (d' > 0): True

- One-sample t-test against chance: t(2) = 11.185, p = 0.0079

**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────────────┬───────────┬───────────┬───────────┬───────────┬────────────┬─────┬────────────┐
│ term                        ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat     ┆ df  ┆ p_value    │
│ ---                         ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ --- ┆ ---        │
│ str                         ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64 ┆ f64        │
╞═════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪═════╪════════════╡
│ (Intercept)                 ┆ -0.871821 ┆ 0.084438  ┆ -1.037316 ┆ -0.706326 ┆ -10.325016 ┆ inf ┆ 5.4310e-25 │
│ control_level_c             ┆ 0.208312  ┆ 0.168876  ┆ -0.122678 ┆ 0.539302  ┆ 1.233526   ┆ inf ┆ 0.217379   │
│ item_type_c                 ┆ 0.627936  ┆ 0.168876  ┆ 0.296946  ┆ 0.958926  ┆ 3.718335   ┆ inf ┆ 0.000201   │
│ control_level_c:item_type_c ┆ 0.192928  ┆ 0.337751  ┆ -0.469052 ┆ 0.854908  ┆ 0.571214   ┆ inf ┆ 0.567854   │
└─────────────────────────────┴───────────┴───────────┴───────────┴───────────┴────────────┴─────┴────────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.0      ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ 0.0         │
│ 3     ┆ 0.0         │
│ 4     ┆ 0.0         │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A binomial generalized linear mixed model on target trials revealed that Control Level predicted memory hits, $\beta$ = 0.21, *SE* = 0.17, *z* = 1.23, *p* = 0.217.

**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term                        ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---                         ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str                         ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞═════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)                 ┆ 0.166814  ┆ 0.033304  ┆ 0.027597  ┆ 0.30603   ┆ 5.00882   ┆ 2.06246    ┆ 0.035323 │
│ control_level_c             ┆ -0.034804 ┆ 0.057815  ┆ -0.148792 ┆ 0.079184  ┆ -0.601991 ┆ 204.997054 ┆ 0.547845 │
│ item_type_c                 ┆ -0.068005 ┆ 0.057948  ┆ -0.182267 ┆ 0.046258  ┆ -1.173547 ┆ 201.356053 ┆ 0.241963 │
│ control_level_c:item_type_c ┆ 0.139039  ┆ 0.11565   ┆ -0.088978 ┆ 0.367056  ┆ 1.202234  ┆ 204.990255 ┆ 0.23066  │
└─────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.028764 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.405687 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ -0.00016    │
│ 3     ┆ -0.014421   │
│ 4     ┆ 0.014581    │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A linear mixed model on log-transformed reaction times for correct old items showed an effect of Control Level, $\beta$ = -0.03, *SE* = 0.06, *t* = -0.60, *p* = 0.548.

### 2) Does this memory effect depend on the conscious detection of control?

*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*

**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                            ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                             ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                             ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)                     ┆ -0.554613 ┆ 0.14966   ┆ -0.847942 ┆ -0.261284 ┆ -3.705812 ┆ inf ┆ 0.000211 │
│ detection_accuracy_c            ┆ 0.053924  ┆ 0.299321  ┆ -0.532734 ┆ 0.640582  ┆ 0.180155  ┆ inf ┆ 0.857031 │
│ control_level_c                 ┆ 0.63707   ┆ 0.299321  ┆ 0.050412  ┆ 1.223728  ┆ 2.128386  ┆ inf ┆ 0.033305 │
│ detection_accuracy_c:control_l… ┆ -1.052161 ┆ 0.598642  ┆ -2.225477 ┆ 0.121155  ┆ -1.75758  ┆ inf ┆ 0.078819 │
└─────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬───────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate  ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---       ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64       ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪═══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 1.0326e-7 ┆ null     ┆ null      │
└─────────────┴─────────────────┴───────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ -4.2414e-14 │
│ 3     ┆ 7.2803e-14  │
│ 4     ┆ -3.0389e-14 │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy significantly predicted subsequent recognition of controlled items, $\beta$ = 0.05, *SE* = 0.30, *z* = 0.18, *p* = 0.857.

**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────┬──────────┐
│ term                            ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df    ┆ p_value  │
│ ---                             ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---   ┆ ---      │
│ str                             ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64   ┆ f64      │
╞═════════════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════╪══════════╡
│ (Intercept)                     ┆ 0.154931  ┆ 0.043572  ┆ 0.068689  ┆ 0.241173  ┆ 3.555717  ┆ 124.0 ┆ 0.000535 │
│ detection_accuracy_c            ┆ -0.071962 ┆ 0.087145  ┆ -0.244445 ┆ 0.100522  ┆ -0.825773 ┆ 124.0 ┆ 0.410519 │
│ control_level_c                 ┆ 0.044182  ┆ 0.087145  ┆ -0.128302 ┆ 0.216665  ┆ 0.506991  ┆ 124.0 ┆ 0.613061 │
│ detection_accuracy_c:control_l… ┆ -0.019541 ┆ 0.174289  ┆ -0.364509 ┆ 0.325426  ┆ -0.112121 ┆ 124.0 ┆ 0.910909 │
└─────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.0      ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.370402 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ 0.0         │
│ 3     ┆ 0.0         │
│ 4     ┆ 0.0         │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy was associated with reaction times for correctly remembered controlled items, $\beta$ = -0.07, *SE* = 0.09, *t* = -0.83, *p* = 0.411.

### 3) Can we predict this effect from participants' subjective agency ratings?

*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*

**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬───────────┐
│ term                     ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value   │
│ ---                      ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---       │
│ str                      ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64       │
╞══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪═══════════╡
│ (Intercept)              ┆ -0.581639 ┆ 0.116926  ┆ -0.81081  ┆ -0.352467 ┆ -4.974403 ┆ inf ┆ 6.5449e-7 │
│ agency_z                 ┆ -0.155754 ┆ 0.116677  ┆ -0.384438 ┆ 0.072929  ┆ -1.334914 ┆ inf ┆ 0.181904  │
│ control_level_c          ┆ 0.39084   ┆ 0.23304   ┆ -0.065911 ┆ 0.84759   ┆ 1.677134  ┆ inf ┆ 0.093516  │
│ agency_z:control_level_c ┆ 0.147425  ┆ 0.233435  ┆ -0.3101   ┆ 0.604949  ┆ 0.631544  ┆ inf ┆ 0.527685  │
└──────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴───────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.015661 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ -0.00055    │
│ 3     ┆ 0.001751    │
│ 4     ┆ -0.001201   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Within-participant variation in subjective agency ratings significantly predicted recognition hits for controlled items, $\beta$ = -0.16, *SE* = 0.12, *z* = -1.33, *p* = 0.182.

**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────┬──────────┐
│ term                     ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df    ┆ p_value  │
│ ---                      ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---   ┆ ---      │
│ str                      ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64   ┆ f64      │
╞══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════╪══════════╡
│ (Intercept)              ┆ 0.122866  ┆ 0.034434  ┆ 0.054711  ┆ 0.191021  ┆ 3.568126  ┆ 124.0 ┆ 0.000512 │
│ agency_z                 ┆ -0.039579 ┆ 0.032271  ┆ -0.103452 ┆ 0.024294  ┆ -1.226472 ┆ 124.0 ┆ 0.222345 │
│ control_level_c          ┆ 0.063576  ┆ 0.068869  ┆ -0.072734 ┆ 0.199887  ┆ 0.923155  ┆ 124.0 ┆ 0.357719 │
│ agency_z:control_level_c ┆ 0.032129  ┆ 0.064542  ┆ -0.095617 ┆ 0.159875  ┆ 0.497803  ┆ 124.0 ┆ 0.619505 │
└──────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.0      ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.369194 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (3, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 2     ┆ 0.0         │
│ 3     ┆ 0.0         │
│ 4     ┆ 0.0         │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Subjective agency ratings predicted recognition reaction times for controlled items, $\beta$ = -0.04, *SE* = 0.03, *t* = -1.23, *p* = 0.222.

