# CDmem Comprehensive Analysis Report

## Data Loading & Preprocessing

- Excluded for timeout: []
- Excluded for accuracy outliers: []
- Excluded for calibration failure: []
- Excluded for manipulation failure (not preregistered): [np.int64(3), np.int64(11)]
  P3: EXCLUDED -- partial convergence, OR t-test failed (acc=+0.033 p=0.1564, agency=+0.733 p=0.0001)
  P4: PASS -- partial convergence, both t-tests sig (acc=+0.317 p=0.0002, agency=+2.417 p=0.0000)
  P5: PASS -- both staircases converged (acc=+0.333, agency=+1.383)
  P6: PASS -- both staircases converged (acc=+0.167, agency=+1.600)
  P7: PASS -- partial convergence, both t-tests sig (acc=+0.283 p=0.0000, agency=+3.983 p=0.0000)
  P11: EXCLUDED -- both diffs wrong direction (acc=+0.000, agency=-0.300)
- Total recognition trials excluded for RT outliers: 41
- Final N participants: 4

## Sanity Checks

**Sanity Check: Agency Ratings (High vs Low)**
We expect agency ratings to be significantly higher in the 'high' control condition.
- High Mean (SD): 5.129 (0.611)
- Low Mean (SD): 2.783 (0.627)
- Result: *t*(3) = 3.980, *p* = 0.0284, Cohen's *d* = 3.790

**Sanity Check: Detection Accuracy (High vs Low)**
We expect detection accuracy to be significantly higher in the 'high' control condition.
- High Mean (SD): 0.867 (0.131)
- Low Mean (SD): 0.592 (0.091)
- Result: *t*(3) = 7.318, *p* = 0.0053, Cohen's *d* = 2.446

## ANALYSES

### 1) Does higher motor control lead to better memory?

**1A) Descriptive d' Summary (above-chance check)**
**Participant-level d' values:**
  - P4: d' = 0.752
  - P5: d' = 1.884
  - P6: d' = 0.591
  - P7: d' = 0.476

- Mean d' = 0.926 (SD = 0.649)
- All above chance (d' > 0): True

- One-sample t-test against chance: t(3) = 2.853, p = 0.0649

**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                 ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                 ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)         ┆ -0.538192 ┆ 0.483901  ┆ -1.48662  ┆ 0.410237  ┆ -1.112194 ┆ inf ┆ 0.266055 │
│ control_level_c     ┆ -0.009408 ┆ 0.14886   ┆ -0.301168 ┆ 0.282351  ┆ -0.063202 ┆ inf ┆ 0.949605 │
│ item_type_c         ┆ 0.378286  ┆ 0.149223  ┆ 0.085814  ┆ 0.670757  ┆ 2.535038  ┆ inf ┆ 0.011244 │
│ control_level_c:ite ┆ 0.9472    ┆ 0.298882  ┆ 0.361401  ┆ 1.532998  ┆ 3.16914   ┆ inf ┆ 0.001529 │
│ m_type_c            ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.955854 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ -0.300555   │
│ 5     ┆ 1.483119    │
│ 6     ┆ -1.111681   │
│ 7     ┆ -0.059488   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A binomial generalized linear mixed model on target trials revealed that Control Level predicted memory hits, $\beta$ = -0.01, *SE* = 0.15, *z* = -0.06, *p* = 0.950.

**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ 0.094627  ┆ 0.16882   ┆ -0.445586 ┆ 0.634839  ┆ 0.560519  ┆ 2.97125    ┆ 0.614611 │
│ control_leve ┆ -0.048337 ┆ 0.041254  ┆ -0.129468 ┆ 0.032794  ┆ -1.171698 ┆ 356.980703 ┆ 0.2421   │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ item_type_c  ┆ -0.026122 ┆ 0.041241  ┆ -0.107228 ┆ 0.054985  ┆ -0.633383 ┆ 356.972041 ┆ 0.526889 │
│ control_leve ┆ -0.02503  ┆ 0.082579  ┆ -0.187433 ┆ 0.137373  ┆ -0.303097 ┆ 356.999931 ┆ 0.761992 │
│ l_c:item_typ ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ e_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.334315 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.388923 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ 0.131435    │
│ 5     ┆ -0.205247   │
│ 6     ┆ 0.402098    │
│ 7     ┆ -0.328286   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A linear mixed model on log-transformed reaction times for correct old items showed an effect of Control Level, $\beta$ = -0.05, *SE* = 0.04, *t* = -1.17, *p* = 0.242.

### 2) Does this memory effect depend on the conscious detection of control?

*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*

**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌─────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                 ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                 ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)         ┆ -0.489872 ┆ 0.52459   ┆ -1.51805  ┆ 0.538306  ┆ -0.933818 ┆ inf ┆ 0.350398 │
│ detection_accuracy_ ┆ 0.515036  ┆ 0.290274  ┆ -0.053891 ┆ 1.083964  ┆ 1.774309  ┆ inf ┆ 0.076012 │
│ c                   ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
│ control_level_c     ┆ 0.24906   ┆ 0.287043  ┆ -0.313534 ┆ 0.811654  ┆ 0.867675  ┆ inf ┆ 0.385572 │
│ detection_accuracy_ ┆ 0.344392  ┆ 0.57181   ┆ -0.776335 ┆ 1.465119  ┆ 0.602284  ┆ inf ┆ 0.546985 │
│ c:control_l…        ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 1.009734 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ -0.349006   │
│ 5     ┆ 1.587791    │
│ 6     ┆ -1.078737   │
│ 7     ┆ -0.147293   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy significantly predicted subsequent recognition of controlled items, $\beta$ = 0.52, *SE* = 0.29, *z* = 1.77, *p* = 0.076.

**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ 0.114227  ┆ 0.146766  ┆ -0.339436 ┆ 0.567891  ┆ 0.778297  ┆ 3.164115   ┆ 0.490436 │
│ detection_ac ┆ -0.073487 ┆ 0.081002  ┆ -0.233249 ┆ 0.086275  ┆ -0.907228 ┆ 192.992755 ┆ 0.365417 │
│ curacy_c     ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ control_leve ┆ 0.063275  ┆ 0.079652  ┆ -0.093829 ┆ 0.220378  ┆ 0.794386  ┆ 192.486801 ┆ 0.427949 │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ detection_ac ┆ -0.343829 ┆ 0.157903  ┆ -0.655275 ┆ -0.032384 ┆ -2.177467 ┆ 192.216268 ┆ 0.030664 │
│ curacy_c:con ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ trol_l…      ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.281563 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.364925 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ 0.092789    │
│ 5     ┆ -0.189612   │
│ 6     ┆ 0.346962    │
│ 7     ┆ -0.250139   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy was associated with reaction times for correctly remembered controlled items, $\beta$ = -0.07, *SE* = 0.08, *t* = -0.91, *p* = 0.365.

### 3) Can we predict this effect from participants' subjective agency ratings?

*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*

**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────────────┬───────────┬───────────┬───────────┬───────────┬──────────┬─────┬──────────┐
│ term                 ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat   ┆ df  ┆ p_value  │
│ ---                  ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---      ┆ --- ┆ ---      │
│ str                  ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64      ┆ f64 ┆ f64      │
╞══════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪══════════╪═════╪══════════╡
│ (Intercept)          ┆ -0.427443 ┆ 0.534906  ┆ -1.475839 ┆ 0.620953  ┆ -0.7991  ┆ inf ┆ 0.424232 │
│ agency_z             ┆ 0.035197  ┆ 0.137602  ┆ -0.234498 ┆ 0.304893  ┆ 0.25579  ┆ inf ┆ 0.798113 │
│ control_level_c      ┆ 0.435348  ┆ 0.271959  ┆ -0.097682 ┆ 0.968379  ┆ 1.600784 ┆ inf ┆ 0.109425 │
│ agency_z:control_lev ┆ 0.27737   ┆ 0.286101  ┆ -0.283378 ┆ 0.838119  ┆ 0.969483 ┆ inf ┆ 0.332304 │
│ el_c                 ┆           ┆           ┆           ┆           ┆          ┆     ┆          │
└──────────────────────┴───────────┴───────────┴───────────┴───────────┴──────────┴─────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 1.034465 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ -0.369267   │
│ 5     ┆ 1.623177    │
│ 6     ┆ -1.113813   │
│ 7     ┆ -0.127168   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Within-participant variation in subjective agency ratings significantly predicted recognition hits for controlled items, $\beta$ = 0.04, *SE* = 0.14, *z* = 0.26, *p* = 0.798.

**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ 0.086922  ┆ 0.145314  ┆ -0.368994 ┆ 0.542838  ┆ 0.598168  ┆ 3.077729   ┆ 0.590892 │
│ agency_z     ┆ -0.01002  ┆ 0.036682  ┆ -0.082372 ┆ 0.062331  ┆ -0.273169 ┆ 192.315431 ┆ 0.785016 │
│ control_leve ┆ -0.042893 ┆ 0.069099  ┆ -0.179183 ┆ 0.093398  ┆ -0.620738 ┆ 192.13009  ┆ 0.535508 │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ agency_z:con ┆ -0.053039 ┆ 0.075025  ┆ -0.201013 ┆ 0.094935  ┆ -0.70695  ┆ 192.942861 ┆ 0.48045  │
│ trol_level_c ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.280615 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.36856  ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (4, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 4     ┆ 0.103679    │
│ 5     ┆ -0.199036   │
│ 6     ┆ 0.339926    │
│ 7     ┆ -0.244568   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Subjective agency ratings predicted recognition reaction times for controlled items, $\beta$ = -0.01, *SE* = 0.04, *t* = -0.27, *p* = 0.785.

