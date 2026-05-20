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
**Fixed Effects:**
shape: (4, 8)
┌──────────┬──────────┬──────────┬──────────┬──────────┬─────────┬─────┬─────────┐
│ term     ┆ estimate ┆ std_erro ┆ conf_low ┆ conf_hig ┆ z_stat  ┆ df  ┆ p_value │
│ ---      ┆ ---      ┆ r        ┆ ---      ┆ h        ┆ ---     ┆ --- ┆ ---     │
│ str      ┆ f64      ┆ ---      ┆ f64      ┆ ---      ┆ f64     ┆ f64 ┆ f64     │
│          ┆          ┆ f64      ┆          ┆ f64      ┆         ┆     ┆         │
╞══════════╪══════════╪══════════╪══════════╪══════════╪═════════╪═════╪═════════╡
│ (Interce ┆ -0.38750 ┆ 0.237063 ┆ -0.85214 ┆ 0.077129 ┆ -1.6346 ┆ inf ┆ 0.10213 │
│ pt)      ┆ 6        ┆          ┆ 2        ┆          ┆ 12      ┆     ┆         │
│ control_ ┆ 0.242877 ┆ 0.098865 ┆ 0.049105 ┆ 0.436648 ┆ 2.45665 ┆ inf ┆ 0.01402 │
│ level_c  ┆          ┆          ┆          ┆          ┆ 2       ┆     ┆ 4       │
│ item_typ ┆ 0.362796 ┆ 0.09897  ┆ 0.168818 ┆ 0.556773 ┆ 3.66571 ┆ inf ┆ 0.00024 │
│ e_c      ┆          ┆          ┆          ┆          ┆ 2       ┆     ┆ 7       │
│ control_ ┆ 0.180154 ┆ 0.197636 ┆ -0.20720 ┆ 0.567513 ┆ 0.91154 ┆ inf ┆ 0.36200 │
│ level_c: ┆          ┆          ┆ 6        ┆          ┆ 3       ┆     ┆ 9       │
│ item_typ ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
│ e_c      ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┴─────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.655602 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ 0.649639    │
│ 11    ┆ -0.820997   │
│ 12    ┆ -0.401079   │
│ 13    ┆ 0.244887    │
│ 14    ┆ 1.240669    │
│ 15    ┆ -0.482046   │
│ 16    ┆ -0.413951   │
│ 17    ┆ -0.003728   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A binomial generalized linear mixed model on target trials revealed that Control Level predicted memory hits, $\beta$ = 0.24, *SE* = 0.10, *z* = 2.46, *p* = 0.014.

**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ term     ┆ estimat ┆ std_err ┆ conf_lo ┆ conf_hi ┆ t_stat  ┆ df      ┆ p_value │
│ ---      ┆ e       ┆ or      ┆ w       ┆ gh      ┆ ---     ┆ ---     ┆ ---     │
│ str      ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ f64     ┆ f64     ┆ f64     │
│          ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆         ┆         ┆         │
╞══════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╡
│ (Interce ┆ -0.1922 ┆ 0.05354 ┆ -0.3190 ┆ -0.0654 ┆ -3.5907 ┆ 6.95589 ┆ 0.00894 │
│ pt)      ┆ 78      ┆ 8       ┆ 62      ┆ 95      ┆ 67      ┆ 4       ┆ 3       │
│ control_ ┆ -0.0091 ┆ 0.01748 ┆ -0.0435 ┆ 0.02512 ┆ -0.5260 ┆ 765.128 ┆ 0.59901 │
│ level_c  ┆ 96      ┆ 1       ┆ 13      ┆ 1       ┆ 36      ┆ 196     ┆ 6       │
│ item_typ ┆ -0.0240 ┆ 0.01745 ┆ -0.0582 ┆ 0.01024 ┆ -1.3762 ┆ 764.979 ┆ 0.16913 │
│ e_c      ┆ 21      ┆ 4       ┆ 84      ┆ 2       ┆ 82      ┆ 317     ┆ 7       │
│ control_ ┆ 0.04786 ┆ 0.03502 ┆ -0.0208 ┆ 0.11662 ┆ 1.36661 ┆ 765.201 ┆ 0.17214 │
│ level_c: ┆ 6       ┆ 5       ┆ 91      ┆ 2       ┆ 7       ┆ 939     ┆ 7       │
│ item_typ ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
│ e_c      ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
└──────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.149166 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.241337 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ -0.164578   │
│ 11    ┆ 0.199964    │
│ 12    ┆ 0.11271     │
│ 13    ┆ -0.032962   │
│ 14    ┆ -0.139361   │
│ 15    ┆ 0.19144     │
│ 16    ┆ -0.0939     │
│ 17    ┆ -0.073313   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> A linear mixed model on log-transformed reaction times for correct old items showed an effect of Control Level, $\beta$ = -0.01, *SE* = 0.02, *t* = -0.53, *p* = 0.599.

### 2) Does this memory effect depend on the conscious detection of control?

*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*

**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────┬──────────┬──────────┬──────────┬──────────┬─────────┬─────┬─────────┐
│ term     ┆ estimate ┆ std_erro ┆ conf_low ┆ conf_hig ┆ z_stat  ┆ df  ┆ p_value │
│ ---      ┆ ---      ┆ r        ┆ ---      ┆ h        ┆ ---     ┆ --- ┆ ---     │
│ str      ┆ f64      ┆ ---      ┆ f64      ┆ ---      ┆ f64     ┆ f64 ┆ f64     │
│          ┆          ┆ f64      ┆          ┆ f64      ┆         ┆     ┆         │
╞══════════╪══════════╪══════════╪══════════╪══════════╪═════════╪═════╪═════════╡
│ (Interce ┆ -0.33456 ┆ 0.274171 ┆ -0.87193 ┆ 0.202799 ┆ -1.2202 ┆ inf ┆ 0.22235 │
│ pt)      ┆ 6        ┆          ┆ 2        ┆          ┆ 84      ┆     ┆ 7       │
│ detectio ┆ 0.612657 ┆ 0.194498 ┆ 0.231448 ┆ 0.993866 ┆ 3.14993 ┆ inf ┆ 0.00163 │
│ n_accura ┆          ┆          ┆          ┆          ┆ 8       ┆     ┆ 3       │
│ cy_c     ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
│ control_ ┆ 0.222802 ┆ 0.19221  ┆ -0.15392 ┆ 0.599528 ┆ 1.15915 ┆ inf ┆ 0.24639 │
│ level_c  ┆          ┆          ┆ 3        ┆          ┆ 7       ┆     ┆ 2       │
│ detectio ┆ -0.27748 ┆ 0.388075 ┆ -1.03809 ┆ 0.48313  ┆ -0.7150 ┆ inf ┆ 0.47459 │
│ n_accura ┆ 4        ┆          ┆ 7        ┆          ┆ 25      ┆     ┆ 3       │
│ cy_c:con ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
│ trol_l…  ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┴─────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.725993 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ 0.69843     │
│ 11    ┆ -0.91595    │
│ 12    ┆ -0.418803   │
│ 13    ┆ 0.331571    │
│ 14    ┆ 1.338125    │
│ 15    ┆ -0.468851   │
│ 16    ┆ -0.487345   │
│ 17    ┆ -0.065073   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy significantly predicted subsequent recognition of controlled items, $\beta$ = 0.61, *SE* = 0.19, *z* = 3.15, *p* = 0.002.

**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ term     ┆ estimat ┆ std_err ┆ conf_lo ┆ conf_hi ┆ t_stat  ┆ df      ┆ p_value │
│ ---      ┆ e       ┆ or      ┆ w       ┆ gh      ┆ ---     ┆ ---     ┆ ---     │
│ str      ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ f64     ┆ f64     ┆ f64     │
│          ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆         ┆         ┆         │
╞══════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╡
│ (Interce ┆ -0.1955 ┆ 0.05105 ┆ -0.3140 ┆ -0.0771 ┆ -3.8314 ┆ 7.73792 ┆ 0.00533 │
│ pt)      ┆ 99      ┆ 1       ┆ 2       ┆ 79      ┆ 82      ┆ 2       ┆ 1       │
│ detectio ┆ -0.0366 ┆ 0.03239 ┆ -0.1002 ┆ 0.02707 ┆ -1.1299 ┆ 418.406 ┆ 0.25914 │
│ n_accura ┆ 06      ┆ 6       ┆ 85      ┆ 3       ┆ 52      ┆ 636     ┆ 4       │
│ cy_c     ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
│ control_ ┆ 0.02780 ┆ 0.03150 ┆ -0.0341 ┆ 0.08974 ┆ 0.88243 ┆ 416.383 ┆ 0.37805 │
│ level_c  ┆ 4       ┆ 9       ┆ 32      ┆ 1       ┆ 5       ┆ 26      ┆ 1       │
│ detectio ┆ -0.0139 ┆ 0.06417 ┆ -0.1400 ┆ 0.11223 ┆ -0.2167 ┆ 417.919 ┆ 0.82847 │
│ n_accura ┆ 12      ┆ 3       ┆ 55      ┆         ┆ 92      ┆ 416     ┆ 6       │
│ cy_c:con ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
│ trol_l…  ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
└──────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.136871 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.223259 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ -0.155437   │
│ 11    ┆ 0.114526    │
│ 12    ┆ 0.121571    │
│ 13    ┆ -0.04423    │
│ 14    ┆ -0.141466   │
│ 15    ┆ 0.210682    │
│ 16    ┆ -0.051294   │
│ 17    ┆ -0.054352   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Detection accuracy was associated with reaction times for correctly remembered controlled items, $\beta$ = -0.04, *SE* = 0.03, *t* = -1.13, *p* = 0.259.

### 3) Can we predict this effect from participants' subjective agency ratings?

*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*

**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────┬──────────┬──────────┬──────────┬──────────┬─────────┬─────┬─────────┐
│ term     ┆ estimate ┆ std_erro ┆ conf_low ┆ conf_hig ┆ z_stat  ┆ df  ┆ p_value │
│ ---      ┆ ---      ┆ r        ┆ ---      ┆ h        ┆ ---     ┆ --- ┆ ---     │
│ str      ┆ f64      ┆ ---      ┆ f64      ┆ ---      ┆ f64     ┆ f64 ┆ f64     │
│          ┆          ┆ f64      ┆          ┆ f64      ┆         ┆     ┆         │
╞══════════╪══════════╪══════════╪══════════╪══════════╪═════════╪═════╪═════════╡
│ (Interce ┆ -0.24504 ┆ 0.262585 ┆ -0.75969 ┆ 0.269614 ┆ -0.9331 ┆ inf ┆ 0.35072 │
│ pt)      ┆ 3        ┆          ┆ 9        ┆          ┆ 94      ┆     ┆         │
│ agency_z ┆ 0.086396 ┆ 0.082475 ┆ -0.07525 ┆ 0.248043 ┆ 1.04754 ┆ inf ┆ 0.29484 │
│          ┆          ┆          ┆ 1        ┆          ┆ 6       ┆     ┆ 8       │
│ control_ ┆ 0.245103 ┆ 0.164523 ┆ -0.07735 ┆ 0.567564 ┆ 1.48977 ┆ inf ┆ 0.13628 │
│ level_c  ┆          ┆          ┆ 7        ┆          ┆ 8       ┆     ┆ 3       │
│ agency_z ┆ 0.139145 ┆ 0.174579 ┆ -0.20302 ┆ 0.481314 ┆ 0.79703 ┆ inf ┆ 0.42543 │
│ :control ┆          ┆          ┆ 4        ┆          ┆         ┆     ┆ 3       │
│ _level_c ┆          ┆          ┆          ┆          ┆         ┆     ┆         │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┴─────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (1, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.703359 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ 0.660732    │
│ 11    ┆ -0.890179   │
│ 12    ┆ -0.419894   │
│ 13    ┆ 0.313024    │
│ 14    ┆ 1.293424    │
│ 15    ┆ -0.461936   │
│ 16    ┆ -0.472001   │
│ 17    ┆ -0.010538   │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Within-participant variation in subjective agency ratings significantly predicted recognition hits for controlled items, $\beta$ = 0.09, *SE* = 0.08, *z* = 1.05, *p* = 0.295.

**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
**Fixed Effects:**
shape: (4, 8)
┌──────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ term     ┆ estimat ┆ std_err ┆ conf_lo ┆ conf_hi ┆ t_stat  ┆ df      ┆ p_value │
│ ---      ┆ e       ┆ or      ┆ w       ┆ gh      ┆ ---     ┆ ---     ┆ ---     │
│ str      ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ f64     ┆ f64     ┆ f64     │
│          ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆         ┆         ┆         │
╞══════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╡
│ (Interce ┆ -0.2004 ┆ 0.05026 ┆ -0.3180 ┆ -0.0828 ┆ -3.9883 ┆ 7.38842 ┆ 0.00471 │
│ pt)      ┆ 62      ┆ 2       ┆ 57      ┆ 66      ┆ 55      ┆ 9       ┆ 9       │
│ agency_z ┆ -0.0056 ┆ 0.01354 ┆ -0.0322 ┆ 0.02101 ┆ -0.4138 ┆ 415.260 ┆ 0.67917 │
│          ┆ 04      ┆         ┆ 19      ┆ 1       ┆ 81      ┆ 002     ┆ 5       │
│ control_ ┆ 0.02167 ┆ 0.02677 ┆ -0.0309 ┆ 0.07430 ┆ 0.80966 ┆ 415.146 ┆ 0.41859 │
│ level_c  ┆ 8       ┆ 5       ┆ 52      ┆ 9       ┆ 1       ┆ 905     ┆ 9       │
│ agency_z ┆ -0.0232 ┆ 0.02986 ┆ -0.0819 ┆ 0.03545 ┆ -0.7786 ┆ 420.557 ┆ 0.43664 │
│ :control ┆ 55      ┆ 7       ┆ 63      ┆ 3       ┆ 07      ┆ 527     ┆ 9       │
│ _level_c ┆         ┆         ┆         ┆         ┆         ┆         ┆         │
└──────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

**Random Effects Variance Components (ranef_var):**
shape: (2, 5)
┌─────────────┬─────────────────┬──────────┬──────────┬───────────┐
│ group       ┆ term            ┆ estimate ┆ conf_low ┆ conf_high │
│ ---         ┆ ---             ┆ ---      ┆ ---      ┆ ---       │
│ str         ┆ str             ┆ f64      ┆ f64      ┆ f64       │
╞═════════════╪═════════════════╪══════════╪══════════╪═══════════╡
│ participant ┆ sd__(Intercept) ┆ 0.136297 ┆ null     ┆ null      │
│ Residual    ┆ sd__Observation ┆ 0.223456 ┆ null     ┆ null      │
└─────────────┴─────────────────┴──────────┴──────────┴───────────┘

**Random Effects by Participant (ranef):**
shape: (8, 2)
┌───────┬─────────────┐
│ level ┆ (Intercept) │
│ ---   ┆ ---         │
│ str   ┆ f64         │
╞═══════╪═════════════╡
│ 9     ┆ -0.15673    │
│ 11    ┆ 0.116888    │
│ 12    ┆ 0.122582    │
│ 13    ┆ -0.044573   │
│ 14    ┆ -0.137396   │
│ 15    ┆ 0.207096    │
│ 16    ┆ -0.050697   │
│ 17    ┆ -0.05717    │
└───────┴─────────────┘

> **APA 7 Reporting Example:**
> Subjective agency ratings predicted recognition reaction times for controlled items, $\beta$ = -0.01, *SE* = 0.01, *t* = -0.41, *p* = 0.679.

