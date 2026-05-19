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
┌──────────────────────┬───────────┬───────────┬───────────┬───────────┬──────────┬─────┬──────────┐
│ term                 ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat   ┆ df  ┆ p_value  │
│ ---                  ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---      ┆ --- ┆ ---      │
│ str                  ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64      ┆ f64 ┆ f64      │
╞══════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪══════════╪═════╪══════════╡
│ (Intercept)          ┆ -0.387506 ┆ 0.237068  ┆ -0.852151 ┆ 0.077138  ┆ -1.63458 ┆ inf ┆ 0.102137 │
│ control_level_c      ┆ 0.242877  ┆ 0.098865  ┆ 0.049104  ┆ 0.436649  ┆ 2.456639 ┆ inf ┆ 0.014024 │
│ item_type_c          ┆ 0.362796  ┆ 0.098971  ┆ 0.168817  ┆ 0.556775  ┆ 3.665687 ┆ inf ┆ 0.000247 │
│ control_level_c:item ┆ 0.180154  ┆ 0.197647  ┆ -0.207228 ┆ 0.567535  ┆ 0.91149  ┆ inf ┆ 0.362037 │
│ _type_c              ┆           ┆           ┆           ┆           ┆          ┆     ┆          │
└──────────────────────┴───────────┴───────────┴───────────┴───────────┴──────────┴─────┴──────────┘

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
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ -0.192278 ┆ 0.053548  ┆ -0.319062 ┆ -0.065495 ┆ -3.590767 ┆ 6.955894   ┆ 0.008943 │
│ control_leve ┆ -0.009196 ┆ 0.017481  ┆ -0.043513 ┆ 0.025121  ┆ -0.526036 ┆ 765.128196 ┆ 0.599016 │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ item_type_c  ┆ -0.024021 ┆ 0.017454  ┆ -0.058284 ┆ 0.010242  ┆ -1.376282 ┆ 764.979317 ┆ 0.169137 │
│ control_leve ┆ 0.047866  ┆ 0.035025  ┆ -0.020891 ┆ 0.116622  ┆ 1.366617  ┆ 765.201939 ┆ 0.172147 │
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
┌─────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                 ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                 ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)         ┆ -0.334566 ┆ 0.274171  ┆ -0.871931 ┆ 0.202798  ┆ -1.220285 ┆ inf ┆ 0.222357 │
│ detection_accuracy_ ┆ 0.612657  ┆ 0.194499  ┆ 0.231446  ┆ 0.993868  ┆ 3.149922  ┆ inf ┆ 0.001633 │
│ c                   ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
│ control_level_c     ┆ 0.222802  ┆ 0.192215  ┆ -0.153932 ┆ 0.599536  ┆ 1.159131  ┆ inf ┆ 0.246403 │
│ detection_accuracy_ ┆ -0.277484 ┆ 0.38809   ┆ -1.038127 ┆ 0.48316   ┆ -0.714997 ┆ inf ┆ 0.474611 │
│ c:control_l…        ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

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
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ -0.195599 ┆ 0.051051  ┆ -0.31402  ┆ -0.077179 ┆ -3.831482 ┆ 7.737922   ┆ 0.005331 │
│ detection_ac ┆ -0.036606 ┆ 0.032396  ┆ -0.100285 ┆ 0.027073  ┆ -1.129952 ┆ 418.406636 ┆ 0.259144 │
│ curacy_c     ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ control_leve ┆ 0.027804  ┆ 0.031509  ┆ -0.034132 ┆ 0.089741  ┆ 0.882435  ┆ 416.38326  ┆ 0.378051 │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ detection_ac ┆ -0.013912 ┆ 0.064173  ┆ -0.140055 ┆ 0.11223   ┆ -0.216792 ┆ 417.919416 ┆ 0.828476 │
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
┌─────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────┬──────────┐
│ term                ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ z_stat    ┆ df  ┆ p_value  │
│ ---                 ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ --- ┆ ---      │
│ str                 ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64 ┆ f64      │
╞═════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════╪══════════╡
│ (Intercept)         ┆ -0.245043 ┆ 0.262586  ┆ -0.759701 ┆ 0.269616  ┆ -0.933191 ┆ inf ┆ 0.350721 │
│ agency_z            ┆ 0.086396  ┆ 0.082475  ┆ -0.075252 ┆ 0.248044  ┆ 1.047544  ┆ inf ┆ 0.294849 │
│ control_level_c     ┆ 0.245103  ┆ 0.164524  ┆ -0.077358 ┆ 0.567565  ┆ 1.489769  ┆ inf ┆ 0.136285 │
│ agency_z:control_le ┆ 0.139145  ┆ 0.17458   ┆ -0.203026 ┆ 0.481316  ┆ 0.797025  ┆ inf ┆ 0.425437 │
│ vel_c               ┆           ┆           ┆           ┆           ┆           ┆     ┆          │
└─────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────┴──────────┘

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
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────┬──────────┐
│ term         ┆ estimate  ┆ std_error ┆ conf_low  ┆ conf_high ┆ t_stat    ┆ df         ┆ p_value  │
│ ---          ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---        ┆ ---      │
│ str          ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64        ┆ f64      │
╞══════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪════════════╪══════════╡
│ (Intercept)  ┆ -0.200462 ┆ 0.050262  ┆ -0.318057 ┆ -0.082866 ┆ -3.988355 ┆ 7.388429   ┆ 0.004719 │
│ agency_z     ┆ -0.005604 ┆ 0.01354   ┆ -0.032219 ┆ 0.021011  ┆ -0.413881 ┆ 415.260002 ┆ 0.679175 │
│ control_leve ┆ 0.021678  ┆ 0.026775  ┆ -0.030952 ┆ 0.074309  ┆ 0.809661  ┆ 415.146905 ┆ 0.418599 │
│ l_c          ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
│ agency_z:con ┆ -0.023255 ┆ 0.029867  ┆ -0.081963 ┆ 0.035453  ┆ -0.778607 ┆ 420.557527 ┆ 0.436649 │
│ trol_level_c ┆           ┆           ┆           ┆           ┆           ┆            ┆          │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────┴──────────┘

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

