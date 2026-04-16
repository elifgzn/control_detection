# CDmem Power Analysis Overview

**File:** `allpossible_power_CDmem.py`
Monte Carlo simulation-based power analysis for the Control Detection + Memory (CDmem) experiment.

---

## Purpose

Estimates statistical power for detecting a **control-level effect on recognition memory** (High control vs. Low control) across four primary and four supplementary analysis strategies, crossed with two effect sizes and four sample sizes.

The script covers all plausible analysis approaches so the most powerful justified strategy can be selected before data collection.

---

## Simulation Settings

| Parameter | Value | Notes |
|---|---|---|
| `N_SIMULATIONS` | 1000 | Monte Carlo iterations per cell. Use 200 for quick testing, 5000 for final results. |
| `SAMPLE_SIZES` | [30, 40, 50, 60] | Target range for an EEG study with practical constraints |
| `ALPHA` | 0.05 | Two-tailed significance threshold |
| `N_TRIALS_PER_CONDITION` | 60 | Target (old) trials per condition (High / Low control) |
| `N_FOILS` | 240 | New (never-seen) foil trials total |
| `np.random.seed` | 42 | Fixed for reproducibility |

---

## Parameters from the Literature

| Parameter | Value | Source |
|---|---|---|
| `BASELINE_DPRIME` | 0.74 | Average of congruent (0.78) and incongruent (0.70) conditions — Schreiner et al. (2024) |
| `DPRIME_SD` | 0.75 | Pooled SD across conditions — Schreiner et al. (2024) |
| `BASELINE_HIT_RATE` | 0.62 | Schreiner et al. (2024), congruent condition; confirmed by Wu et al. (2025), Exp. 4 |
| `HIT_RATE_SD` | 0.20 | Schreiner et al. (2024) |
| `BASELINE_FA_RATE` | 0.20 | Not directly reported; assumed as plausible value for old/new recognition. Affects only d′-to-hit-rate conversion. |
| `RI_SD` | 1.36 | Random intercept SD estimated by converting d′ SD to log-odds: `DPRIME_SD × (π / √3) ≈ 0.75 × 1.814 ≈ 1.36` |

---

## Effect Size Conversions

Cohen's d is converted to raw differences via:

- **d′ difference:** `Δd′ = Cohen's d × DPRIME_SD`
- **Hit rate difference:** `ΔHR = Cohen's d × HIT_RATE_SD`

| Cohen's d | Δd′ | d′ Low vs. High | ΔHR | HR Low vs. High |
|---|---|---|---|---|
| 0.2 (Small) | 0.150 | 0.665 vs. 0.815 | 0.040 | 0.600 vs. 0.640 |
| 0.5 (Medium) | 0.375 | 0.553 vs. 0.928 | 0.100 | 0.570 vs. 0.670 |

---

## Helper Functions

| Function | Description |
|---|---|
| `cohens_d_to_dprime_diff(d)` | Converts Cohen's d to a raw d′ difference using `DPRIME_SD` |
| `cohens_d_to_hr_diff(d)` | Converts Cohen's d to a raw hit rate difference using `HIT_RATE_SD` |
| `hr_to_log_odds(hr)` | Converts a probability to log-odds; clipped to [0.001, 0.999] to avoid ±∞ |

---

## Why GEE (Not GLMM) for Trial-Level Analyses?

GEE (Generalized Estimating Equations) is used for the trial-level power simulations instead of GLMM because:

1. **Binary outcomes** — trial-level recognition is 0/1
2. **Within-participant correlation** — handled via exchangeable working correlation structure
3. **Robust** — marginal estimates are consistent even if the working correlation is misspecified
4. **No random slopes required** — avoids the risk of singular fits with small N that would be common in GLMMs

---

## Primary Analyses

These test the **High vs. Low control** contrast only (no uncontrolled items).

---

### Analysis 1 — d′ Paired t-test

| Attribute | Detail |
|---|---|
| Type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| IV | Control condition (High vs. Low) — within-subject |
| DV | d′ per participant per condition |
| Data generating process | d′ per participant drawn from N(mean_condition, DPRIME_SD); means separated by Δd′ |
| Reference | Standard SDT approach — Schreiner et al. (2024) |

---

### Analysis 2 — Hit Rate Paired t-test

| Attribute | Detail |
|---|---|
| Type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| IV | Control condition (High vs. Low) |
| DV | Mean hit rate per participant per condition |
| Data generating process | Per-participant hit rate drawn from N(mean_condition, HIT_RATE_SD) |
| Reference | Wu et al. (2025); supplemental analysis in Schreiner et al. (2024) |

> Does not account for false alarms. Because foils have no condition structure in this design, hit rate is a clean and interpretable measure.

---

### Analysis 3 — GEE: Target Trials Only

| Attribute | Detail |
|---|---|
| Type | Generalized Estimating Equations (GEE), Binomial family, logit link |
| Package | `statsmodels.formula.api.gee` |
| IV | `control`: contrast-coded (High = +0.5, Low = −0.5) |
| DV | `recognized`: binary hit/miss per trial (0/1) |
| Working correlation | Exchangeable |
| Groups | Participant |
| Test statistic | p-value for `control` coefficient |

**Model formula:**
```
recognized ~ control
groups = participant
```

**Data generating process:**
1. Each participant gets a random intercept drawn from N(0, RI_SD)
2. Trial log-odds: `intercept + RI + true_slope × control`
3. Hit/miss sampled from Bernoulli with p = logistic(log-odds)

> Avoids the foil dummy-assignment problem entirely by analysing only target trials.

---

### Analysis 4 — GEE: Item Type × Control Interaction (All Trials)

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial family, logit link |
| Package | `statsmodels.formula.api.gee` |
| IVs | `item_type`: Target = +0.5, Foil = −0.5; `control`: High = +0.5, Low = −0.5; interaction |
| DV | `said_old`: binary 'old' response per trial (0/1) |
| Working correlation | Exchangeable |
| Test statistic | p-value for `item_type:control` interaction coefficient |

**Model formula:**
```
said_old ~ item_type * control
groups = participant
```

> **Why the interaction captures d′ logic:** Hit rate (targets) changes across conditions → interaction is non-zero. FA rate (foils) does **not** change → no contribution from foils. The interaction therefore selectively reflects the change in hit rate, which equals a change in d′ given constant FA rate.

> **Foil dummy assignment:** Foils have no genuine control condition and are split equally (120 per dummy condition). Because FA rate is identical across dummy conditions by construction, this does not bias the interaction estimate — but it is an approximation (resolved cleanly in Supp 4).

---

## Supplementary Analyses

These incorporate **uncontrolled items** (seen simultaneously with controlled items but never under participant control) as a **third condition**.

**Design rationale:** On each encoding trial, participants see two images — one *controlled* item (the cursor tracked it) and one *uncontrolled* item (the other image, moving randomly). Both appear as old items in the recognition test alongside completely new foils. This gives three categories of old items:

| Category | N per participant | Description |
|---|---|---|
| High control items | 60 | Controlled target, High condition trial |
| Low control items | 60 | Controlled target, Low condition trial |
| Uncontrolled items | 120 (60+60) | Co-presented uncontrolled item, pooled across conditions |

> **Important caveat:** Uncontrolled items were viewed in the same context and at the same time as controlled items — they are *not* a clean "no encoding" baseline. Interpret supplementary results with caution.

> **Conservative assumption:** Uncontrolled hit rate = Low control hit rate (worst case for detecting a motor control effect). Power for Supp 1–4 would be higher if uncontrolled items are remembered less well than low control items.

---

### Supplementary Analysis 1 — d′ One-Way Repeated-Measures ANOVA (3 conditions)

| Attribute | Detail |
|---|---|
| Type | One-way repeated-measures ANOVA (approximated via `scipy.stats.f_oneway`) |
| Package | `scipy.stats.f_oneway` |
| IV | Control type: High / Low / Uncontrolled (3 levels, within-subject) |
| DV | d′ per participant per condition |
| Note | `f_oneway` is for independent groups — ignores within-person correlation, slightly overestimates power. Sufficient for planning purposes. |

> Uncontrolled items from High and Low condition trials are pooled into a single 'uncontrolled' category (120 trials → more stable d′ estimate).

---

### Supplementary Analysis 2 — Hit Rate One-Way Repeated-Measures ANOVA (3 conditions)

| Attribute | Detail |
|---|---|
| Type | One-way repeated-measures ANOVA (approximated via `scipy.stats.f_oneway`) |
| Package | `scipy.stats.f_oneway` |
| IV | Control type: High / Low / Uncontrolled (within-subject) |
| DV | Hit rate per participant per condition |

---

### Supplementary Analysis 3 — GEE: All Old Items, 3-Level Control Type

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial family, logit link |
| Package | `statsmodels.formula.api.gee` |
| IVs | `is_high` (1 = High control target, 0 otherwise); `is_low` (1 = Low control target, 0 otherwise); reference = Uncontrolled |
| DV | `recognized`: binary hit/miss per trial |
| Test statistic | p-value for `is_high` coefficient (High control vs. Uncontrolled) |

**Model formula:**
```
recognized ~ is_high + is_low
groups = participant
```

> The key contrast is `is_high`: does motor control in the High condition boost memory beyond mere co-viewing? Uncontrolled items (both from High and Low condition trials) are the reference category.

---

### Supplementary Analysis 4 — GEE: item_is_old × Control Type Interaction, 3-Level (All Trials)

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial family, logit link |
| Package | `statsmodels.formula.api.gee` |
| IVs | `item_is_old` (1 = any old item, 0 = new foil); `is_high` (1 = High control); `is_low` (1 = Low control); interactions `item_is_old:is_high`, `item_is_old:is_low` |
| DV | `said_old`: binary 'old' response |
| Test statistic | p-value for `item_is_old:is_high` coefficient |

**Model formula:**
```
said_old ~ item_is_old * is_high + item_is_old * is_low
groups = participant
```

> **Key advantage over primary Analysis 4:** Foils require **no dummy assignment** to control conditions because `control_type` is a property of old items only. New foils simply have `item_is_old = 0` and are not assigned to any condition. This is the conceptually cleanest version of the interaction model.

---

## Grid of Simulations

| Dimension | Values |
|---|---|
| Effect sizes | Small (d = 0.2), Medium (d = 0.5) |
| Sample sizes | N = 30, 40, 50, 60 |
| Analyses | 4 primary + 4 supplementary = 8 |
| Total cells | 2 × 4 × 8 = **64** |

Each cell runs `N_SIMULATIONS` = 1000 Monte Carlo iterations. Power = proportion of iterations yielding p < 0.05.

---

## Output

| Output | Description |
|---|---|
| Console | Progress log per cell: analysis, effect size, N, power, runtime |
| `results_CDmem.md` | Formatted markdown tables: parameters, analysis descriptions, effect size conversions, power tables (Primary & Supplementary), notes |

Power cells meeting the **80% threshold** are flagged with `[80%+]` in the output tables.

---

## Notes and Limitations

| Limitation | Detail |
|---|---|
| Random intercept SD | Estimated indirectly from d′ SD via log-odds conversion. Actual between-participant variability is unknown. |
| False alarm rate | Assumed 0.20. Does not directly affect Analyses 1, 2, 3, S1, S2, or S3. Only enters Analyses 4 and S4 via the foil log-odds. |
| Foil dummy assignment | Only relevant for primary Analysis 4 — resolved cleanly in Supplementary Analysis S4. |
| Literature mismatch | Parameters are from a choice paradigm (Schreiner et al.) — the true effect in a cursor control paradigm may differ. |
| Small effect underpowering | For d = 0.2, N = 30–60 is likely underpowered regardless of analysis. Frame as exploratory if the true effect is small. |
| GEE approximation | `f_oneway` used for Supp 1 & 2 ignores within-person correlation → slight power overestimate for planning. |
| Conservative Supp assumption | If uncontrolled items are remembered *less* than low control items (plausible), power for Supp 1–4 would be higher than estimated. |

---

## Recommendation (from script)

> For **small effects (d = 0.2):** N = 30–60 is likely underpowered across all analysis strategies. Frame as exploratory.
>
> For **medium effects (d = 0.5):** GEE analyses (Analysis 3 & 4) offer more power than paired t-tests by leveraging trial-level variance. Supplementary GEE analyses (S3, S4) additionally exploit the item-type structure.

---

*Generated from `allpossible_power_CDmem.py`*
*References: Schreiner et al. (2024); Wu et al. (2025)*
