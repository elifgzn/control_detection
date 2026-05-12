# CDmem Preregistration: Recommended Analyses

## Experiment Overview (for context)

Participants detect which of two moving shapes they are controlling (via partial mouse-direction coupling), across **High** (~75% accuracy target) and **Low** (~55% accuracy target) control conditions calibrated individually with QUEST+. After the control-detection task, a surprise recognition memory test probes whether images encoded during High vs. Low control trials are remembered differently. A foil set (never-seen images) provides the false-alarm baseline for SDT measures.

Key design features relevant to analysis choice:
- **Within-participant 2-condition design** (High vs. Low control at the trial level).
- Each trial also has two image roles: the **controlled** image (the one the participant moved) vs. the **uncontrolled** image (distractor).
- Agency ratings (1–7 Likert) are collected per trial.
- Trial-level binary recognition responses are the primary DV.

---

## Recommended Pre-registered Analyses

### 0. Participant Exclusion Criteria *(must be pre-registered)*

These are already in the script and well-justified — include both verbatim.

| Criterion | Threshold | Reference |
|---|---|---|
| Timeout rate ≥ 50% in either condition (test phase) | `TIMEOUT_THRESHOLD = 0.50` | Haridi et al., 2025 |
| Accuracy z-score > ±2.5 SD from group mean in either condition | `ACCURACY_SD_THRESHOLD = 2.5` | Standard outlier criterion |

**Trial-level:** recognition trials with RT > participant mean + 3 SD excluded (Ren et al., 2026).

---

### 1. Manipulation Check — Control Detection Accuracy & SoA Ratings

> **Pre-register as a prerequisite.** If control detection and sense of agency do not differ between conditions, the memory effect cannot be attributed to the control manipulation.

**Tests:** 
1. **Control Detection Accuracy:** Paired t-test on participant-level mean accuracy (High vs. Low) + binomial GLMM with contrast-coded `control_c` (High = +0.5, Low = −0.5) and random intercepts/slopes per participant.
2. **Sense of Agency (SoA) Ratings:** Paired t-test on participant-level mean SoA ratings across conditions.

**Why appropriate:**
- The QUEST+ staircase targets 85% (High) and 55% (Low) — these targets must be reached for the memory comparison to be interpretable.
- The SoA ratings provide a subjective manipulation check confirming that the objective control differences translated into subjective feelings of agency.
- Paired t-tests are standard for 2-condition within-subject designs.
- The GLMM handles trial-level binary accuracy data correctly and accounts for participant random effects.

---

### 2. Primary Analysis 1 — d-prime: High vs. Low Control (`run_analysis_1_dprime_ttest`)

> **Pre-register as the primary confirmatory test.**

**Test:** Paired t-test comparing participant-level d-prime between the **High-control** and **Low-control** conditions (all items). Report Cohen's *d*.

**Why this is the primary measure:**
- d-prime is the gold-standard SDT measure; it corrects hit rates for response bias (false alarm rate), making it a pure index of memory discriminability — critical when participants might show yes-bias.
- It directly operationalises the core hypothesis: *"Items encoded under high subjective control are better recognised."*
- The paired t-test is the natural test for the within-participant 2-condition comparison and is fully powered by your Monte Carlo simulation.
- Precedent: virtually all recognition memory × agency studies use d-prime as the primary DV (e.g., Eitam & Higgins, 2010; Chambon et al., 2014; Humphreys et al., 2023).

> [!IMPORTANT]
> d-prime from **all items**, using the shared foil false alarm rate as the noise baseline.

---

### 3. Primary Analysis 2 (trial-level) — GLMM Interaction on All Trials (`run_analysis_4_interaction_glmm`)

> **Pre-register as confirmatory, alongside Analysis 1.**

**Test:** Binomial GLMM on **all recognition trials** (targets + foils):
```
said_old ~ item_type_c * control_c + (1 | participant)
```
Contrast coding: Target = +0.5 / Foil = −0.5; High = +0.5 / Low = −0.5.  
Use the simpler fallback random effects structure `(1 | participant)`.

**Why Analysis 4, not Analysis 3, is the correct trial-level primary test:**

Analysis 3 (`said_old ~ control_c`, targets only) has **no false-alarm baseline** — its fixed effect of `control_c` captures only how often participants say "old" to old items, i.e., it is a **hit-rate GLMM**, not a d-prime analog.

Analysis 4 includes foils and therefore:
- The **`item_type_c` main effect** captures overall recognition sensitivity (hit rate vs. FA rate — the logit-scale equivalent of overall d-prime).
- The **`item_type_c × control_c` interaction** captures whether the hit-minus-FA gap is larger in the High condition — which is exactly what d-prime tests at the participant level, now estimated at the trial level via a logit link.

This is the proper trial-level signal-detection GLM (see DeCarlo, 1998; Sheu & Chen, 2024 for SDT-GLMM equivalences).

> [!NOTE]
> **Foil dummy-coding caveat:** Foils have no real control condition (they were never shown during the control task). In the script they receive a balanced dummy `control_c` assignment (first half +0.5, second half −0.5 per participant). Because FA rates are expected to be identical across both dummy cells by design, the interaction is empirically driven by the **target trials only**, even though foils provide the FA baseline. State this explicitly in the Methods.

---

### 4. Primary Analysis 3 — False Alarm Rate Sanity Check (`run_supp_analysis_5_fa_check`)

> **Pre-register as a primary data quality check.**

**Test:** One-sample t-test of FA rates against 0.

**Why include:**
- The entire SDT approach depends on foils being genuinely unseen. If FA rates are high or variable, d-prime estimates are unreliable.
- Pre-registering this is good practice and increases reviewer confidence.
- It's a quick, unambiguous check.

---

### 5. Secondary Analysis — 2×2 GLMM: Trial Level × Item Type (`run_supp_analysis_3_glmm_2x2`)

> **Pre-register as a secondary analysis** (not primary — it goes beyond the core hypothesis but is theoretically motivated).

**Test:** Binomial GLMM on all old items (controlled + uncontrolled):
```
said_old ~ trial_level_c * item_type_c + (1 | participant)
```
Contrast coding: High = +0.5 / Low = −0.5; Controlled = +0.5 / Uncontrolled = −0.5.

**Why this matters scientifically:**
- This is arguably the most theoretically informative analysis. It tests whether the memory advantage is specifically for the **controlled** object (the one actively moved), or whether it generalises to the co-present uncontrolled object.
- A **main effect of trial_level only** (no interaction) → control level at encoding generally boosts memory for all items in the trial (arousal/attention account).
- An **interaction** → the memory boost is specific to the controlled item, implicating action-binding or selective attention (agency account).
- This dissociation is central to distinguishing agency accounts (Elsner & Hommel, 2001) from general arousal/attention accounts.
- Compatible papers (Ren et al., 2026; Chambon et al., 2014) report exactly this 2×2 structure.

> [!NOTE]
> Pre-register the **GLMM version** (Supp Analysis 3) as the primary test of this 2×2 question. The RM-ANOVA versions (Supp 1 & 2) can be mentioned as summary-statistics complements but are less appropriate for binary DVs.

---

### 6. Exploratory (Pre-registered as Exploratory) — Agency → Memory (`run_analysis_7_agency_glmm`)

> **Pre-register as an exploratory analysis** (openly, not as confirmatory). Label it clearly in the paper.

**Test:** Binomial GLMM on old items with continuous, within-participant z-scored agency rating as predictor:
```
said_old ~ agency_rating_z + (1 | participant)
```

**Why include:**
- Agency ratings are the subjective counterpart to the objective control manipulation. Testing whether *felt* agency predicts memory — over and above the objective condition — is a key mechanistic question.
- Pre-registering it as exploratory is honest and protects against inflated Type I error, while signalling you intended to test it.
- Multiple studies show that within-trial SoA ratings predict memory independently of accuracy (e.g., Damen et al., 2024; Humphreys et al., 2023 style analyses).
- Within-participant z-scoring (already implemented) is best practice to remove scale-use bias.

---

### 7. Exploratory — Recognition RT ~ Control Level (`run_analysis_8_rt_control`)

> **Pre-register as an exploratory analysis.** RT in the recognition phase provides a complementary measure to accuracy-based sensitivity.

**Test:** Lognormal LMM on correct recognition trials (hits + correct rejections):
```
log(mem_rt) ~ is_old + is_old:control_c + (1 | participant)
```
`is_old` = 1 for targets, 0 for foils. `control_c`: High = +0.5, Low = −0.5 (set to 0 for foils, zeroed out by the interaction).

A paired t-test on participant-level mean raw RT (High vs. Low, correct old items) is reported as a descriptive complement.

**Why include:**
- Recognition RT captures processing fluency — faster responses to old items from one condition may indicate stronger memory traces, even when accuracy-based sensitivity (d′) does not differ.
- The lognormal model (Gaussian on log-RT) is appropriate for right-skewed RT distributions (Ulrich & Miller, 1993; Van der Linden, 2006).
- The `is_old:control_c` interaction tests the theoretically relevant question: does the control manipulation at encoding affect recognition *speed* specifically for old items?
- Ren et al. (2026) use the same nested-interaction structure.
- An all-trials version (including errors) is retained in the code (commented out) for the case where d′ is not significantly affected by the manipulation.

---

### 8. Exploratory — Recognition RT ~ Agency Rating (`run_analysis_9_rt_agency`)

> **Pre-register as an exploratory analysis.** Tests whether within-participant variation in subjective agency at encoding predicts recognition processing speed.

**Test:** Lognormal LMM on correct recognition trials:
```
log(mem_rt) ~ is_old + is_old:agency_z + (1 | participant)
```
`agency_z` = agency_rating z-scored within each participant. Set to 0 for foils (zeroed out by `is_old` interaction).

**Why include:**
- Complements Analysis 9's test on accuracy (Analysis 7): even if SoA does not predict *whether* an item is recognised, it may predict *how quickly* old items are recognised.
- Within-participant z-scoring removes scale-use bias, isolating trial-to-trial variation in felt agency.
- Same lognormal model rationale as Analysis 8.

---

### 9. Exploratory — Recognition Sensitivity by Detection (`run_supp_analysis_6_detection_glmm`)

> **Pre-register as an exploratory analysis.** Tests whether correctly detecting control at encoding predicts better subsequent memory for controlled items.

**Test:** Binomial GLMM on **controlled target items only**:
```
said_old ~ detection_c + (1 | participant)
```
`detection_c`: Detected = +0.5, Not Detected = −0.5. Uncontrolled items are excluded (detection accuracy is not applicable to them).

**Why include:**
- This analysis directly tests the "awareness of control → memory" link: do participants remember controlled items better when they *noticed* the control?
- It goes beyond the trial-level control condition (high vs. low) by asking whether the participant's *subjective detection* of control matters for memory.
- The existing detection breakdown plots (hit rate and d′ split by detected/not-detected) already visualise this effect — this adds the corresponding statistical test.

---

### 10. Exploratory — Recognition RT by Detection (`run_analysis_10_rt_detection`)

> **Pre-register as an exploratory analysis.** Tests whether correct detection of control at encoding predicts recognition processing speed for controlled items.

**Test:** Lognormal LMM on **controlled target items** (hits only):
```
log(mem_rt) ~ detection_c + (1 | participant)
```
`detection_c`: Detected = +0.5, Not Detected = −0.5.

**Why include:**
- Complements Supp 6: even if detection accuracy does not predict *whether* a controlled item is recognised, it may predict *how quickly* it is recognised.
- Faster recognition RT for detected items would suggest that awareness of control at encoding creates stronger or more accessible memory representations.
- Same lognormal model rationale as Analyses 8 and 9.

---

### 11. Exploratory — Recognition RT by Item Type × Trial Level (`run_analysis_11_rt_itemtype`)

> **Pre-register as an exploratory analysis.** Tests whether recognition RT differs between controlled and uncontrolled items, and whether this depends on the control level (high vs. low).

**Test:** Lognormal LMM on **target items** (hits only):
```
log(mem_rt) ~ trial_level_c * item_type_c + (1 | participant)
```
`trial_level_c`: High = +0.5, Low = −0.5. `item_type_c`: Controlled = +0.5, Uncontrolled = −0.5.

**Why include:**
- This is the RT counterpart to Supplementary Analysis 3 (the 2×2 GLMM on recognition sensitivity).
- The **main effect of `item_type_c`** tests whether controlled items are recognised faster than uncontrolled items — suggesting preferential encoding of the actively controlled object.
- The **interaction** tests whether the controlled/uncontrolled RT difference depends on trial-level control conditions — a selective RT advantage for controlled items in high-control trials would support action-binding accounts over general attention/arousal.
- Same lognormal model rationale as Analyses 8–10.
