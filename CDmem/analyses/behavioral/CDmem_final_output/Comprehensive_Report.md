# CDmem Comprehensive Analysis Report

## Data Loading & Preprocessing

- Excluded for timeout: []
- Excluded for accuracy outliers: []
- Excluded for calibration failure: []
- Total recognition trials excluded for RT outliers: 10
- Final N participants: 1

## Sanity Checks

## ANALYSES

### 1) Does higher motor control lead to better memory?

**1A) 2 (Control Level) x 2 (Item Type) ANOVA on d'**
**1B) Binomial GLMM on OLD ITEMS ONLY: said_old ~ C(control_level) * C(item_type) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


**1C) Gaussian LMM on OLD ITEMS ONLY: log_mem_rt ~ C(control_level) * C(item_type) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


### 2) Does this memory effect depend on the conscious detection of control?

*Note: Restricted to `item_type == 'controlled'` because detection accuracy is only meaningful for items the participant actively controlled.*

**2D) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


**2E) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(detection_accuracy) * C(control_level) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


### 3) Can we predict this effect from participants' subjective agency ratings?

*Note: Restricted to `item_type == 'controlled'` as agency is primarily manipulated for controlled targets.*

**3F) Binomial GLMM on OLD CONTROLLED ITEMS ONLY: said_old ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


**3G) Gaussian LMM on OLD CONTROLLED ITEMS ONLY: log_mem_rt ~ C(agency_rating_ztransformed) * C(control_level) + (1 | participant)**
Model failed: Error: grouping factors must have > 1 sampled level


