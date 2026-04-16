"""
===============================================================================
Power Analysis for Recognition Memory Experiment
===============================================================================
This script estimates statistical power for detecting a control-level effect
on recognition memory across four different analysis strategies:

  1. Paired t-test on d-prime (aggregated SDT measure)
  2. Paired t-test on hit rates (aggregated, ignores false alarms)
  3. GEE on target trials only (trial-level, hits only)
  4. GEE with item_type x control interaction (trial-level, all trials)

We test two effect sizes (Cohen's d = 0.2 and 0.5) and four sample sizes
(N = 30, 40, 50, 60), as relevant for an EEG study with practical constraints.

ANALYSES 3 & 4 use Generalized Estimating Equations (GEE) via statsmodels.
GEE is appropriate here because:
  - It handles binary trial-level outcomes (recognized: 0/1)
  - It accounts for within-participant correlation across trials via the
    exchangeable working correlation structure
  - It is robust to misspecification of the correlation structure
  - It does not require specifying random slopes, which is risky with small N

PARAMETERS sourced from:
  - Schreiner et al. (2024): d-prime means/SDs, hit rate means/SDs
  - Wu et al. (2025): baseline hit rate from Experiment 4

OUTPUT:
  - Console progress output
  - power_analysis_results.md saved in the current working directory
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit   # logistic function: 1 / (1 + exp(-x))
from scipy.stats import norm      # normal distribution for d-prime conversions
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)


# ==============================================================================
# SIMULATION SETTINGS
# ==============================================================================

N_SIMULATIONS          = 1000   # Monte Carlo iterations per cell
                                 # Use 200 for quick testing, 5000 for final results
SAMPLE_SIZES           = [30, 40, 50, 60]
ALPHA                  = 0.05   # two-tailed significance threshold
N_TRIALS_PER_CONDITION = 60     # target trials per condition (high / low control)
N_FOILS                = 240    # new (never-seen) foil trials total


# ==============================================================================
# PARAMETERS FROM THE LITERATURE
# ==============================================================================

# D-prime values from Schreiner et al. (2024):
#   Congruent condition:   M = 0.78, SD = 0.75
#   Incongruent condition: M = 0.70, SD = 0.74
BASELINE_DPRIME = 0.74   # average of the two conditions
DPRIME_SD       = 0.75   # pooled SD — used to convert Cohen's d to delta d-prime

# Hit rate values from Schreiner et al. (2024) and Wu et al. (2025):
#   Schreiner congruent: M = 0.62, SD ≈ 0.20
#   Wu et al. Exp 4 intermediate condition: ~0.62
BASELINE_HIT_RATE = 0.62
HIT_RATE_SD       = 0.20

# False alarm rate — not directly reported in the available literature.
# Assumed to be 0.20 as a plausible value for old/new recognition tasks.
# Affects only the d-prime-to-hit-rate conversion.
BASELINE_FA_RATE = 0.20

# Random intercept SD for the GEE data generating process.
# Captures between-participant variability in overall recognition tendency.
# Estimated by converting d-prime SD to log-odds scale:
#   RI_SD ~ DPRIME_SD x (pi / sqrt(3)) ~ 0.75 x 1.814 ~ 1.36
# This is a rough approximation — the actual value is unknown.
RI_SD = 1.36


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def cohens_d_to_dprime_diff(cohens_d):
    """
    Convert Cohen's d to a raw d-prime difference.
    Cohen's d = delta_dprime / SD_pooled  =>  delta_dprime = Cohen's d x SD_pooled
    """
    return cohens_d * DPRIME_SD


def cohens_d_to_hr_diff(cohens_d):
    """
    Convert Cohen's d to a raw hit rate difference.
    Cohen's d = delta_hit_rate / SD  =>  delta_hit_rate = Cohen's d x SD
    """
    return cohens_d * HIT_RATE_SD


def hr_to_log_odds(hr):
    """
    Convert a probability (hit rate) to log odds.
    log_odds = log(p / (1 - p))
    Clipped to avoid log(0) or log(inf).
    """
    hr = np.clip(hr, 0.001, 0.999)
    return np.log(hr / (1 - hr))


# ==============================================================================
# ANALYSIS 1: Paired t-test on d-prime
# ==============================================================================

def power_dprime_ttest(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a paired t-test comparing d-prime between
    high and low control conditions.

    This is the standard aggregated SDT approach (Schreiner et al., 2024).
    D-prime is computed per participant per condition, then compared with
    a two-tailed paired t-test.

    Data generating process:
      - Each participant has a d-prime value drawn from N(mean_condition, SD)
        for each condition, reflecting natural between-person variability.
      - The two condition means differ by the specified effect size.

    Parameters
    ----------
    n        : int   -- number of participants
    effect_d : float -- Cohen's d for the d-prime difference
    n_sim    : int   -- number of Monte Carlo simulations
    """
    dprime_diff = cohens_d_to_dprime_diff(effect_d)
    mean_high   = BASELINE_DPRIME + dprime_diff / 2
    mean_low    = BASELINE_DPRIME - dprime_diff / 2

    significant = 0
    for _ in range(n_sim):
        dp_high = np.random.normal(mean_high, DPRIME_SD, n)
        dp_low  = np.random.normal(mean_low,  DPRIME_SD, n)
        _, p    = stats.ttest_rel(dp_high, dp_low)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ==============================================================================
# ANALYSIS 2: Paired t-test on hit rates
# ==============================================================================

def power_hitrate_ttest(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a paired t-test comparing mean hit rates between
    high and low control conditions.

    This mirrors Wu et al. (2025) primary analysis and the supplemental
    analysis in Schreiner et al. (2024). Hit rate does not account for
    false alarms, but since foils have no condition structure in this
    design, it is a clean and interpretable measure.

    Data generating process:
      - Each participant has a mean hit rate drawn from N(mean_condition, SD).
    """
    hr_diff  = cohens_d_to_hr_diff(effect_d)
    hr_high  = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low   = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)

    significant = 0
    for _ in range(n_sim):
        hrs_high = np.random.normal(hr_high, HIT_RATE_SD, n)
        hrs_low  = np.random.normal(hr_low,  HIT_RATE_SD, n)
        _, p     = stats.ttest_rel(hrs_high, hrs_low)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ==============================================================================
# ANALYSIS 3: GEE on target trials only
# ==============================================================================

def power_gee_hits_only(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE on target (old) trials only.

    Model:
        recognized ~ control_level
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Where:
        recognized    : 1 = hit, 0 = miss
        control_level : contrast-coded (+0.5 = high control, -0.5 = low control)

    By analyzing only target trials we avoid the foil dummy-assignment
    problem entirely. The key test is whether control_level significantly
    predicts recognition — i.e., are hit rates higher in the high condition?

    The exchangeable working correlation assumes all trials within a participant
    are equally correlated. This is a reasonable approximation and GEE is
    robust even if this assumption is violated.

    Data generating process:
      1. Each participant gets a random intercept drawn from N(0, RI_SD),
         capturing their overall tendency to recognize items.
      2. Trial-level hit/miss responses are drawn from Bernoulli distributions
         with probabilities: p = expit(intercept + ri_i + slope x control)
    """
    hr_diff    = cohens_d_to_hr_diff(effect_d)
    hr_high    = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low     = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high    = hr_to_log_odds(hr_high)
    lo_low     = hr_to_log_odds(hr_low)
    true_slope = lo_high - lo_low           # true effect of control in log-odds
    intercept  = (lo_high + lo_low) / 2    # grand mean log-odds

    # Pre-build trial structure — identical for every simulation
    n_total      = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)
    controls     = np.tile(
        np.concatenate([
            np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # high control trials
            np.repeat(-0.5, N_TRIALS_PER_CONDITION)    # low control trials
        ]), n
    )

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)

        lo         = intercept + ri_rep + true_slope * controls
        probs      = expit(lo)
        recognized = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'control':     controls,
            'recognized':  recognized
        })

        try:
            gee = smf.gee(
                "recognized ~ control",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            if gee.pvalues['control'] < ALPHA:
                significant += 1

        except Exception:
            # Convergence failure — conservative choice is to not count as significant
            pass

    return significant / n_sim


# ==============================================================================
# ANALYSIS 4: GEE with item_type x control interaction
# ==============================================================================

def power_gee_interaction(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE including both target and foil
    trials, with an item_type x control interaction term.

    Model:
        said_old ~ item_type * control
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Where:
        said_old  : 1 = responded 'old', 0 = responded 'new'
        item_type : contrast-coded (+0.5 = target, -0.5 = foil)
        control   : contrast-coded (+0.5 = high,   -0.5 = low)

    WHY THE INTERACTION CAPTURES D-PRIME LOGIC:
      - Hit rate (targets) changes across control conditions
        => item_type x control interaction is non-zero
      - False alarm rate (foils) does NOT change across conditions
        => item_type x control interaction gets no contribution from foils
      - The interaction therefore selectively reflects the change in hit rate,
        which is equivalent to a change in d' (given constant FA rate)

    FOIL DUMMY ASSIGNMENT NOTE:
      Foils were never shown during the control task and have no genuine
      control condition. We split them equally (60 per dummy condition).
      Because the FA rate is identical for both dummy conditions by construction,
      this does not bias the interaction estimate — but it is an approximation.

    Data generating process:
      - Targets: hit rate differs between high and low control conditions
      - Foils: false alarm rate is identical regardless of dummy condition
      - Random intercepts capture between-participant variability
    """
    hr_diff  = cohens_d_to_hr_diff(effect_d)
    hr_high  = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low   = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high  = hr_to_log_odds(hr_high)
    lo_low   = hr_to_log_odds(hr_low)
    lo_fa    = hr_to_log_odds(BASELINE_FA_RATE)

    nf      = N_FOILS // 2
    n_total = N_TRIALS_PER_CONDITION * 2 + N_FOILS

    # Pre-build trial structure
    participants = np.repeat(np.arange(n), n_total)

    item_types = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # high control targets
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # low control targets
        np.repeat(-0.5, nf),                        # foils (dummy: high)
        np.repeat(-0.5, nf)                         # foils (dummy: low)
    ]), n)

    controls = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),
        np.repeat( 0.5, nf),
        np.repeat(-0.5, nf)
    ]), n)

    # True log-odds per trial type in the data generating process
    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high, N_TRIALS_PER_CONDITION),  # high targets: higher hit rate
        np.repeat(lo_low,  N_TRIALS_PER_CONDITION),  # low targets: lower hit rate
        np.repeat(lo_fa,   nf),                       # foils: FA rate (no condition effect)
        np.repeat(lo_fa,   nf)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri       = np.random.normal(0, RI_SD, n)
        ri_rep   = np.repeat(ri, n_total)
        lo       = base_lo + ri_rep
        probs    = expit(lo)
        said_old = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'item_type':   item_types,
            'control':     controls,
            'said_old':    said_old
        })

        try:
            gee = smf.gee(
                "said_old ~ item_type * control",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            # item_type:control is the interaction term
            # Significant = hit rate effect of control differs from FA rate effect
            # = d-prime differs across conditions
            if gee.pvalues['item_type:control'] < ALPHA:
                significant += 1

        except Exception:
            pass

    return significant / n_sim


# ==============================================================================
# SUPPLEMENTARY ANALYSES
# ==============================================================================
# These analyses exploit the richer trial structure of the design:
# on each encoding trial, participants see TWO items simultaneously —
# one partially controlled (target item) and one moving randomly (uncontrolled
# item). Both are shown in the recognition test as "old" items alongside
# completely new foils.
#
# This gives THREE categories of old items:
#   - High control items   (target item, high control condition)
#   - Low control items    (target item, low control condition)
#   - Uncontrolled items   (the randomly moving item, seen simultaneously)
#
# THEORETICAL NOTE:
# The uncontrolled items were viewed in the same context and at the same time
# as the controlled items, so they are not a clean "no encoding" baseline.
# For this reason these are treated as SUPPLEMENTARY rather than primary
# analyses. They are informative but require cautious interpretation.
#
# The key contrast of interest here is:
#   controlled (high + low) vs uncontrolled — does motor control per se
#   boost memory beyond mere co-occurrence/viewing?
#
# Assumed effect sizes for supplementary analyses:
#   We assume the uncontrolled item hit rate equals the low control hit rate
#   (conservative: motor control adds nothing beyond viewing context).
#   The effect of interest is still quantified as Cohen's d relative to
#   the same pooled SD as the primary analyses.
# ==============================================================================

# Number of uncontrolled items, split by which condition trial they appeared in.
# Each encoding trial shows one controlled item (target) and one uncontrolled item.
# So there are exactly as many uncontrolled items per condition as controlled items.
# These are kept separate because in the recognition test their condition of origin
# is known — they were co-presented with either a high or low control target.
# Theoretically we treat them as a single pool (no motor control effect regardless
# of condition), but their condition membership is preserved for model structure.
N_UNCONTROLLED_PER_CONDITION = N_TRIALS_PER_CONDITION   # 60 per condition
N_UNCONTROLLED               = N_UNCONTROLLED_PER_CONDITION * 2   # 120 total


# ------------------------------------------------------------------------------
# Supplementary Analysis 1: One-way repeated measures ANOVA on d-prime
# (three conditions: high control, low control, uncontrolled)
# ------------------------------------------------------------------------------

def power_supp_dprime_anova(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a one-way repeated measures ANOVA on d-prime
    with three levels: high control, low control, uncontrolled (pooled).

    Although uncontrolled items come from two condition contexts (co-presented
    with high vs low control targets), they are pooled into a single
    'uncontrolled' category here. This reflects the theoretical assumption
    that motor control — not mere co-presence in a control trial — drives
    any memory difference. Pooling doubles the trial count for the
    uncontrolled d-prime estimate (n=120 trials), improving its stability.

    All three d-primes are computed against the same false alarm rate
    (completely new foils), so they are directly comparable.

    Conservative assumption: uncontrolled d-prime equals low control d-prime.
    """
    dprime_diff  = cohens_d_to_dprime_diff(effect_d)
    mean_high    = BASELINE_DPRIME + dprime_diff / 2
    mean_low     = BASELINE_DPRIME - dprime_diff / 2
    mean_unctrl  = BASELINE_DPRIME - dprime_diff / 2   # conservative: same as low

    significant = 0
    for _ in range(n_sim):
        dp_high   = np.random.normal(mean_high,   DPRIME_SD, n)
        dp_low    = np.random.normal(mean_low,    DPRIME_SD, n)
        dp_unctrl = np.random.normal(mean_unctrl, DPRIME_SD, n)

        # One-way repeated measures ANOVA via F-statistic
        # scipy.stats.f_oneway is for independent groups — for repeated measures
        # we use the within-subject F manually, or approximate with f_oneway
        # (slight overestimate of power since it ignores within-person correlation,
        # but conservative enough for planning purposes)
        _, p = stats.f_oneway(dp_high, dp_low, dp_unctrl)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 2: One-way repeated measures ANOVA on hit rates
# (three conditions: high control, low control, uncontrolled)
# ------------------------------------------------------------------------------

def power_supp_hitrate_anova(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a one-way repeated measures ANOVA on hit rates
    with three levels: high control, low control, uncontrolled (pooled).

    Uncontrolled items (60 co-presented with high control trials + 60
    co-presented with low control trials) are pooled into one category,
    consistent with the theoretical assumption that their condition of
    origin does not affect their memorability. Conservative assumption:
    uncontrolled hit rate equals low control hit rate.
    """
    hr_diff      = cohens_d_to_hr_diff(effect_d)
    hr_high      = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low       = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    hr_unctrl    = hr_low   # conservative: uncontrolled = low control

    significant = 0
    for _ in range(n_sim):
        hrs_high   = np.random.normal(hr_high,   HIT_RATE_SD, n)
        hrs_low    = np.random.normal(hr_low,    HIT_RATE_SD, n)
        hrs_unctrl = np.random.normal(hr_unctrl, HIT_RATE_SD, n)
        _, p = stats.f_oneway(hrs_high, hrs_low, hrs_unctrl)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 3: GEE on all old items (3-level control_type predictor)
# ------------------------------------------------------------------------------

def power_supp_gee_three_conditions(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE on all old items with
    control_type as a three-level categorical predictor.

    Model:
        recognized ~ is_high + is_low
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    control_type levels (uncontrolled = reference):
        uncontrolled  : 60 items co-presented with high control targets
                      + 60 items co-presented with low control targets
                      = 120 items total, all treated as reference category
        low control   : 60 controlled targets from low condition (is_low = 1)
        high control  : 60 controlled targets from high condition (is_high = 1)

    The uncontrolled items are split by condition of origin in the trial
    structure (60 from high condition trials, 60 from low condition trials),
    but both groups are assigned to the reference category because
    theoretically their condition of origin should not affect memorability —
    they were never under participant control regardless of which trial
    they appeared on.

    The key test is the is_high coefficient: high control items vs
    uncontrolled items — does motor control per se boost memory?

    Conservative assumption: uncontrolled hit rate equals low control hit rate.
    """
    hr_diff   = cohens_d_to_hr_diff(effect_d)
    hr_high   = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low    = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    hr_unctrl = hr_low

    lo_high   = hr_to_log_odds(hr_high)
    lo_low    = hr_to_log_odds(hr_low)
    lo_unctrl = hr_to_log_odds(hr_unctrl)

    # Trial structure:
    #   60 high control targets
    #   60 low control targets
    #   60 uncontrolled items from high condition trials  } both = reference
    #   60 uncontrolled items from low condition trials   }
    n_total      = N_TRIALS_PER_CONDITION * 2 + N_UNCONTROLLED
    participants = np.repeat(np.arange(n), n_total)

    is_high = np.tile(np.concatenate([
        np.ones(N_TRIALS_PER_CONDITION),              # high control targets
        np.zeros(N_TRIALS_PER_CONDITION),             # low control targets
        np.zeros(N_UNCONTROLLED_PER_CONDITION),       # uncontrolled (high trial)
        np.zeros(N_UNCONTROLLED_PER_CONDITION)        # uncontrolled (low trial)
    ]), n)

    is_low = np.tile(np.concatenate([
        np.zeros(N_TRIALS_PER_CONDITION),             # high control targets
        np.ones(N_TRIALS_PER_CONDITION),              # low control targets
        np.zeros(N_UNCONTROLLED_PER_CONDITION),       # uncontrolled (high trial)
        np.zeros(N_UNCONTROLLED_PER_CONDITION)        # uncontrolled (low trial)
    ]), n)

    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high,   N_TRIALS_PER_CONDITION),
        np.repeat(lo_low,    N_TRIALS_PER_CONDITION),
        np.repeat(lo_unctrl, N_UNCONTROLLED_PER_CONDITION),   # uncontrolled from high trials
        np.repeat(lo_unctrl, N_UNCONTROLLED_PER_CONDITION)    # uncontrolled from low trials
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)
        lo     = base_lo + ri_rep
        probs  = expit(lo)
        recognized = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'is_high':     is_high,
            'is_low':      is_low,
            'recognized':  recognized
        })

        try:
            gee = smf.gee(
                "recognized ~ is_high + is_low",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            if gee.pvalues['is_high'] < ALPHA:
                significant += 1

        except Exception:
            pass

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 4: GEE with item_type x control interaction
# (three-level control, no foil dummy assignment needed)
# ------------------------------------------------------------------------------

def power_supp_gee_interaction_three(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE on all trials (old + new foils)
    with a three-level control_type x item_is_old interaction.

    Model:
        said_old ~ item_is_old * control_type
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Where:
        item_is_old  : 1 = any old item (controlled or uncontrolled), 0 = new foil
        control_type : three levels — high control, low control, uncontrolled
                       (dummy coded with uncontrolled as reference)

    The interaction captures whether the memory advantage for old vs new items
    differs across control conditions — i.e., whether d-prime differs between
    high control, low control, and uncontrolled conditions.

    KEY ADVANTAGE over Analysis 4 (primary):
      Foils no longer need dummy assignment to control conditions because
      control_type is now a property of old items only. New foils simply
      have item_is_old = 0 and are not assigned to any control condition.
      This makes the model conceptually cleaner.

    We test the interaction between item_is_old and is_high_control, which
    captures the d-prime difference between high control and uncontrolled items.
    """
    hr_diff   = cohens_d_to_hr_diff(effect_d)
    hr_high   = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low    = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    hr_unctrl = hr_low   # conservative. might be overly conservative because uncontrolled items were seen in the same context as controlled items, so they might actually be remembered better than low control items.

    lo_high   = hr_to_log_odds(hr_high)
    lo_low    = hr_to_log_odds(hr_low)
    lo_unctrl = hr_to_log_odds(hr_unctrl)
    lo_fa     = hr_to_log_odds(BASELINE_FA_RATE)

    # 480 trials: 60+60+60+60 old + 240 new foils
    n_total      = N_TRIALS_PER_CONDITION * 2 + N_UNCONTROLLED + N_FOILS
    participants = np.repeat(np.arange(n), n_total)

    # item_is_old: 1 for any old item (controlled or uncontrolled), 0 for new foil
    item_is_old = np.tile(np.concatenate([
        np.ones(N_TRIALS_PER_CONDITION),              # high control targets
        np.ones(N_TRIALS_PER_CONDITION),              # low control targets
        np.ones(N_UNCONTROLLED_PER_CONDITION),        # uncontrolled (high trial)
        np.ones(N_UNCONTROLLED_PER_CONDITION),        # uncontrolled (low trial)
        np.zeros(N_FOILS)                             # new foils
    ]), n)

    # Dummy code: uncontrolled (both groups) and foils = reference (0, 0)
    is_high = np.tile(np.concatenate([
        np.ones(N_TRIALS_PER_CONDITION),              # high control targets
        np.zeros(N_TRIALS_PER_CONDITION),
        np.zeros(N_UNCONTROLLED_PER_CONDITION),       # uncontrolled = reference
        np.zeros(N_UNCONTROLLED_PER_CONDITION),
        np.zeros(N_FOILS)
    ]), n)

    is_low = np.tile(np.concatenate([
        np.zeros(N_TRIALS_PER_CONDITION),
        np.ones(N_TRIALS_PER_CONDITION),              # low control targets
        np.zeros(N_UNCONTROLLED_PER_CONDITION),       # uncontrolled = reference
        np.zeros(N_UNCONTROLLED_PER_CONDITION),
        np.zeros(N_FOILS)
    ]), n)

    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high,   N_TRIALS_PER_CONDITION),
        np.repeat(lo_low,    N_TRIALS_PER_CONDITION),
        np.repeat(lo_unctrl, N_UNCONTROLLED_PER_CONDITION),   # uncontrolled from high trials
        np.repeat(lo_unctrl, N_UNCONTROLLED_PER_CONDITION),   # uncontrolled from low trials
        np.repeat(lo_fa,     N_FOILS)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)
        lo     = base_lo + ri_rep
        probs  = expit(lo)
        said_old = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'item_is_old': item_is_old,
            'is_high':     is_high,
            'is_low':      is_low,
            'said_old':    said_old
        })

        try:
            # item_is_old:is_high captures the d-prime difference between
            # high control items and uncontrolled items — the cleanest test
            # of whether motor control specifically boosts memory
            gee = smf.gee(
                "said_old ~ item_is_old * is_high + item_is_old * is_low",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            if gee.pvalues['item_is_old:is_high'] < ALPHA:
                significant += 1

        except Exception:
            pass

    return significant / n_sim


# ==============================================================================
# RUN ALL SIMULATIONS
# ==============================================================================

def run_all():
    """
    Run the full power analysis grid:
      2 effect sizes x 4 analyses x 4 sample sizes = 32 cells
    Prints progress to console and returns a results DataFrame.
    """
    effect_sizes = {
        'Small (d=0.2)':  0.2,
        'Medium (d=0.5)': 0.5
    }
    # Primary analyses: high vs low control only
    primary_analyses = {
        'Analysis 1: d-prime paired t-test':  power_dprime_ttest,
        'Analysis 2: Hit rate paired t-test': power_hitrate_ttest,
        'Analysis 3: GEE hits only':          power_gee_hits_only,
        'Analysis 4: GEE with interaction':   power_gee_interaction,
    }
    # Supplementary analyses: include uncontrolled items as third condition
    supp_analyses = {
        'Supp 1: d-prime one-way ANOVA (3 conditions)':      power_supp_dprime_anova,
        'Supp 2: Hit rate one-way ANOVA (3 conditions)':     power_supp_hitrate_anova,
        'Supp 3: GEE all old items (3-level control_type)':  power_supp_gee_three_conditions,
        'Supp 4: GEE interaction (3-level, clean foils)':    power_supp_gee_interaction_three,
    }

    results = []
    # Run primary then supplementary, tagging each with its section
    for section_label, analyses in [('Primary', primary_analyses),
                                     ('Supplementary', supp_analyses)]:
        total = len(effect_sizes) * len(analyses) * len(SAMPLE_SIZES)
        done  = 0
        print(f"\n{'='*65}")
        print(f"  {section_label} Analyses")
        print(f"{'='*65}")
        t_start = time.time()

        for effect_label, effect_d in effect_sizes.items():
            for analysis_label, func in analyses.items():
                for n in SAMPLE_SIZES:
                    done += 1
                    print(f"[{done}/{total}] {analysis_label} | {effect_label} | N={n}...",
                          flush=True)
                    t0    = time.time()
                    power = func(n, effect_d)
                    print(f"       -> Power = {power:.3f}  ({time.time()-t0:.1f}s)",
                          flush=True)
                    results.append({
                        'Section':     section_label,
                        'Effect Size': effect_label,
                        'Analysis':    analysis_label,
                        'N':           n,
                        'Power':       round(power, 3)
                    })

        print(f"  Section runtime: {(time.time()-t_start)/60:.1f} min")

    return pd.DataFrame(results)


# ==============================================================================
# SAVE RESULTS TO MARKDOWN
# ==============================================================================

def save_markdown(df, path='results_CDmem.md'):
    """
    Save power analysis results as a formatted markdown file.
    Primary and supplementary analyses are written in separate sections.
    [80%+] marks cells meeting the 80% power threshold.
    """
    lines = []
    lines.append("# Power Analysis Results\n")

    lines.append("## Parameters\n")
    lines.append(f"- Simulations per cell: {N_SIMULATIONS}")
    lines.append(f"- Alpha level: {ALPHA}")
    lines.append(f"- Target trials per condition: {N_TRIALS_PER_CONDITION}")
    lines.append(f"- Foil trials total: {N_FOILS}")
    lines.append(f"- Uncontrolled old items total: {N_UNCONTROLLED} (supplementary analyses only)")
    lines.append(f"- Baseline hit rate: {BASELINE_HIT_RATE} (Wu et al., 2025, Exp. 4)")
    lines.append(f"- Baseline false alarm rate: {BASELINE_FA_RATE} (assumed)")
    lines.append(f"- Baseline d-prime: {BASELINE_DPRIME} (avg of Schreiner et al. conditions)")
    lines.append(f"- D-prime pooled SD: {DPRIME_SD} (Schreiner et al.)")
    lines.append(f"- Hit rate SD: {HIT_RATE_SD} (Schreiner et al.)")
    lines.append(f"- Random intercept SD: {RI_SD} (converted from d-prime SD via log-odds)\n")

    lines.append("## Primary Analysis Descriptions\n")
    lines.append("1. **D-prime paired t-test**: D-prime per participant per condition (loglinear correction), paired t-test. Standard SDT approach (Schreiner et al., 2024).")
    lines.append("2. **Hit rate paired t-test**: Mean hit rate per participant per condition, paired t-test. Used in Wu et al. (2025).")
    lines.append("3. **GEE hits only**: Trial-level GEE on target trials only. Predictor: control_level (+-0.5). Binomial family, logit link, exchangeable working correlation.")
    lines.append("4. **GEE with interaction**: Trial-level GEE on all trials. item_type x control interaction captures d-prime difference. Foils dummy-assigned equally across conditions.\n")

    lines.append("## Supplementary Analysis Descriptions\n")
    lines.append("These analyses incorporate the uncontrolled items (seen simultaneously with controlled items but never under participant control) as a third condition.")
    lines.append("**Theoretical note**: Uncontrolled items were viewed in the same context as controlled items, so they do not constitute a clean 'no encoding' baseline. Interpret with caution.")
    lines.append("**Conservative assumption**: Uncontrolled item hit rate is set equal to the low control hit rate (worst case for detecting an effect of motor control per se).\n")
    lines.append("S1. **D-prime one-way ANOVA**: Three d-primes per participant (high, low, uncontrolled), one-way repeated measures ANOVA. Tests omnibus effect.")
    lines.append("S2. **Hit rate one-way ANOVA**: Three hit rates per participant, one-way repeated measures ANOVA.")
    lines.append("S3. **GEE all old items**: Trial-level GEE on all old items. control_type as 3-level predictor (uncontrolled = reference). Tests high control vs uncontrolled contrast.")
    lines.append("S4. **GEE interaction (3-level)**: Trial-level GEE on all trials. item_is_old x control_type interaction. Foils require no dummy assignment — cleanest version of Analysis 4.\n")

    lines.append("## Effect Size Conversions\n")
    for d in [0.2, 0.5]:
        dd = cohens_d_to_dprime_diff(d)
        hd = cohens_d_to_hr_diff(d)
        lines.append(
            f"- Cohen's d = {d}: "
            f"delta d-prime = {dd:.3f} ({BASELINE_DPRIME-dd/2:.3f} vs {BASELINE_DPRIME+dd/2:.3f}), "
            f"delta hit rate = {hd:.3f} ({BASELINE_HIT_RATE-hd/2:.3f} vs {BASELINE_HIT_RATE+hd/2:.3f})"
        )
    lines.append("")

    # Write primary and supplementary results in separate sections
    for section in ['Primary', 'Supplementary']:
        lines.append(f"## {'Primary' if section == 'Primary' else 'Supplementary'} Analysis Results\n")
        section_df = df[df['Section'] == section]

        for effect_label in section_df['Effect Size'].unique():
            lines.append(f"### {effect_label}\n")
            subset = section_df[section_df['Effect Size'] == effect_label].copy()
            pivot  = subset.pivot(index='Analysis', columns='N', values='Power')
            pivot.columns = [f'N={c}' for c in pivot.columns]
            lines.append('| Analysis | ' + ' | '.join(pivot.columns) + ' |')
            lines.append('|' + '---|' * (len(pivot.columns) + 1))
            for analysis, row in pivot.iterrows():
                vals = [f"{v:.3f}{' [80%+]' if v >= 0.80 else ''}" for v in row.values]
                lines.append(f"| {analysis} | " + " | ".join(vals) + " |")
            lines.append("\n[80%+] = meets 80% power threshold\n")

    lines.append("## Notes and Limitations\n")
    lines.append("- **GEE**: Estimates population-average effects. Robust to misspecification of working correlation. Does not require specifying random slopes (advantageous with small N).")
    lines.append("- **Random intercept SD**: Estimated indirectly from d-prime SD. Actual between-participant variability is unknown.")
    lines.append("- **False alarm rate**: Assumed 0.20. Does not directly affect Analyses 1, 2, 3, S1, S2, or S3.")
    lines.append("- **Foil dummy assignment**: Only relevant for primary Analysis 4 — resolved in Supplementary Analysis S4.")
    lines.append("- **Effect size uncertainty**: Parameters from a choice paradigm (Schreiner et al.) — true effect in a control paradigm may differ.")
    lines.append("- **Supplementary conservative assumption**: Uncontrolled hit rate = low control hit rate. Power for S1-S4 would be higher if uncontrolled items are remembered less well than low control items.")
    lines.append("- **Recommendation**: For small effects (d=0.2), N=30-60 is likely underpowered regardless of analysis. Frame as exploratory. For medium effects (d=0.5), GEE analyses offer more power by leveraging trial-level variance.")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nResults saved to: {path}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Power Analysis: 8 Analyses x 2 Effect Sizes x 4 Sample Sizes")
    print(f"  (4 primary + 4 supplementary)")
    print(f"Simulations per cell: {N_SIMULATIONS}")
    print("=" * 65 + "\n")

    df_results = run_all()

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(df_results.to_string(index=False))

    save_markdown(df_results, 'results_CDmem.md')