import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.stats import norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import warnings
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# SIMULATION SETTINGS
# ==============================================================================
N_SIMULATIONS = 1000
SAMPLE_SIZES = [30, 40, 50, 60]
ALPHA = 0.05
N_TRIALS_PER_CONDITION = 60

# d' Parameters (from pilot data)
BASELINE_DPRIME = 0.74
DPRIME_SD = 0.75

# Hit/FA Rate Parameters (from pilot data)
BASELINE_HIT_RATE = 0.62
HIT_RATE_SD = 0.20
BASELINE_FA_RATE = 0.20

# Log-RT Parameters (from pilot data)
# Intercept matches pilot LMM fixed effect: -0.192
BASELINE_LOG_RT = -0.20
LOG_RT_SD = 0.20

# Detection Accuracy parameters (from pilot data)
DETECT_HIGH = 0.892
DETECT_LOW = 0.577

# Random intercept SD.
# NOTE: The value below was used in initial simulations but its exact pilot
# source is unconfirmed. It is on the log-odds scale and is most relevant
# for the binomial GLMM models (1B, 2D, 3F).
# ACTION REQUIRED: Re-extract this value from the pilot binomial GLMM via:
#   model = glmer("said_old_int ~ control_level_c * item_type_c + (1|participant)", ...)
#   model.fit(); print(model.ranef_var)
# The Std column for 'participant' is RI_SD. Update this value accordingly.
# For the Gaussian LMM models (1C, 2E, 3G), RI_SD should be extracted
# separately from the LMM random effects and will be on the log-RT scale
# (expected ~0.05-0.15 based on pilot intercept of -0.192).
RI_SD = 1.36

# Within-person correlation across conditions (standard assumption)
WITHIN_PERSON_CORR = 0.50

# Agency slopes for H3 (3F binomial model).
# Specified in log-odds units because agency_z is a continuous z-scored
# predictor. Approximate Cohen's d equivalents are provided via the
# conversion: d ≈ log-odds / 1.81 (Borenstein et al., 2009).
#   Small:  0.20 log-odds ≈ d = 0.11
#   Medium: 0.50 log-odds ≈ d = 0.28
# Note: these are smaller than the d=0.20/0.50 used for H1/H2 because
# agency is a within-trial continuous predictor, not a condition mean difference.
AGENCY_SLOPE_SMALL  = 0.20   # ≈ d = 0.11
AGENCY_SLOPE_MEDIUM = 0.50   # ≈ d = 0.28

# Conversion factor for log-odds to Cohen's d (Borenstein et al., 2009)
LOG_ODDS_TO_D = 1.81

def cohens_d_to_dprime_diff(cohens_d): return cohens_d * DPRIME_SD
def cohens_d_to_hr_diff(cohens_d): return cohens_d * HIT_RATE_SD
def hr_to_log_odds(hr):
    hr = np.clip(hr, 0.001, 0.999)
    return np.log(hr / (1 - hr))


# ==============================================================================
# 1A) 2x2 RM ANOVA on d'
# ==============================================================================
def calc_dprime(hit_rate, fa_rate):
    hr = np.clip(hit_rate, 0.01, 0.99)
    fa = np.clip(fa_rate, 0.01, 0.99)
    return norm.ppf(hr) - norm.ppf(fa)

def run_1a_dprime_anova(n, effect_d, n_sim=N_SIMULATIONS):
    dprime_diff = cohens_d_to_dprime_diff(effect_d)
    mean_high = BASELINE_DPRIME + dprime_diff / 2
    mean_low  = BASELINE_DPRIME - dprime_diff / 2

    # Decompose variance into between- and within-person components
    between_sd = DPRIME_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = DPRIME_SD * np.sqrt(1 - WITHIN_PERSON_CORR)

    n_total = n * 4
    participants   = np.repeat(np.arange(n), 4)
    item_types     = np.tile(['controlled', 'controlled', 'uncontrolled', 'uncontrolled'], n)
    control_levels = np.tile(['high', 'low', 'high', 'low'], n)

    significant = 0
    for _ in range(n_sim):
        person_means = np.random.normal(0, between_sd, n)
        pm_rep = np.repeat(person_means, 4)

        # Control-level effect applied equally to BOTH item types.
        # This correctly reflects H1: high motor control leads to better
        # recognition than low motor control, regardless of item type.
        # The ANOVA tests the main effect of control level; any interaction
        # with item type is exploratory and not assumed here.
        # Order: ctrl_high, ctrl_low, unctrl_high, unctrl_low
        base_means = np.tile([mean_high, mean_low, mean_high, mean_low], n)

        dprimes = base_means + pm_rep + np.random.normal(0, within_sd, n_total)

        df = pd.DataFrame({
            'participant':   participants,
            'item_type':     item_types,
            'control_level': control_levels,
            'dprime':        dprimes
        })

        try:
            anova = AnovaRM(df, depvar='dprime', subject='participant',
                            within=['control_level', 'item_type']).fit()
            if anova.anova_table.loc['control_level', 'Pr > F'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 1B) Binomial GLMM (GEE) on ALL OLD ITEMS
# ==============================================================================
# NOTE: The planned analysis uses lme4-based GLMMs (via pymer4). GEE with an
# exchangeable correlation structure is used here as a computationally
# efficient approximation. GEE is expected to yield slightly conservative
# power estimates for binary outcomes at small N, making these results a
# defensible lower bound on true power.
def run_1b_binomial_glmm_all_old(n, effect_d, n_sim=N_SIMULATIONS):
    hr_diff = cohens_d_to_hr_diff(effect_d)
    hr_high = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low  = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high = hr_to_log_odds(hr_high)
    lo_low  = hr_to_log_odds(hr_low)
    lo_uncontrolled = hr_to_log_odds(BASELINE_HIT_RATE)

    # 4 cells: ctrl_high, ctrl_low, unctrl_high, unctrl_low
    n_total = N_TRIALS_PER_CONDITION * 4
    participants = np.repeat(np.arange(n), n_total)

    item_type_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION * 2),   # controlled
        np.repeat(-0.5, N_TRIALS_PER_CONDITION * 2)   # uncontrolled
    ]), n)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),    # ctrl_high
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),   # ctrl_low
        np.repeat(0.5, N_TRIALS_PER_CONDITION),    # unctrl_high
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)    # unctrl_low
    ]), n)

    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high, N_TRIALS_PER_CONDITION),
        np.repeat(lo_low, N_TRIALS_PER_CONDITION),
        np.repeat(lo_uncontrolled, N_TRIALS_PER_CONDITION * 2)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri    = np.repeat(np.random.normal(0, RI_SD, n), n_total)
        probs = expit(base_lo + ri)
        said_old = np.random.binomial(1, probs)

        df = pd.DataFrame({
            'participant':    participants,
            'item_type_c':    item_type_c,
            'control_level_c': control_level_c,
            'said_old':       said_old
        })
        try:
            gee = smf.gee(
                "said_old ~ control_level_c * item_type_c",
                groups="participant", data=df,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['control_level_c'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 1C) Gaussian LMM (GEE) on ALL OLD ITEMS (Reaction Time)
# ==============================================================================
# NOTE: GEE approximation used — see note on 1B above.
def run_1c_gaussian_lmm_all_old(n, effect_d, n_sim=N_SIMULATIONS):
    rt_diff = effect_d * LOG_RT_SD
    rt_high = BASELINE_LOG_RT - rt_diff / 2   # High control → faster RT
    rt_low  = BASELINE_LOG_RT + rt_diff / 2
    rt_uncontrolled = BASELINE_LOG_RT

    n_total = N_TRIALS_PER_CONDITION * 4
    participants = np.repeat(np.arange(n), n_total)

    item_type_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION * 2),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION * 2)
    ]), n)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)
    ]), n)

    base_rt = np.tile(np.concatenate([
        np.repeat(rt_high, N_TRIALS_PER_CONDITION),
        np.repeat(rt_low, N_TRIALS_PER_CONDITION),
        np.repeat(rt_uncontrolled, N_TRIALS_PER_CONDITION * 2)
    ]), n)

    between_sd = LOG_RT_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = LOG_RT_SD * np.sqrt(1 - WITHIN_PERSON_CORR)

    significant = 0
    for _ in range(n_sim):
        ri     = np.repeat(np.random.normal(0, between_sd, n), n_total)
        log_rt = base_rt + ri + np.random.normal(0, within_sd, n * n_total)

        df = pd.DataFrame({
            'participant':    participants,
            'item_type_c':    item_type_c,
            'control_level_c': control_level_c,
            'log_mem_rt':     log_rt
        })
        try:
            gee = smf.gee(
                "log_mem_rt ~ control_level_c * item_type_c",
                groups="participant", data=df,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['control_level_c'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 2D) Binomial GLMM (GEE) on OLD CONTROLLED ITEMS (Detection Accuracy → Memory)
# ==============================================================================
# NOTE: GEE approximation used — see note on 1B above.
def run_2d_binomial_glmm_detection(n, effect_d, n_sim=N_SIMULATIONS):
    hr_diff    = cohens_d_to_hr_diff(effect_d)
    true_slope = (hr_to_log_odds(BASELINE_HIT_RATE + hr_diff / 2) -
                  hr_to_log_odds(BASELINE_HIT_RATE - hr_diff / 2))
    intercept  = hr_to_log_odds(BASELINE_HIT_RATE)

    # 2 cells: high control, low control (controlled items only)
    n_total = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)
    ]), n)

    # Detection probabilities differ by condition (from pilot data)
    detect_probs = np.tile(np.concatenate([
        np.repeat(DETECT_HIGH, N_TRIALS_PER_CONDITION),
        np.repeat(DETECT_LOW, N_TRIALS_PER_CONDITION)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        detect_acc = np.random.binomial(1, detect_probs)
        detect_c   = detect_acc - 0.5   # contrast coded

        ri       = np.repeat(np.random.normal(0, RI_SD, n), n_total)
        lo       = intercept + ri + true_slope * detect_c
        said_old = np.random.binomial(1, expit(lo))

        df = pd.DataFrame({
            'participant':        participants,
            'detection_accuracy_c': detect_c,
            'control_level_c':    control_level_c,
            'said_old':           said_old
        })
        try:
            gee = smf.gee(
                "said_old ~ detection_accuracy_c * control_level_c",
                groups="participant", data=df,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['detection_accuracy_c'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 2E) Gaussian LMM (GEE) on OLD CONTROLLED ITEMS (Detection Accuracy → RT)
# ==============================================================================
# NOTE: GEE approximation used — see note on 1B above.
def run_2e_gaussian_lmm_detection(n, effect_d, n_sim=N_SIMULATIONS):
    # Negative slope: correctly detected items → faster (lower) RT
    rt_slope = -(effect_d * LOG_RT_SD)

    n_total = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)
    ]), n)

    detect_probs = np.tile(np.concatenate([
        np.repeat(DETECT_HIGH, N_TRIALS_PER_CONDITION),
        np.repeat(DETECT_LOW, N_TRIALS_PER_CONDITION)
    ]), n)

    between_sd = LOG_RT_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = LOG_RT_SD * np.sqrt(1 - WITHIN_PERSON_CORR)

    significant = 0
    for _ in range(n_sim):
        detect_acc = np.random.binomial(1, detect_probs)
        detect_c   = detect_acc - 0.5

        ri     = np.repeat(np.random.normal(0, between_sd, n), n_total)
        log_rt = BASELINE_LOG_RT + ri + rt_slope * detect_c + np.random.normal(0, within_sd, n * n_total)

        df = pd.DataFrame({
            'participant':        participants,
            'detection_accuracy_c': detect_c,
            'control_level_c':    control_level_c,
            'log_mem_rt':         log_rt
        })
        try:
            gee = smf.gee(
                "log_mem_rt ~ detection_accuracy_c * control_level_c",
                groups="participant", data=df,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['detection_accuracy_c'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 3F) Binomial GLMM (GEE) on OLD CONTROLLED ITEMS (Agency → Memory)
# ==============================================================================
# NOTE: GEE approximation used — see note on 1B above.
# Effect size is specified as a log-odds slope (not Cohen's d) because
# agency_z is a continuous within-participant z-scored predictor.
# See AGENCY_SLOPE_SMALL / AGENCY_SLOPE_MEDIUM definitions at top of script
# for approximate Cohen's d equivalents.
def run_3f_binomial_glmm_agency(n, effect_slope, n_sim=N_SIMULATIONS):
    intercept = hr_to_log_odds(BASELINE_HIT_RATE)

    n_total = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        # Agency ratings are z-scored within participant → simulate as N(0,1)
        agency_z = np.random.normal(0, 1, n * n_total)

        ri       = np.repeat(np.random.normal(0, RI_SD, n), n_total)
        lo       = intercept + ri + effect_slope * agency_z
        said_old = np.random.binomial(1, expit(lo))

        df = pd.DataFrame({
            'participant':    participants,
            'agency_z':       agency_z,
            'control_level_c': control_level_c,
            'said_old':       said_old
        })
        try:
            gee = smf.gee(
                "said_old ~ agency_z * control_level_c",
                groups="participant", data=df,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['agency_z'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# 3G) Gaussian LMM (GEE) on OLD CONTROLLED ITEMS (Agency → RT)
# ==============================================================================
# NOTE: GEE approximation used — see note on 1B above.
def run_3g_gaussian_lmm_agency(n, effect_d, n_sim=N_SIMULATIONS):
    # 1 SD increase in agency → faster (lower) RT by effect_d * LOG_RT_SD
    rt_slope = -(effect_d * LOG_RT_SD)

    n_total = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)

    control_level_c = np.tile(np.concatenate([
        np.repeat(0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION)
    ]), n)

    between_sd = LOG_RT_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = LOG_RT_SD * np.sqrt(1 - WITHIN_PERSON_CORR)

    significant = 0
    for _ in range(n_sim):
        agency_z = np.random.normal(0, 1, n * n_total)

        ri     = np.repeat(np.random.normal(0, between_sd, n), n_total)
        log_rt = BASELINE_LOG_RT + ri + rt_slope * agency_z + np.random.normal(0, within_sd, n * n_total)

        df = pd.DataFrame({
            'participant':    participants,
            'agency_z':       agency_z,
            'control_level_c': control_level_c,
            'log_mem_rt':     log_rt
        })
        try:
            gee = smf.gee(
                "log_mem_rt ~ agency_z * control_level_c",
                groups="participant", data=df,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['agency_z'] < ALPHA:
                significant += 1
        except:
            pass

    return significant / n_sim


# ==============================================================================
# EXECUTION AND REPORTING
# ==============================================================================
if __name__ == '__main__':
    print(f"Running power analysis matching the 7 analyses in CDmem_analyses_final.py")
    print(f"Simulations per cell: {N_SIMULATIONS}\n")

    results = []
    start_time = time.time()

    for n in SAMPLE_SIZES:
        print(f"--- Simulating N = {n} ---")

        for effect_d in [0.20, 0.50]:
            print(f"  Effect Size d = {effect_d:.2f}")

            p_1a = run_1a_dprime_anova(n, effect_d)
            p_1b = run_1b_binomial_glmm_all_old(n, effect_d)
            p_1c = run_1c_gaussian_lmm_all_old(n, effect_d)
            p_2d = run_2d_binomial_glmm_detection(n, effect_d)
            p_2e = run_2e_gaussian_lmm_detection(n, effect_d)

            # 3F uses a log-odds slope (not Cohen's d) — see parameter
            # definitions at top of script for rationale and d equivalents.
            if effect_d == 0.20:
                effect_slope    = AGENCY_SLOPE_SMALL
                agency_d_approx = round(AGENCY_SLOPE_SMALL / LOG_ODDS_TO_D, 2)   # ≈ 0.11
            else:
                effect_slope    = AGENCY_SLOPE_MEDIUM
                agency_d_approx = round(AGENCY_SLOPE_MEDIUM / LOG_ODDS_TO_D, 2)  # ≈ 0.28

            p_3f = run_3f_binomial_glmm_agency(n, effect_slope)
            p_3g = run_3g_gaussian_lmm_agency(n, effect_d)

            results.append({
                'N':                        n,
                # Effect size for H1 & H2 analyses (1A-2E): Cohen's d
                'Effect_Size_d_H1_H2':      effect_d,
                # Effect size for H3 3F (binomial): log-odds slope
                'Effect_Size_H3_logodds':   effect_slope,
                # Approximate Cohen's d equivalent for 3F (log-odds / 1.81)
                'Effect_Size_d_H3_approx':  agency_d_approx,
                '1A_ANOVA_dprime':          p_1a,
                '1B_GLMM_HitRate':          p_1b,
                '1C_LMM_RT':                p_1c,
                '2D_Detect_HitRate':        p_2d,
                '2E_Detect_RT':             p_2e,
                # 3F power is for the log-odds slope above, not for d_H1_H2
                '3F_Agency_HitRate':        p_3f,
                # 3G uses Cohen's d (same scale as H1/H2)
                '3G_Agency_RT':             p_3g
            })

    df_results = pd.DataFrame(results)

    with open('CDmem_power_report.md', 'w') as f:
        f.write("# CDmem Power Analysis Report (Matched to Final Analyses)\n\n")
        f.write(f"**Iterations per simulation:** {N_SIMULATIONS}\n")
        f.write(f"**Alpha level:** {ALPHA}\n\n")

        f.write("## Methods\n\n")
        f.write(
            "Power was estimated via Monte Carlo simulation (1,000 iterations per cell). "
            "Each of the seven planned analyses (1A–3G) was simulated separately. "
            "Baseline parameters (hit rates, FA rates, log-RT distributions, "
            "detection accuracy by condition) were derived from pilot data. "
            "The random intercept SD (RI_SD) used in binomial models is on the "
            "log-odds scale; **this value should be re-extracted from the pilot "
            "binomial GLMM random effects before finalising the pre-registration** "
            "(see inline code comment).\n\n"
        )

        f.write("## Effect Size Notes\n\n")
        f.write(
            "- **H1 & H2 analyses (1A–2E):** Effect sizes are Cohen's *d*, applied as "
            "mean differences scaled by the relevant outcome SD.\n"
        )
        f.write(
            "- **H3 analysis 3F (binomial):** Effect size is specified as a log-odds "
            "slope because agency_z is a continuous z-scored within-participant predictor. "
            "Approximate Cohen's *d* equivalents are provided via the conversion "
            "log-odds / 1.81 (Borenstein et al., 2009). These are smaller than the "
            "d = 0.20 / 0.50 used elsewhere because the effect operates at the "
            "trial level on a continuous predictor, not as a condition mean difference.\n"
        )
        f.write(
            "- **H3 analysis 3G (Gaussian):** Effect size is Cohen's *d*, "
            "consistent with H1/H2.\n\n"
        )

        f.write("## GEE Approximation Note\n\n")
        f.write(
            "The planned analyses use `lme4`-based mixed models (via pymer4). "
            "For computational efficiency, power simulations use Generalized Estimating "
            "Equations (GEE) with an exchangeable correlation structure as an approximation. "
            "GEE tends to yield slightly conservative power estimates for binary outcomes "
            "at small N; the results below should therefore be interpreted as a lower bound "
            "on the power of the planned mixed models.\n\n"
        )

        f.write("## ANOVA Simulation Note\n\n")
        f.write(
            "For analysis 1A, the control-level effect is applied equally to both "
            "controlled and uncontrolled items, consistent with H1 (a main effect of "
            "motor control on recognition). No interaction with item type is assumed; "
            "any such interaction is exploratory.\n\n"
        )

        f.write("## Power Estimates\n\n")
        f.write(
            "Columns: `Effect_Size_d_H1_H2` = Cohen's d for analyses 1A–2E and 3G. "
            "`Effect_Size_H3_logodds` = log-odds slope for 3F. "
            "`Effect_Size_d_H3_approx` = approximate Cohen's d equivalent for 3F.\n\n"
        )
        f.write(df_results.to_markdown(index=False))

    print(f"\nCompleted in {(time.time() - start_time)/60:.1f} minutes.")
    print("Results saved to CDmem_power_report.md")