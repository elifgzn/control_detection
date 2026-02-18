#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timescale Inference Analysis Script
==========================================

This script analyzes data from the control detection task to test whether
agency reflects a multi-timescale inference process with history-dependent
temporal integration.

Key Analyses:
1. Descriptive statistics and data quality checks
2. Lag-by-lag logistic regression (stimulus and choice history effects)
3. Exponential decay model fitting (temporal integration window estimation)
4. Comparison of history effects between Easy (2-shape) and Complex (4-shape) conditions
5. Analysis of both performance (accuracy) and subjective agency ratings

Based on: "Agency as Multi-Timescale Inference: Serial Dependence and Temporal 
Integration in Control Detection"
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.special import expit  # logistic function
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent / "data" / "subjects"
OUTPUT_DIR = Path(__file__).parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis parameters
MAX_LAG = 5  # Maximum number of lags to consider for history effects
MIN_TRIALS_PER_SUBJECT = 50  # Minimum trials required per condition
ALPHA = 0.05  # Significance level

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_all_data():
    """Load all subject data files and combine into a single DataFrame."""
    
    # Try both naming conventions
    pattern1 = str(DATA_DIR / "CDT_staircase_2vs4_*.csv")
    pattern2 = str(DATA_DIR / "MTI_*.csv")
    
    all_files = glob.glob(pattern1) + glob.glob(pattern2)
    
    # Filter out kinematics files
    all_files = [f for f in all_files if 'kinematics' not in f.lower()]
    
    if not all_files:
        print(f"No data files found in {DATA_DIR}")
        return None
    
    print(f"Found {len(all_files)} data files")
    
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Extract participant ID from filename if not in data
            if 'participant' not in df.columns or df['participant'].isna().all():
                filename = os.path.basename(f)
                # Extract ID from filename
                parts = filename.replace('.csv', '').split('_')
                participant_id = parts[-1] if parts else 'unknown'
                df['participant'] = participant_id
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} total trials from {len(dfs)} files")
    
    return data


def preprocess_data(data):
    """Clean and preprocess the data for analysis."""
    
    df = data.copy()
    
    # Standardize column names
    column_mapping = {
        'complexity': 'n_shapes',
        'control_level': 'prop_used'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Convert complexity strings to numeric if needed
    if 'n_shapes' in df.columns:
        if df['n_shapes'].dtype == object:
            complexity_map = {'easy': 2, 'complex': 4, 'hard': 4}
            df['n_shapes'] = df['n_shapes'].map(lambda x: complexity_map.get(str(x).lower(), x))
        df['n_shapes'] = pd.to_numeric(df['n_shapes'], errors='coerce')
    
    # Create complexity factor
    df['complexity'] = df['n_shapes'].map({2: 'easy', 4: 'complex'})
    
    # Ensure accuracy is numeric
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    # Ensure agency_rating is numeric
    df['agency_rating'] = pd.to_numeric(df['agency_rating'], errors='coerce')
    
    # Ensure prop_used is numeric
    df['prop_used'] = pd.to_numeric(df['prop_used'], errors='coerce')
    
    # Filter to test phase only for main analyses
    if 'phase' in df.columns:
        df_test = df[df['phase'] == 'test'].copy()
    else:
        df_test = df.copy()
    
    # Remove trials with missing critical variables
    df_test = df_test.dropna(subset=['accuracy', 'prop_used', 'participant'])
    
    # Remove timeout trials
    if 'is_timeout' in df_test.columns:
        df_test = df_test[df_test['is_timeout'] != True]
    if 'resp_shape' in df_test.columns:
        df_test = df_test[df_test['resp_shape'] != 'timeout']
    
    # Sort by participant and trial number
    df_test = df_test.sort_values(['participant', 'trial_num']).reset_index(drop=True)
    
    print(f"After preprocessing: {len(df_test)} valid test trials")
    print(f"Participants: {df_test['participant'].nunique()}")
    if 'complexity' in df_test.columns:
        print(f"Trials per complexity: {df_test.groupby('complexity').size().to_dict()}")
    
    return df_test


def compute_history_variables(df, max_lag=MAX_LAG):
    """
    Compute stimulus history and choice history variables.
    
    Stimulus history: Previous control levels (prop_used)
    Choice history: Previous accuracy (correct=1, incorrect=0) or agency ratings
    """
    
    df = df.copy()
    
    # Create lagged variables for each participant and complexity condition
    for lag in range(1, max_lag + 1):
        # Stimulus history: previous control level (centered)
        df[f'stim_lag{lag}'] = df.groupby(['participant', 'complexity'])['prop_used'].shift(lag)
        
        # Choice history: previous accuracy
        df[f'choice_lag{lag}'] = df.groupby(['participant', 'complexity'])['accuracy'].shift(lag)
        
        # Previous agency rating (if available)
        if 'agency_rating' in df.columns:
            df[f'agency_lag{lag}'] = df.groupby(['participant', 'complexity'])['agency_rating'].shift(lag)
    
    # Center stimulus history around mean
    mean_prop = df['prop_used'].mean()
    for lag in range(1, max_lag + 1):
        df[f'stim_lag{lag}_centered'] = df[f'stim_lag{lag}'] - mean_prop
    
    # Center choice history around 0.5 (chance)
    for lag in range(1, max_lag + 1):
        df[f'choice_lag{lag}_centered'] = df[f'choice_lag{lag}'] - 0.5
    
    return df


# ============================================================================
# ANALYSIS 1: DESCRIPTIVE STATISTICS
# ============================================================================

def compute_descriptive_stats(df):
    """Compute descriptive statistics for performance and agency ratings."""
    
    results = {}
    
    # Overall accuracy by complexity
    accuracy_by_complexity = df.groupby('complexity')['accuracy'].agg(['mean', 'std', 'count'])
    results['accuracy_by_complexity'] = accuracy_by_complexity
    
    # Accuracy by complexity and difficulty level (prop_used)
    if 'difficulty_level' in df.columns:
        accuracy_by_level = df.groupby(['complexity', 'difficulty_level'])['accuracy'].agg(['mean', 'std', 'count'])
        results['accuracy_by_level'] = accuracy_by_level
    
    # Agency ratings by complexity
    if 'agency_rating' in df.columns and df['agency_rating'].notna().any():
        agency_by_complexity = df.groupby('complexity')['agency_rating'].agg(['mean', 'std', 'count'])
        results['agency_by_complexity'] = agency_by_complexity
        
        # Agency by accuracy (correct vs incorrect)
        agency_by_accuracy = df.groupby(['complexity', 'accuracy'])['agency_rating'].agg(['mean', 'std', 'count'])
        results['agency_by_accuracy'] = agency_by_accuracy
    
    # Psychometric function: accuracy as function of control level
    prop_bins = pd.cut(df['prop_used'], bins=10)
    psychometric = df.groupby([prop_bins, 'complexity'])['accuracy'].agg(['mean', 'std', 'count'])
    results['psychometric'] = psychometric
    
    return results


def plot_descriptive_stats(df, output_dir):
    """Create plots for descriptive statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Psychometric function by complexity
    ax = axes[0, 0]
    for complexity in df['complexity'].unique():
        subset = df[df['complexity'] == complexity]
        # Bin prop_used and compute mean accuracy
        bins = np.linspace(subset['prop_used'].min(), subset['prop_used'].max(), 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        subset['prop_bin'] = pd.cut(subset['prop_used'], bins=bins, labels=bin_centers)
        psychometric = subset.groupby('prop_bin')['accuracy'].agg(['mean', 'sem']).reset_index()
        psychometric['prop_bin'] = psychometric['prop_bin'].astype(float)
        ax.errorbar(psychometric['prop_bin'], psychometric['mean'], 
                   yerr=psychometric['sem'], label=f'{complexity}', marker='o', capsize=3)
    ax.set_xlabel('Control Level (prop self-motion)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Psychometric Function by Complexity')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Accuracy by complexity (bar plot)
    ax = axes[0, 1]
    accuracy_data = df.groupby('complexity')['accuracy'].agg(['mean', 'sem']).reset_index()
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax.bar(accuracy_data['complexity'], accuracy_data['mean'], 
                  yerr=accuracy_data['sem'], capsize=5, color=colors)
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy by Complexity')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim([0, 1])
    
    # 3. Agency ratings by complexity and accuracy
    ax = axes[1, 0]
    if 'agency_rating' in df.columns and df['agency_rating'].notna().any():
        agency_data = df.groupby(['complexity', 'accuracy'])['agency_rating'].agg(['mean', 'sem']).reset_index()
        agency_data['condition'] = agency_data['complexity'] + '_' + agency_data['accuracy'].map({0: 'incorrect', 1: 'correct'})
        sns.barplot(data=df, x='complexity', y='agency_rating', hue='accuracy', 
                   ax=ax, errorbar='se')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Agency Rating (1-7)')
        ax.set_title('Agency Ratings by Complexity and Accuracy')
        ax.legend(title='Correct')
    else:
        ax.text(0.5, 0.5, 'No agency rating data available', 
               transform=ax.transAxes, ha='center')
    
    # 4. RT distribution by complexity
    ax = axes[1, 1]
    if 'rt_choice' in df.columns and df['rt_choice'].notna().any():
        df_rt = df[df['rt_choice'] > 0]  # Filter valid RTs
        for complexity in df['complexity'].unique():
            subset = df_rt[df_rt['complexity'] == complexity]['rt_choice']
            ax.hist(subset, bins=30, alpha=0.5, label=complexity, density=True)
        ax.set_xlabel('Reaction Time (s)')
        ax.set_ylabel('Density')
        ax.set_title('RT Distribution by Complexity')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No RT data available', 
               transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'descriptive_statistics.png', dpi=150)
    plt.close()
    
    print("Saved: descriptive_statistics.png")


# ============================================================================
# ANALYSIS 2: LAG-BY-LAG LOGISTIC REGRESSION
# ============================================================================

def run_lag_by_lag_regression(df, dependent_var='accuracy', max_lag=MAX_LAG):
    """
    Run lag-by-lag logistic regression to estimate stimulus and choice history effects.
    
    For each lag, fit a model:
    accuracy_t ~ current_prop + stim_lag_k + choice_lag_k
    
    This dissociates stimulus history from choice history effects.
    """
    
    results = {'easy': {}, 'complex': {}}
    
    for complexity in ['easy', 'complex']:
        df_cond = df[df['complexity'] == complexity].copy()
        
        if len(df_cond) < MIN_TRIALS_PER_SUBJECT:
            print(f"Insufficient data for {complexity} condition")
            continue
        
        results[complexity] = {
            'lag': [],
            'stim_coef': [], 'stim_se': [], 'stim_pval': [],
            'choice_coef': [], 'choice_se': [], 'choice_pval': [],
            'n_obs': []
        }
        
        for lag in range(1, max_lag + 1):
            # Prepare data
            stim_col = f'stim_lag{lag}_centered'
            choice_col = f'choice_lag{lag}_centered'
            
            # Drop rows with missing lagged values
            df_lag = df_cond.dropna(subset=[stim_col, choice_col, dependent_var, 'prop_used'])
            
            if len(df_lag) < 30:
                continue
            
            try:
                # Fit logistic regression
                if dependent_var == 'accuracy':
                    # Binary outcome: use logistic regression
                    X = sm.add_constant(df_lag[['prop_used', stim_col, choice_col]])
                    y = df_lag[dependent_var]
                    model = sm.Logit(y, X).fit(disp=0)
                else:
                    # Continuous outcome (agency rating): use OLS
                    formula = f"{dependent_var} ~ prop_used + {stim_col} + {choice_col}"
                    model = smf.ols(formula, data=df_lag).fit()
                
                results[complexity]['lag'].append(lag)
                results[complexity]['stim_coef'].append(model.params[stim_col])
                results[complexity]['stim_se'].append(model.bse[stim_col])
                results[complexity]['stim_pval'].append(model.pvalues[stim_col])
                results[complexity]['choice_coef'].append(model.params[choice_col])
                results[complexity]['choice_se'].append(model.bse[choice_col])
                results[complexity]['choice_pval'].append(model.pvalues[choice_col])
                results[complexity]['n_obs'].append(len(df_lag))
                
            except Exception as e:
                print(f"Error fitting model for {complexity}, lag {lag}: {e}")
                continue
    
    return results


def plot_lag_by_lag_results(results, output_dir, title_suffix=''):
    """Plot lag-by-lag regression coefficients for stimulus and choice history."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'easy': '#1f77b4', 'complex': '#ff7f0e'}
    markers = {'easy': 'o', 'complex': 's'}
    
    # Plot stimulus history coefficients
    ax = axes[0]
    for complexity in ['easy', 'complex']:
        if complexity not in results or not results[complexity]['lag']:
            continue
        res = results[complexity]
        lags = res['lag']
        coefs = res['stim_coef']
        ses = res['stim_se']
        
        ax.errorbar(lags, coefs, yerr=[1.96*se for se in ses], 
                   label=f'{complexity.capitalize()}', 
                   marker=markers[complexity], color=colors[complexity],
                   capsize=3, markersize=8, linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (trials back)')
    ax.set_ylabel('Coefficient (log-odds)')
    ax.set_title(f'Stimulus History Effects{title_suffix}')
    ax.legend()
    ax.set_xticks(range(1, MAX_LAG + 1))
    
    # Plot choice history coefficients
    ax = axes[1]
    for complexity in ['easy', 'complex']:
        if complexity not in results or not results[complexity]['lag']:
            continue
        res = results[complexity]
        lags = res['lag']
        coefs = res['choice_coef']
        ses = res['choice_se']
        
        ax.errorbar(lags, coefs, yerr=[1.96*se for se in ses], 
                   label=f'{complexity.capitalize()}', 
                   marker=markers[complexity], color=colors[complexity],
                   capsize=3, markersize=8, linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (trials back)')
    ax.set_ylabel('Coefficient (log-odds)')
    ax.set_title(f'Choice History Effects{title_suffix}')
    ax.legend()
    ax.set_xticks(range(1, MAX_LAG + 1))
    
    plt.tight_layout()
    filename = f'lag_by_lag_regression{title_suffix.replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    
    print(f"Saved: {filename}")


# ============================================================================
# ANALYSIS 3: EXPONENTIAL DECAY MODEL
# ============================================================================

def exponential_decay(lag, A, w):
    """
    Exponential decay function: f(lag) = A * exp(-lag / w)
    
    Parameters:
    - A: Amplitude (effect size at lag 0)
    - w: Decay constant (larger w = longer temporal integration window)
    """
    return A * np.exp(-lag / w)


def fit_exponential_decay(lags, coefficients, weights=None):
    """
    Fit exponential decay model to lag-by-lag coefficients.
    
    Returns: A (amplitude), w (decay constant), r_squared
    """
    lags = np.array(lags)
    coefficients = np.array(coefficients)
    
    if len(lags) < 3:
        return None, None, None
    
    try:
        # Initial guesses
        A0 = coefficients[0] if coefficients[0] != 0 else 0.1
        w0 = 2.0  # Initial guess for decay constant
        
        # Bounds: A can be positive or negative, w must be positive
        bounds = ([-np.inf, 0.1], [np.inf, 20.0])
        
        if weights is not None:
            popt, pcov = curve_fit(exponential_decay, lags, coefficients, 
                                  p0=[A0, w0], bounds=bounds, sigma=weights, maxfev=5000)
        else:
            popt, pcov = curve_fit(exponential_decay, lags, coefficients, 
                                  p0=[A0, w0], bounds=bounds, maxfev=5000)
        
        A, w = popt
        
        # Compute R-squared
        y_pred = exponential_decay(lags, A, w)
        ss_res = np.sum((coefficients - y_pred) ** 2)
        ss_tot = np.sum((coefficients - np.mean(coefficients)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return A, w, r_squared
    
    except Exception as e:
        print(f"Error fitting exponential decay: {e}")
        return None, None, None


def run_exponential_decay_analysis(lag_results):
    """
    Fit exponential decay models to lag-by-lag regression coefficients.
    
    Compares decay parameters between Easy and Complex conditions.
    """
    
    decay_results = {}
    
    for complexity in ['easy', 'complex']:
        if complexity not in lag_results or not lag_results[complexity]['lag']:
            continue
        
        res = lag_results[complexity]
        lags = res['lag']
        
        # Fit decay to stimulus history coefficients
        stim_A, stim_w, stim_r2 = fit_exponential_decay(
            lags, res['stim_coef'], weights=res['stim_se']
        )
        
        # Fit decay to choice history coefficients
        choice_A, choice_w, choice_r2 = fit_exponential_decay(
            lags, res['choice_coef'], weights=res['choice_se']
        )
        
        decay_results[complexity] = {
            'stim_A': stim_A, 'stim_w': stim_w, 'stim_r2': stim_r2,
            'choice_A': choice_A, 'choice_w': choice_w, 'choice_r2': choice_r2
        }
    
    return decay_results


def plot_exponential_decay_fit(lag_results, decay_results, output_dir):
    """Plot lag-by-lag coefficients with exponential decay fits."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'easy': '#1f77b4', 'complex': '#ff7f0e'}
    
    # Plot stimulus history with decay fit
    ax = axes[0]
    lag_smooth = np.linspace(1, MAX_LAG, 100)
    
    for complexity in ['easy', 'complex']:
        if complexity not in lag_results or not lag_results[complexity]['lag']:
            continue
        
        res = lag_results[complexity]
        lags = res['lag']
        coefs = res['stim_coef']
        ses = res['stim_se']
        
        # Plot data points
        ax.errorbar(lags, coefs, yerr=[1.96*se for se in ses], 
                   label=f'{complexity.capitalize()} (data)', 
                   marker='o', color=colors[complexity],
                   capsize=3, markersize=8, linestyle='none')
        
        # Plot decay fit
        if complexity in decay_results and decay_results[complexity]['stim_w'] is not None:
            A = decay_results[complexity]['stim_A']
            w = decay_results[complexity]['stim_w']
            r2 = decay_results[complexity]['stim_r2']
            
            y_fit = exponential_decay(lag_smooth, A, w)
            ax.plot(lag_smooth, y_fit, color=colors[complexity], linestyle='--',
                   label=f'{complexity.capitalize()} fit (w={w:.2f}, R²={r2:.2f})')
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag (trials back)')
    ax.set_ylabel('Coefficient (log-odds)')
    ax.set_title('Stimulus History: Exponential Decay Fit')
    ax.legend()
    
    # Plot choice history with decay fit
    ax = axes[1]
    
    for complexity in ['easy', 'complex']:
        if complexity not in lag_results or not lag_results[complexity]['lag']:
            continue
        
        res = lag_results[complexity]
        lags = res['lag']
        coefs = res['choice_coef']
        ses = res['choice_se']
        
        # Plot data points
        ax.errorbar(lags, coefs, yerr=[1.96*se for se in ses], 
                   label=f'{complexity.capitalize()} (data)', 
                   marker='o', color=colors[complexity],
                   capsize=3, markersize=8, linestyle='none')
        
        # Plot decay fit
        if complexity in decay_results and decay_results[complexity]['choice_w'] is not None:
            A = decay_results[complexity]['choice_A']
            w = decay_results[complexity]['choice_w']
            r2 = decay_results[complexity]['choice_r2']
            
            y_fit = exponential_decay(lag_smooth, A, w)
            ax.plot(lag_smooth, y_fit, color=colors[complexity], linestyle='--',
                   label=f'{complexity.capitalize()} fit (w={w:.2f}, R²={r2:.2f})')
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag (trials back)')
    ax.set_ylabel('Coefficient (log-odds)')
    ax.set_title('Choice History: Exponential Decay Fit')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exponential_decay_fit.png', dpi=150)
    plt.close()
    
    print("Saved: exponential_decay_fit.png")


# ============================================================================
# ANALYSIS 4: MIXED-EFFECTS MODELS
# ============================================================================

def run_mixed_effects_analysis(df, max_lag=3):
    """
    Run mixed-effects logistic regression with participant as random effect.
    
    Tests interaction between complexity and history effects.
    """
    
    # Prepare data with lagged variables
    df_model = df.dropna(subset=['stim_lag1_centered', 'choice_lag1_centered', 'accuracy'])
    
    # Create complexity dummy (easy=0, complex=1)
    df_model['is_complex'] = (df_model['complexity'] == 'complex').astype(int)
    
    results = {}
    
    # Model 1: Main effects only
    try:
        formula1 = "accuracy ~ prop_used + stim_lag1_centered + choice_lag1_centered + is_complex"
        model1 = smf.mixedlm(formula1, df_model, groups=df_model['participant']).fit()
        results['main_effects'] = model1
    except Exception as e:
        print(f"Error fitting main effects model: {e}")
    
    # Model 2: Interactions with complexity
    try:
        formula2 = """accuracy ~ prop_used + stim_lag1_centered * is_complex + 
                      choice_lag1_centered * is_complex"""
        model2 = smf.mixedlm(formula2, df_model, groups=df_model['participant']).fit()
        results['interaction'] = model2
    except Exception as e:
        print(f"Error fitting interaction model: {e}")
    
    return results


# ============================================================================
# ANALYSIS 5: AGENCY RATING ANALYSIS
# ============================================================================

def run_agency_rating_analysis(df, max_lag=MAX_LAG):
    """
    Analyze agency ratings as function of current and past trial features.
    
    Tests whether subjective agency shows similar history effects as performance.
    """
    
    if 'agency_rating' not in df.columns or df['agency_rating'].isna().all():
        print("No agency rating data available")
        return None
    
    # Filter to trials with valid agency ratings
    df_agency = df.dropna(subset=['agency_rating']).copy()
    
    results = {}
    
    # 1. Agency as function of current accuracy and control level
    for complexity in ['easy', 'complex']:
        df_cond = df_agency[df_agency['complexity'] == complexity]
        
        if len(df_cond) < 30:
            continue
        
        try:
            formula = "agency_rating ~ accuracy + prop_used + accuracy:prop_used"
            model = smf.ols(formula, data=df_cond).fit()
            results[f'{complexity}_agency_model'] = model
        except Exception as e:
            print(f"Error fitting agency model for {complexity}: {e}")
    
    # 2. Lag-by-lag regression for agency ratings
    agency_lag_results = run_lag_by_lag_regression(df_agency, dependent_var='agency_rating', max_lag=max_lag)
    results['lag_by_lag'] = agency_lag_results
    
    return results


def plot_agency_analysis(df, results, output_dir):
    """Plot agency rating analysis results."""
    
    if 'agency_rating' not in df.columns or df['agency_rating'].isna().all():
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Agency by accuracy and complexity
    ax = axes[0, 0]
    sns.barplot(data=df, x='complexity', y='agency_rating', hue='accuracy', 
               ax=ax, errorbar='se')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Agency Rating (1-7)')
    ax.set_title('Agency Ratings by Accuracy and Complexity')
    ax.legend(title='Correct')
    
    # 2. Agency by control level (prop_used)
    ax = axes[0, 1]
    for complexity in df['complexity'].unique():
        subset = df[df['complexity'] == complexity].dropna(subset=['agency_rating'])
        # Bin prop_used
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        subset['prop_bin'] = pd.cut(subset['prop_used'], bins=bins, labels=bin_centers)
        agency_by_prop = subset.groupby('prop_bin')['agency_rating'].agg(['mean', 'sem']).reset_index()
        agency_by_prop['prop_bin'] = agency_by_prop['prop_bin'].astype(float)
        ax.errorbar(agency_by_prop['prop_bin'], agency_by_prop['mean'], 
                   yerr=agency_by_prop['sem'], label=complexity, marker='o', capsize=3)
    ax.set_xlabel('Control Level (prop self-motion)')
    ax.set_ylabel('Agency Rating (1-7)')
    ax.set_title('Agency Ratings by Control Level')
    ax.legend()
    
    # 3. Lag-by-lag results for agency ratings (stimulus history)
    ax = axes[1, 0]
    if results and 'lag_by_lag' in results:
        lag_results = results['lag_by_lag']
        colors = {'easy': '#1f77b4', 'complex': '#ff7f0e'}
        for complexity in ['easy', 'complex']:
            if complexity in lag_results and lag_results[complexity]['lag']:
                res = lag_results[complexity]
                ax.errorbar(res['lag'], res['stim_coef'], 
                           yerr=[1.96*se for se in res['stim_se']], 
                           label=f'{complexity.capitalize()}', 
                           marker='o', color=colors[complexity], capsize=3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag (trials back)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Stimulus History Effects on Agency Ratings')
        ax.legend()
    
    # 4. Lag-by-lag results for agency ratings (choice history)
    ax = axes[1, 1]
    if results and 'lag_by_lag' in results:
        lag_results = results['lag_by_lag']
        for complexity in ['easy', 'complex']:
            if complexity in lag_results and lag_results[complexity]['lag']:
                res = lag_results[complexity]
                ax.errorbar(res['lag'], res['choice_coef'], 
                           yerr=[1.96*se for se in res['choice_se']], 
                           label=f'{complexity.capitalize()}', 
                           marker='o', color=colors[complexity], capsize=3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag (trials back)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Choice History Effects on Agency Ratings')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agency_rating_analysis.png', dpi=150)
    plt.close()
    
    print("Saved: agency_rating_analysis.png")


# ============================================================================
# ANALYSIS 6: HYPOTHESIS TESTING
# ============================================================================

def test_hypotheses(lag_results, decay_results, agency_results=None):
    """
    Test the main hypotheses from the theoretical framework.
    
    H1: Two-process structure
        - Stimulus history produces repulsive effects (negative coefficients)
        - Choice history produces attractive effects (positive coefficients)
    
    H2: Complexity-dependent gain and window
        - Complex condition shows stronger history effects at lag-1
        - Complex condition shows larger decay parameter w (longer integration window)
    """
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    # H1: Two-process structure
    print("\n--- H1: Two-Process Structure ---")
    
    for complexity in ['easy', 'complex']:
        if complexity not in lag_results or not lag_results[complexity]['lag']:
            continue
        
        res = lag_results[complexity]
        if len(res['stim_coef']) == 0:
            continue
            
        stim_lag1 = res['stim_coef'][0]
        choice_lag1 = res['choice_coef'][0]
        
        print(f"\n{complexity.capitalize()} condition:")
        print(f"  Stimulus history lag-1: β = {stim_lag1:.3f}, p = {res['stim_pval'][0]:.4f}")
        print(f"    Expected: Repulsive (negative) -> {'✓' if stim_lag1 < 0 else '✗'}")
        
        print(f"  Choice history lag-1: β = {choice_lag1:.3f}, p = {res['choice_pval'][0]:.4f}")
        print(f"    Expected: Attractive (positive) -> {'✓' if choice_lag1 > 0 else '✗'}")
    
    # H2: Complexity-dependent effects
    print("\n--- H2: Complexity-Dependent Integration Window ---")
    
    if 'easy' in decay_results and 'complex' in decay_results:
        easy = decay_results['easy']
        complex_ = decay_results['complex']
        
        print("\nStimulus history integration window (w):")
        if easy['stim_w'] is not None and complex_['stim_w'] is not None:
            print(f"  Easy: w = {easy['stim_w']:.2f}")
            print(f"  Complex: w = {complex_['stim_w']:.2f}")
            print(f"  Expected: Complex > Easy -> {'✓' if complex_['stim_w'] > easy['stim_w'] else '✗'}")
        
        print("\nChoice history integration window (w):")
        if easy['choice_w'] is not None and complex_['choice_w'] is not None:
            print(f"  Easy: w = {easy['choice_w']:.2f}")
            print(f"  Complex: w = {complex_['choice_w']:.2f}")
            print(f"  Expected: Complex > Easy -> {'✓' if complex_['choice_w'] > easy['choice_w'] else '✗'}")
        
        print("\nLag-1 effect magnitude:")
        if 'easy' in lag_results and 'complex' in lag_results:
            if lag_results['easy']['stim_coef'] and lag_results['complex']['stim_coef']:
                easy_stim = abs(lag_results['easy']['stim_coef'][0])
                complex_stim = abs(lag_results['complex']['stim_coef'][0])
                print(f"  Stimulus: Easy={easy_stim:.3f}, Complex={complex_stim:.3f}")
                print(f"  Expected: Complex > Easy -> {'✓' if complex_stim > easy_stim else '✗'}")
            
            if lag_results['easy']['choice_coef'] and lag_results['complex']['choice_coef']:
                easy_choice = abs(lag_results['easy']['choice_coef'][0])
                complex_choice = abs(lag_results['complex']['choice_coef'][0])
                print(f"  Choice: Easy={easy_choice:.3f}, Complex={complex_choice:.3f}")
                print(f"  Expected: Complex > Easy -> {'✓' if complex_choice > easy_choice else '✗'}")
    
    print("\n" + "="*60)


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(df, desc_stats, lag_results, decay_results, 
                           mixed_results, agency_results, output_dir):
    """Generate a comprehensive summary report."""
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("MULTI-TIMESCALE INFERENCE ANALYSIS REPORT")
    report_lines.append("="*70)
    
    # Data summary
    report_lines.append("\n--- DATA SUMMARY ---")
    report_lines.append(f"Total valid trials: {len(df)}")
    report_lines.append(f"Number of participants: {df['participant'].nunique()}")
    for complexity in df['complexity'].unique():
        n_trials = len(df[df['complexity'] == complexity])
        report_lines.append(f"  {complexity.capitalize()}: {n_trials} trials")
    
    # Descriptive statistics
    report_lines.append("\n--- DESCRIPTIVE STATISTICS ---")
    if 'accuracy_by_complexity' in desc_stats:
        report_lines.append("\nAccuracy by Complexity:")
        report_lines.append(desc_stats['accuracy_by_complexity'].to_string())
    
    if 'agency_by_complexity' in desc_stats:
        report_lines.append("\nAgency Ratings by Complexity:")
        report_lines.append(desc_stats['agency_by_complexity'].to_string())
    
    # Lag-by-lag results
    report_lines.append("\n--- LAG-BY-LAG REGRESSION RESULTS ---")
    for complexity in ['easy', 'complex']:
        if complexity not in lag_results or not lag_results[complexity]['lag']:
            continue
        
        res = lag_results[complexity]
        report_lines.append(f"\n{complexity.capitalize()} Condition:")
        
        for i, lag in enumerate(res['lag']):
            report_lines.append(f"  Lag {lag}:")
            report_lines.append(f"    Stimulus: β={res['stim_coef'][i]:.4f}, p={res['stim_pval'][i]:.4f}")
            report_lines.append(f"    Choice:   β={res['choice_coef'][i]:.4f}, p={res['choice_pval'][i]:.4f}")
    
    # Exponential decay results
    report_lines.append("\n--- EXPONENTIAL DECAY MODEL RESULTS ---")
    for complexity in ['easy', 'complex']:
        if complexity not in decay_results:
            continue
        
        res = decay_results[complexity]
        report_lines.append(f"\n{complexity.capitalize()} Condition:")
        report_lines.append(f"  Stimulus history: A={res['stim_A']:.4f}, w={res['stim_w']:.2f}, R²={res['stim_r2']:.3f}")
        report_lines.append(f"  Choice history: A={res['choice_A']:.4f}, w={res['choice_w']:.2f}, R²={res['choice_r2']:.3f}")
    
    # Write report
    report_text = '\n'.join(report_lines)
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\nSaved: analysis_report.txt")
    print(report_text)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """Run the complete analysis pipeline."""
    
    print("="*60)
    print("Multi-Timescale Inference Analysis")
    print("="*60)
    
    # 1. Load data
    print("\n--- Loading Data ---")
    raw_data = load_all_data()
    
    if raw_data is None or len(raw_data) == 0:
        print("No data available for analysis. Please check data directory.")
        return
    
    # 2. Preprocess data
    print("\n--- Preprocessing Data ---")
    df = preprocess_data(raw_data)
    
    if len(df) < MIN_TRIALS_PER_SUBJECT:
        print(f"Insufficient data for analysis (need at least {MIN_TRIALS_PER_SUBJECT} trials)")
        return
    
    # 3. Compute history variables
    print("\n--- Computing History Variables ---")
    df = compute_history_variables(df)
    
    # 4. Descriptive statistics
    print("\n--- Computing Descriptive Statistics ---")
    desc_stats = compute_descriptive_stats(df)
    plot_descriptive_stats(df, OUTPUT_DIR)
    
    # 5. Lag-by-lag regression (for accuracy)
    print("\n--- Running Lag-by-Lag Regression (Accuracy) ---")
    lag_results_accuracy = run_lag_by_lag_regression(df, dependent_var='accuracy')
    plot_lag_by_lag_results(lag_results_accuracy, OUTPUT_DIR, title_suffix=' (Accuracy)')
    
    # 6. Exponential decay model
    print("\n--- Fitting Exponential Decay Models ---")
    decay_results = run_exponential_decay_analysis(lag_results_accuracy)
    plot_exponential_decay_fit(lag_results_accuracy, decay_results, OUTPUT_DIR)
    
    # 7. Mixed-effects analysis
    print("\n--- Running Mixed-Effects Analysis ---")
    mixed_results = run_mixed_effects_analysis(df)
    
    # 8. Agency rating analysis
    print("\n--- Analyzing Agency Ratings ---")
    agency_results = run_agency_rating_analysis(df)
    if agency_results:
        plot_agency_analysis(df, agency_results, OUTPUT_DIR)
    
    # 9. Hypothesis testing
    test_hypotheses(lag_results_accuracy, decay_results, agency_results)
    
    # 10. Generate summary report
    print("\n--- Generating Summary Report ---")
    generate_summary_report(df, desc_stats, lag_results_accuracy, decay_results,
                           mixed_results, agency_results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to:", OUTPUT_DIR)
    print("="*60)
    
    return {
        'data': df,
        'descriptive_stats': desc_stats,
        'lag_results_accuracy': lag_results_accuracy,
        'decay_results': decay_results,
        'mixed_results': mixed_results,
        'agency_results': agency_results
    }


if __name__ == "__main__":
    results = main()
