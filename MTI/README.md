# Multi-Timescale Inference (MTI) Experiment

A PsychoPy-based experiment investigating agency as a multi-timescale inference process, examining how stimulus and choice history effects unfold across different temporal windows.

## Overview

This project tests whether agency reflects a multi-timescale inference process with history-dependent temporal integration. The experiment compares performance and subjective agency ratings between Easy (2-shape) and Complex (4-shape) conditions to test whether complexity lengthens the temporal integration window.

## Research Questions

1. **Two-process structure**: Do stimulus history and choice history produce different effects (repulsive vs. attractive)?
2. **Complexity-dependent integration**: Does the Complex condition show longer temporal integration windows than the Easy condition?
3. **Alignment of performance and subjective SoA**: Do subjective agency ratings show similar history effects as performance?

## Experimental Design

- **Block 1**: 2-shape trials (calibration + test)
- **Block 2**: 4-shape trials (calibration + test)
- **Calibration**: 3-up-1-down staircase to find 75% accuracy threshold
- **Test Phase**: 4 difficulty levels based on calibrated threshold

## Repository Structure

```
MTI/
├── MT Inference.py              # Main experiment script
├── MT_Inference_Analysis.py     # Analysis script for history effects
├── Motion_Library/              # Pre-recorded motion trajectories
│   ├── core_pool.npy
│   ├── core_pool_feats.npy
│   ├── core_pool_labels.npy
│   └── *.json                   # Cluster/scaling parameters
├── data/
│   └── subjects/                # Participant data files
└── Agency as multi-timescale inference process.txt  # Project documentation
```

## Requirements

- Python 3.8+
- PsychoPy
- NumPy, Pandas, Matplotlib, SciPy, Statsmodels

## Quick Start

### Running the Experiment

```bash
python MT\ Inference.py
```

### Running Analysis

```bash
python MT_Inference_Analysis.py
```

## Key Features

- **3-up-1-down Staircase**: Adaptive calibration to find 75% accuracy threshold
- **Universal Trajectory Set**: Deterministic trajectory selection for consistency
- **History Analysis**: Lag-by-lag regression and exponential decay modeling
- **Multi-timescale Integration**: Tests temporal integration windows across conditions

## Analysis

The analysis script (`MT_Inference_Analysis.py`) performs:

1. **Lag-by-lag logistic regression**: Estimates stimulus and choice history effects at varying lags
2. **Exponential decay modeling**: Fits temporal integration windows (parameter w)
3. **Complexity comparison**: Tests whether Complex condition shows longer integration windows
4. **Agency rating analysis**: Examines whether subjective ratings show similar history effects

## License

This project is part of academic research. Please contact the authors before using.
