# Project 8: Spam Detection with String Kernels

LMU Munich — Applied Machine Learning in Python
Authors: Florian Wienand, <partner>

## Overview
Compare two string kernels — a spectrum (n-gram) kernel and a Levenshtein
edit-distance kernel — inside kernel SVMs for SMS spam detection.
(TODO: 2–3 sentence summary once results exist.)

## Setup
    conda activate applied_ml
    conda install -c conda-forge numpy pandas matplotlib scikit-learn cvxopt rapidfuzz
`courselib/` is included in this repo, so no separate install is needed.

## How to run
    python run_experiments.py
(TODO: reproduces all figures and the results table end-to-end.)

## Structure
- courselib/  course library (vendored from the course repo)
- src/        project code: text cleaning, string kernels, evaluation
- notebooks/  exploration scratchpad (not the source of truth)
- data/       SMS Spam Collection dataset
- figures/    generated plots
- results/    generated metrics/tables
- report/     LaTeX report (written on Overleaf)