"""
File: config.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Configuration file containing hyperparameter settings and dataset paths 
for XGBoost model tuning.
"""

import os

# Binary classification for condition.csv dataset
BINARY = True

# Minimum required samples per class for binary classification
MIN_SAMPLES_PER_CLASS = 5

# ---------------------------------------------------------------------
# Get the root directory of the project, and go back one level        #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#
                                                                      #
# Parameters foler                                                    #
PARAMETERS_FOLDER = os.path.join(ROOT_DIR, "hyperparameters")         #
                                                                      #
# Models folder                                                       #
MODELS_FOLDER = os.path.join(ROOT_DIR, "saved_models")                #
                                                                      #
# Results folder                                                      #
RESULTS_FOLDER = os.path.join(ROOT_DIR, "results")                    #
                                                                      #
# Plots folder
PLOTS_FOLDER = os.path.join(ROOT_DIR, "binary_plots" if BINARY else "regression_plots")                        #
                                                                      #
# ---------------------------------------------------------------------

# color fundus photography (CFP) dataset                                
CSV_PATH = r"/home/s440308/Documents/19_csv_preprocessing/results_Triton_macula_renamed_good_quality_FINETUNED.csv"

# condition dataset (measurement or condition csv)
CONDITION_PATH = r"/home/s440308/Documents/21_statistical_analysis/measurement.csv" if not BINARY else r"/home/s440308/Documents/21_statistical_analysis/condition_occurrence.csv"

# ---------------------------------------------------------------------
CSV_PATH = os.path.join(ROOT_DIR, CSV_PATH)                           #
CONDITION_PATH = os.path.join(ROOT_DIR, CONDITION_PATH)               #
# ---------------------------------------------------------------------

# Hyperparameter tuning phases
# Phase 1
LEARNING_RATE = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]  # 0.01 - 0.3
NUM_ROUND = [100, 250, 500, 750, 1000] # 100 - 1000

# Phase 2
MAX_DEPTH = 4 # 3 - 10
MIN_CHILD_WEIGHT = 3 # 1 - 10
GAMMA = 0.0 # 0 - 5

# Phase 3
REG_LAMBDA = 1 # 1 - 10
REG_ALPHA = 0 # 0 - 10

# Phase 4
SUBSAMPLE = 1.0 # 0.5 - 1.0
COLSAMPLE_BYTREE = 0.888 # 0.5 - 1.0
COLSAMPLE_BYLEVEL = 0.888 # 0.5 - 1.0
