"""
File: config.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Configuration file containing hyperparameter settings and dataset paths 
for XGBoost model tuning.
"""

import os

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
# ---------------------------------------------------------------------

# color fundus photography (CFP) dataset                                
CSV_PATH = r"/home/s440308/Documents/19_csv_preprocessing/cfp_features_etdrs_mean_triton_fovea.csv"

# condition dataset (measurement or condition csv)
CONDITION_PATH = r"/home/s440308/Documents/21_statistical_analysis/measurement.csv"

# optical coherence tomography (OCT) dataset                        
# CSV_PATH = r"data\RetFound_LF_all_OCT_fully_labelled.csv"

# ---------------------------------------------------------------------
CSV_PATH = os.path.join(ROOT_DIR, CSV_PATH)                           #
CONDITION_PATH = os.path.join(ROOT_DIR, CONDITION_PATH)               #
# ---------------------------------------------------------------------

# Choose condition being fitted
CONDITION_CID = 3004410

# Machine region selection
MACHINE_REGION = "triton macula"

# Hyperparameter tuning phases
# Phase 1
LEARNING_RATE = 0.05  # 0.01 - 0.3
NUM_ROUND = 250 # 100 - 1000

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
