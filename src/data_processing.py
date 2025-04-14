"""
File: data_processing.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Module for loading datasets, selecting machine regions, and preparing data for XGBoost model tuning.
"""

import pandas as pd
import time 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import CSV_PATH, MACHINE_REGION

def load_dataset():
    """
    Load the dataset from the specified CSV file.
    """
    return pd.read_csv(CSV_PATH)

def select_machine_region(df):
    """
    Select the machine region from the dataset.
    """
    df_machine_region = df[df["machine_region"] == MACHINE_REGION]
    features_columns = df_machine_region.columns[df_machine_region.columns.str.startswith('feature_')]

    features = df_machine_region[features_columns]
    etdrs_thickness = df_machine_region["etdrs_thickness"]
    recommended_split = df_machine_region["recommended_split"]

    print(f"User selected: {MACHINE_REGION}\n")
    time.sleep(0.1)
    print(f"Features shape: {features.shape}")
    print(f"Health conditions shape: {etdrs_thickness.shape}")

    return features, etdrs_thickness, recommended_split

def prepare_data(features, etdrs_thickness, recommended_split): 
    """
    Encode labels and splits data into train/test sets.
    """
    # Ensure health_conditions is numerical (if not already)
    etdrs_thickness = pd.to_numeric(etdrs_thickness, errors='coerce')

    # split into train, val, test according to recommended split
    train_mask = recommended_split == "train"
    val_mask = recommended_split == "val"
    test_mask = recommended_split == "test"

    # split data
    X_train, y_train = features[train_mask], etdrs_thickness[train_mask]
    X_val, y_val = features[val_mask], etdrs_thickness[val_mask]
    X_test, y_test = features[test_mask], etdrs_thickness[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test