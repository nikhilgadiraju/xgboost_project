"""
File: data_processing.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Module for loading datasets, selecting machine regions, and preparing data for XGBoost model tuning.
"""

import pandas as pd
import numpy as np
import time 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import *

def load_dataset(csv_string):
    """
    Load the dataset from the specified CSV file.
    """
    return pd.read_csv(csv_string)

def convert_condition(string):
    """
    Convert a string representation of a condition into a more usable format.
    """
    string = string.split(',')[1][1:] if isinstance(string, str) and (',' in string) else string
    return string + ')' if isinstance(string, str) and string.count('(') > string.count(')') else string

def create_condition_dict(df):
    """
    Create a dictionary mapping measurement_concept_id to measurement_source_value.
    """
    csv_binary = "condition" if BINARY else "measurement"
    df.dropna(subset=[f'{csv_binary}_concept_id'], inplace=True)
    condition_dict = df.set_index(df[f'{csv_binary}_concept_id'].astype(int))[f'{csv_binary}_source_value'].apply(convert_condition).to_dict()
    return condition_dict

# TODO: Clean up later; consider creating a separate "condition_label_convert" function
def add_condition_column(df, condition_df, condition_cid):
    """
    Add a condition column to the provided dataframe
    """
    condition_dict = create_condition_dict(condition_df)
    condition_label = condition_dict[int(condition_cid)]

    if BINARY:
        filtered_cond_df = condition_df[condition_df["condition_concept_id"] == int(condition_cid)]
        value_dict = filtered_cond_df.set_index("person_id")['condition_source_value'].to_dict()

        column_label ="study_condition"
        df[column_label] = df["id"].map(value_dict)
        df[str(column_label)] = np.where(df[str(column_label)].notna(), 1, 0)

        print(f"Condition selected: {condition_label}")
    else:
        filtered_cond_df = condition_df[condition_df["measurement_concept_id"] == int(condition_cid)]
        value_dict = filtered_cond_df.set_index("person_id")['value_as_number'].to_dict()

        column_label ="study_condition"
        df[column_label] = df["id"].map(value_dict)
        df.dropna(subset=["study_condition"], inplace=True)

        print(f"Condition selected: {condition_label}")

    return df, condition_label


def prepare_feature_data(df, condition_label):
    """
    Select the machine region from the dataset.
    """
    # Get all feature columns
    features_columns = df.columns[df.columns.str.startswith('feature_')]
    
    # Extract features and health condition
    features = df[features_columns]
    health_condition = df["study_condition"]
    recommended_split = df["recommended_split"]

    print(f"\nFeatures shape: {features.shape}")
    print(f"Health conditions shape: {health_condition.shape}")

    return features, health_condition, recommended_split

def prepare_data(features, health_condition, recommended_split): 
    """
    Encode labels and splits data into train/test sets.
    """
    if BINARY:
        # Ensure binary labels are 0 or 1
        health_condition = (health_condition > 0).astype(int)
    else:
        # Original regression code
        health_condition = pd.to_numeric(health_condition, errors='coerce')

    # split into train, val, test according to recommended split
    train_mask = recommended_split == "train"
    val_mask = recommended_split == "val"
    test_mask = recommended_split == "test"

    # split data
    X_train, y_train = features[train_mask], health_condition[train_mask]
    X_val, y_val = features[val_mask], health_condition[val_mask]
    X_test, y_test = features[test_mask], health_condition[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test

