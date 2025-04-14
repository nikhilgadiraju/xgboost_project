"""
File: model_training.py
Author: Roy Huang
Email: ruoqiuhuang@gmial.com
Date: 2025-02-27
Description: Implements a grid search for hyperparameter tuning and model training 
on an XGBoost model with GPU support.
"""

import numpy as np
import itertools
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from config import *
from utils import save_trained_model, copy_and_update_params


def to_list(param):
    """
    Ensure that the parameter is a list.
    """
    if isinstance(param, list):
        return param
    else:
        return [param]
    

def generate_hyperparameter_combinations():
    """
    Generates all possible combinations of hyperparameters.
    """
    param_grid = {
        'learning_rate': to_list(LEARNING_RATE),
        'n_estimators': to_list(NUM_ROUND),

        'max_depth': to_list(MAX_DEPTH),
        'min_child_weight': to_list(MIN_CHILD_WEIGHT),
        'gamma': to_list(GAMMA),

        'reg_lambda': to_list(REG_LAMBDA),
        'reg_alpha': to_list(REG_ALPHA),

        'subsample': to_list(SUBSAMPLE),
        'colsample_bytree': to_list(COLSAMPLE_BYTREE),
        'colsample_bylevel': to_list(COLSAMPLE_BYLEVEL)
    }
    return [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]


def cross_validation_model(X_train, y_train, params, cv_folds=3): 
    """
    Performs k-fold cross-validation and return the mean accuracy.
    """
    # Convert the data into DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Update params for binary/multi-class classification
    updates = {
        'objective': 'reg:squarederror', # Regression objective
        'eval_metric':'rmse',           # RMSE for regression
        'tree_method': 'hist',
        'device': 'cuda'
    }
    params = copy_and_update_params(params, updates)

    # n_estimator issue
    cv_params = params.copy()
    n_estimators = cv_params.pop('n_estimators')

    # Perform k-fold cross-validation
    cv_results = xgb.cv(
        params=cv_params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        nfold=cv_folds,
        stratified=False,
        early_stopping_rounds=10,
        metrics='rmse',
        seed=42
    )

    # Extract the final results
    final_rmse = cv_results['test-rmse-mean'].iloc[-1]
    return final_rmse
    

def manual_grid_search(X_train, y_train, cv_folds=3):
    """
    Performs a manual grid search for hyperparameter tuning.
    """
    # Generate all possible hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations()

    best_score = float('inf') # Lower RMSE is better
    best_params = None

    for params in hyperparameter_combinations: 
        print(f"Evaluating parameters: {params}")

        # Perform k-fold cross-validation
        mean_rmse = cross_validation_model(X_train, y_train, params, cv_folds)
        print(f"Mean CV RMSE: {mean_rmse:.4f}")

        # Select best on the lowest RMSE
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    print("\n-------------------------------------")
    print("Best parameters found:", best_params)
    print("Best cross-validation score:", best_score)
    print("-------------------------------------")

    return best_params, best_score


def train_final_model(X_train, X_val, y_train, y_val, best_params):
    """
    Train the final regression model using the best hyperparameters.
    """
    print("\nTraining final regression model with best hyperparameters...")

    # Update the best_params for regression
    updates = {
        'objective': 'reg:squarederror',  # Regression objective
        'eval_metric': 'rmse',           # RMSE for regression
        'tree_method': 'hist',
        'early_stopping_rounds': 10,  # Early stopping based on validation RMSE
        'device': 'cuda',
    }
    best_params = copy_and_update_params(best_params, updates)

    # Define the model
    final_model = xgb.XGBRegressor(
        **best_params
    )

    # Fit the model with early stopping
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Validation set for early stopping
        verbose=True
    )

    # Get the best iteration
    best_iteration = final_model.get_booster().best_iteration

    # Retrain the model with the best iteration
    if best_iteration is not None:
        print(f"\nRetraining the model with the best iteration: {best_iteration}")
        final_model.set_params(n_estimators=best_iteration + 1)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

    # Return the trained model
    return final_model