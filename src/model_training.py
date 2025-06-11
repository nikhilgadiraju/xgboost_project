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
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from config import *
from utils import *

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
    Generates all possible combinations of hyperparameters for either binary classification
    or regression based on the BINARY flag.
    """
    # Define parameter grid with both shared and task-specific parameters
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
        'colsample_bylevel': to_list(COLSAMPLE_BYLEVEL),
        # Set appropriate objective based on task type
        'objective': ['binary:logistic'] if BINARY else ['reg:squarederror']
    }
    return [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

def cross_validation_model(X_train, y_train, params, cv_folds=3):
    """
    Performs k-fold cross-validation and returns the mean performance metric.
    For binary classification: Uses AUC and logloss
    For regression: Uses RMSE
    """
    # Convert the training data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Update parameters based on task type (binary classification or regression)
    updates = {
        'objective': 'binary:logistic' if BINARY else 'reg:squarederror',
        'eval_metric': ['logloss', 'auc'] if BINARY else 'rmse',
        'tree_method': 'hist',        # Use histogram-based algorithm
        'device': 'cuda'              # Enable GPU acceleration
    }
    params = copy_and_update_params(params, updates)

    # Handle n_estimators separately for cross-validation
    cv_params = params.copy()
    n_estimators = cv_params.pop('n_estimators')

    # Perform k-fold cross-validation with appropriate settings
    cv_results = xgb.cv(
        params=cv_params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        nfold=cv_folds,
        stratified=BINARY,          # Use stratified folds for binary classification
        early_stopping_rounds=10,
        metrics='auc' if BINARY else 'rmse',
        seed=42
    )

    # Extract and return the appropriate metric
    if BINARY:
        final_score = cv_results['test-auc-mean'].iloc[-1]
        return -final_score  # Negative because we want to maximize AUC
    else:
        final_rmse = cv_results['test-rmse-mean'].iloc[-1]
        return final_rmse

def manual_grid_search(X_train, y_train, cv_folds=3):
    """
    Performs a manual grid search for hyperparameter tuning with appropriate metrics
    based on the task type (binary classification or regression).
    """
    # Generate all possible hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations()
    best_score = float('inf')
    best_params = None

    # Evaluate each parameter combination
    for params in hyperparameter_combinations:
        print(f"Evaluating parameters: {params}")
        mean_score = cross_validation_model(X_train, y_train, params, cv_folds)
        
        # Print appropriate metric based on task type
        metric_name = "AUC" if BINARY else "RMSE"
        print(f"Mean CV {metric_name}: {abs(mean_score):.4f}")

        # Update best parameters if score improves
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    # Print final results
    print("\n-------------------------------------")
    print("Best parameters found:", best_params)
    print(f"Best cross-validation {'AUC' if BINARY else 'RMSE'}: {abs(best_score):.4f}")
    print("-------------------------------------")

    return best_params, best_score

def train_final_model(X_train, X_val, y_train, y_val, best_params):
    """
    Train the final model using the best hyperparameters, with appropriate settings
    for either binary classification or regression.
    """
    print(f"\nTraining final {'classification' if BINARY else 'regression'} model...")

    # Update parameters based on task type
    updates = {
        'objective': 'binary:logistic' if BINARY else 'reg:squarederror',
        'eval_metric': ['logloss', 'auc'] if BINARY else 'rmse',
        'tree_method': 'hist',
        'early_stopping_rounds': 10,
        'device': 'cuda'
    }
    best_params = copy_and_update_params(best_params, updates)

    # Initialize appropriate model type
    model_class = xgb.XGBClassifier if BINARY else xgb.XGBRegressor
    final_model = model_class(**best_params)

    # Train the model with early stopping
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Retrain with best iteration if available
    best_iteration = final_model.get_booster().best_iteration
    if best_iteration is not None:
        print(f"\nRetraining with best iteration: {best_iteration}")
        final_model.set_params(n_estimators=best_iteration + 1)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

    return final_model

def evaluate_model(model, X_test, y_test, key=None, condition_label=None):
    """
    Evaluate model performance and create appropriate visualizations based on task type.
    For binary classification: Creates ROC curve and returns accuracy/AUC
    For regression: Creates prediction vs actual plot and returns R²
    """
    # Convert test data to DMatrix format
    dtest = xgb.DMatrix(X_test, label=y_test)
    booster = model.get_booster()
    
    if BINARY:
        y_pred_proba = booster.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot ROC curve with condition label in title
        fig = plot_roc_curve(y_test, y_pred_proba, key=key, condition_label=condition_label)
        metric = (accuracy, auc)
    else:
        y_pred = booster.predict(dtest)
        r2 = r2_score(y_test, y_pred)
        
        # Plot regression predictions with condition label in title
        fig = plot_prediction_vs_actual(y_test, y_pred, key=key, condition_label=condition_label)
        metric = r2
    
    if key:
        plot_type = 'roc_curve' if BINARY else 'predictions'
        fig.savefig(os.path.join(PLOTS_FOLDER, f'{plot_type}_patient_{key}.png'))
        plt.close(fig)
    
    return metric