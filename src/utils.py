"""
File: utils.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Utility functions for user interaction
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 
import json
from config import *
import os
from sklearn.metrics import r2_score, roc_curve, auc


def pre_run_check():
    """
    Check if the parameters and models folders exist.
    Check if data path is valid.
    """
    if not os.path.exists(PARAMETERS_FOLDER):
        os.makedirs(PARAMETERS_FOLDER)

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(CSV_PATH):
        print(f"Data path '{CSV_PATH}' does not exist.")
        print("Exiting...")
        return False
    
    return True

def welcome_message():
    """
    Display a welcome message to the user.
    """
    print("-------------------------------------")
    print("Welcome to the XGBoost Tuning Script!")
    print("-------------------------------------")
    time.sleep(0.1)


def start_or_quit(): 
    """
    Prompt the user to start or quit the script.
    """
    print("\nLast chance to exit the program!")
    print("-------------------------------------")
    time.sleep(0.1)
    print("1: I am ready, just do it!")
    print("0: Exit")
    print("-------------------------------------")

    try: 
        start_int = int(input("Selction: "))
    except ValueError:
        print("Invalid input, exiting...")
        return False
    
    if start_int != 1:
        print("\nExiting...")
        return False

    print("\nOK, let's GO!\n")
    return True


def create_timestamp():
    """
    Create a timestamp for saving files.
    """
    return time.strftime("%Y%m%d-%H%M%S")


def copy_and_update_params(params, updates):
    """
    Copy the parameters and update them with new values.
    """
    params_copy = params.copy()
    params_copy.update(updates)
    return params_copy


def save_hyperparameters(best_params, best_score, timestamp):
    """
    Save the best hyperparameters and score to a JSON file.
    """
    filename = f"best_hyperparameters_{timestamp}.json"

    file_path = os.path.join(PARAMETERS_FOLDER, filename)

    updates = {"best_score": best_score}
    best_params = copy_and_update_params(best_params, updates)

    with open(file_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest hyperparameters saved to '{file_path}'.")

    return file_path


def save_trained_model(model, timestamp): 
    """
    Save the trained model to a file.
    """
    filename = f"xgboost_model_{timestamp}.json"

    file_path = os.path.join(MODELS_FOLDER, filename)

    model.save_model(file_path)

    print(f"\nTrained model saved to '{file_path}'.")

    return file_path


def plot_roc_curve(y_true, y_pred_proba, key=None, condition_label=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        key: Patient ID for filename
        condition_label: Condition description for plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # Add condition label to title if provided
    title = 'Receiver Operating Characteristic (ROC) Curve'
    if condition_label:
        title += f'\n{condition_label}'
    plt.title(title)
    
    plt.legend(loc="lower right")
    return fig


def plot_prediction_vs_actual(y_test, y_pred, key=None, condition_label=None):
    """
    Create scatter plot of predicted vs actual values.
    
    Args:
        y_test: True values
        y_pred: Predicted values
        key: Patient ID for filename
        condition_label: Condition description for plot title
    """
    r2 = r2_score(y_test, y_pred)
    
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add best fit and perfect prediction lines
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'k--', alpha=0.3, label='Perfect Prediction')
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    
    # Add condition label to title if provided
    title = f"Predicted vs Actual Values\nR² = {r2:.3f}"
    if condition_label:
        title += f'\n{condition_label}'
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def plot_r2_bar_chart(r2_df, condition_dict, cid=False, save_path=None):
    """
    Create a bar chart from the r2_df DataFrame and optionally save it to a file.
    The bar labels are set using the condition_dict.

    Args:
        r2_df (pd.DataFrame): DataFrame containing 'condition' and 'r2' columns.
        condition_dict (dict): Dictionary mapping condition keys to their descriptions.
        save_path (str, optional): File path to save the chart. If None, the chart is only displayed.
    """
    # Extract data for the bar chart
    conditions = r2_df['condition'].astype(int).astype(str) # First column for x-axis labels
    r2_values = r2_df['r2']                                 # Second column for bar heights

    # Map conditions to their descriptions using condition_dict
    condition_labels = [condition_dict.get(int(cond), cond) for cond in conditions] if not cid else conditions

    # Create the bar chart
    plt.figure(figsize=(15, 12))
    bars = plt.bar(conditions, r2_values, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('R² Value', fontsize=12)
    plt.title('R² Values by Condition', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels above positive bars and below negative bars
    for bar, label in zip(bars, condition_labels):
        height = bar.get_height()
        if height >= 0:
            # Positive bar: Place label above the bar
            y_position = height + 0.03
            va = 'bottom'
        else:
            # Negative bar: Place label below the bar
            y_position = height - 0.03
            va = 'top'

        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
            y_position,                         # Y-coordinate (above or below the bar)
            label,                              # Text to display (mapped condition label)
            ha='center',                        # Horizontal alignment
            va=va,                              # Vertical alignment
            fontsize=10,                        # Font size
            color='black',                      # Text color
            rotation=90                         # Rotate the label vertically
        )

    # Remove x-axis tick labels since they are now above/below the bars
    plt.xticks([])

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save the chart to the provided file path if specified
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Bar chart saved to {save_path}")


def plot_binary_metrics(metrics_df, condition_dict, cid=False, save_path=None):
    """
    Creates a grouped bar chart comparing accuracy and AUC scores across all conditions.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing columns ['condition', 'accuracy', 'auc']
        condition_dict (dict): Dictionary mapping condition IDs to their descriptions
        cid (bool): If True, use condition IDs as labels instead of descriptions
        save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    # Set style parameters
    plt.style.use('seaborn')
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(12, 6))
    
    # Extract data
    conditions = metrics_df['condition'].astype(str)
    accuracies = metrics_df['accuracy']
    aucs = metrics_df['auc']
    
    # Get x-axis positions
    x = np.arange(len(conditions))
    width = 0.35  # Width of bars
    
    # Create bars
    ax = plt.gca()
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                    color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, aucs, width, label='AUC', 
                    color='lightcoral', edgecolor='black')
    
    # Customize plot
    plt.ylabel('Score', fontsize=12)
    plt.title('Binary Classification Metrics by Condition', fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set x-axis labels
    if cid:
        plt.xlabel('Condition ID', fontsize=12)
        x_labels = conditions
    else:
        plt.xlabel('Condition', fontsize=12)
        x_labels = [condition_dict.get(int(cond), cond) for cond in conditions]
    
    plt.xticks(x, x_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    add_value_labels(rects1)
    add_value_labels(rects2)
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or display plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binary metrics plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig

def create_directory(directory_path):
    """
    Create a directory if it doesn't already exist.

    Args:
        directory_path (str): The path of the directory to create.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created or already exists: {directory_path}")
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")


def training_complete_message(best_params_file_path, final_model_file_path):
    """
    Display a message to indicate that the training is complete.
    """
    print("\nTraining complete!")
    print("-------------------------------------")
    print(f"Best hyperparameters saved to {best_params_file_path}")
    print(f"Trained model saved to {final_model_file_path}")
    print("-------------------------------------")


def testing_complete_message(metrics):
    """
    Display a message to indicate that the testing is complete.
    
    Args:
        metrics: Either a tuple (accuracy, auc) for binary classification
                or a float for regression (r-squared)
    """
    print("\nTesting complete!")
    print("-------------------------------------")
    
    if isinstance(metrics, tuple):
        # Binary classification metrics
        accuracy, auc = metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
    else:
        # Regression metric (R²)
        print(f"R-squared score: {metrics:.4f}")
    
    print("-------------------------------------")
    print("Goodbye!")