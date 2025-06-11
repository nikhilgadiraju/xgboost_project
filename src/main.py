"""
File: main.py
Author: Roy Huang
Email: ruoqiuhuang@gmail.com
Date: 2025-02-27
Description: Main script for running the model training and evaluation pipeline.
"""

from data_processing import *
from model_training import *
from utils import *
from config import *
import json
import time

def training_loop(df, condition_df, condition_dict):
    # Initialize the output metrics value dataframe
    if BINARY:
        metrics_df = pd.DataFrame(columns=['condition', 'accuracy', 'auc'])
    else:
        metrics_df = pd.DataFrame(columns=['condition', 'r2'])

    for index, key, in enumerate(condition_dict.keys()):
        # Add relevant condition column to the dataset
        df, condition_label = add_condition_column(df, condition_df, key)

        # Select machine region
        features, health_condition, recommended_split = select_machine_region(df, condition_label)

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(features, health_condition, recommended_split)
        
        # Create timestamp for saving files
        timestamp = create_timestamp()

        # Train model using manual grid search with cross-validation
        best_params, best_score = manual_grid_search(X_train, y_train)

        # Save the best hyperparameters and score to a JSON file
        #best_params_file_path = save_hyperparameters(best_params, best_score, timestamp)

        # Train the final model using the best hyperparameters
        final_model = train_final_model(X_train, X_val, y_train, y_val, best_params)

        # Save the final model
        #final_model_file_path = save_trained_model(final_model, timestamp)

        if BINARY:
            # Evaluate auc and accuracy for binary classification using final model
            accuracy, auc_score = evaluate_model(final_model, X_test, y_test, 
                                              key=key, 
                                              condition_label=condition_dict[key])

            # Append the accuracy and auc score to the dataframe
            metrics_df.loc[condition_label] = [key, accuracy, auc_score]
            print(f"Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
            
            # Message to user with binary metrics
            testing_complete_message((accuracy, auc_score))
        else:
            # Evaluate r^2 for regression using final model
            r2 = evaluate_model(final_model, X_test, y_test, 
                              key=key, 
                              condition_label=condition_dict[key])

            # Append the r^2 value to the dataframe
            metrics_df.loc[condition_label] = [key, r2]
            print(f"R²: {r2:.3f}")
            
            # Message to user with regression metric
            testing_complete_message(r2)

    return metrics_df


def main(loop=False):
    """
    Main execution script for XGBoost model training and evaluation.
    Supports both binary classification and regression based on BINARY flag.
    """
    # Perform initial checks
    if not pre_run_check():
        return

    welcome_message()

    # Load datasets
    df = load_dataset(CSV_PATH)
    condition_df = load_dataset(CONDITION_PATH)
    condition_dict = create_condition_dict(condition_df)

    # Create necessary directories
    create_directory(RESULTS_FOLDER)
    create_directory(PLOTS_FOLDER)

    if not start_or_quit():
        return

    if loop:
        # Run the model training and evaluation for all conditions
        metrics_df = training_loop(df, condition_df, condition_dict)

        # Save results to CSV with appropriate filename
        if BINARY:
            filename = "binary_classification_results.csv"
            plot_filename = "binary_metrics_chart.png"
        else:
            filename = "regression_results.csv"
            plot_filename = "r2_bar_chart.png"
        
        metrics_df.to_csv(os.path.join(RESULTS_FOLDER, filename), index=False)

        # Create appropriate visualization based on task type
        if BINARY:
            # Add function to plot binary classification metrics
            plot_binary_metrics(metrics_df, condition_dict, 
                              save_path=os.path.join(RESULTS_FOLDER, plot_filename))
        else:
            plot_r2_bar_chart(metrics_df, condition_dict, cid=True, 
                             save_path=os.path.join(RESULTS_FOLDER, plot_filename))
    
    else: 
        # Filter condition_dict for specific concept ID
        filtered_dict = {key: value for key, value in condition_dict.items() 
                        if key == 4182210}
        # measurement: 3004410 (HgbA1C%)
        # condition: 4317977 (Cataracts)
        
        # Run model for single condition
        metrics_df = training_loop(df, condition_df, filtered_dict)

        # Print results based on task type
        if BINARY:
            print(f"\nResults for selected condition:")
            print(f"Accuracy: {metrics_df['accuracy'].values[0]:.3f}")
            print(f"AUC: {metrics_df['auc'].values[0]:.3f}")
        else:
            print(f"\nR² value for selected condition: {metrics_df['r2'].values[0]:.3f}")

if __name__ == "__main__":
    main(loop=True)
    
