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
    # Initialize the r2 value dataframe
    r2_df = pd.DataFrame(columns=['condition', 'r2'])    

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

        # Evaluate r^2 value using final trained model
        r2 = evaluate_model(final_model, X_test, y_test, key)

        # Append the r^2 value to the dataframe
        r2_df.loc[condition_label] = [key, r2]

        # Message to user
        #training_complete_message(best_params_file_path, final_model_file_path)
        testing_complete_message(r2)

    return r2_df


def main(loop=False):
    """
    Main excution script for XGBoost tuning. 
    """

    if not pre_run_check():
        return

    welcome_message()

    # Load the dataset
    df = load_dataset(CSV_PATH)

    # Load the condition dataset
    condition_df = load_dataset(CONDITION_PATH)

    # Load condition dictionary
    condition_dict = create_condition_dict(condition_df)

    # Create results directory
    create_directory(RESULTS_FOLDER)

    # Create plots directory
    create_directory(PLOTS_FOLDER)

    if not start_or_quit():
        return

    if loop:
        # Run the model training and evaluation for a single condition
        r2_df = training_loop(df, condition_df, condition_dict)

        # Save the r^2 results to a CSV file
        r2_df.to_csv(os.path.join(RESULTS_FOLDER, "r2_results.csv"), index=False)

        # Plot r^2 bars
        plot_r2_bar_chart(r2_df, condition_dict, cid=True, save_path=os.path.join(RESULTS_FOLDER, "r2_bar_chart.png"))
    
    else: 
        # Filter the condition_dict to only include the specified concept ID
        filtered_dict = {key: value for key, value in condition_dict.items() if key == 3000744}
        
        # Regenerate r2_df using the provided concept ID
        r2_df = training_loop(df, condition_df, filtered_dict)

        # Print r2_df r2 value
        print(f"\nR^2 values for the selected condition: {r2_df['r2'].values[0]}")

if __name__ == "__main__":
    main(loop=True)
    
    # # Load the condition dataset
    # condition_df = load_dataset(CONDITION_PATH)

    # # Load condition dictionary
    # condition_dict = create_condition_dict(condition_df)

    # # Load r2_df
    # r2_df = load_dataset(os.path.join(RESULTS_FOLDER, "r2_results.csv"))

    # # Plotting priority 
    # plot_r2_bar_chart(r2_df, condition_dict, cid=True, save_path=os.path.join(RESULTS_FOLDER, "r2_bar_chart.png"))
