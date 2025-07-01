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

def training_loop(df, condition_df, condition_dict, filter_classes=None):
    # Initialize the output metrics value dataframe
    if CLASSIFICATION:
        if MULTI_CLASS:
            metrics_df = pd.DataFrame(columns=['condition', 'accuracy', 'report'])
            # For multi-class, process single task
            try:
                # Direct feature extraction (no condition column needed)
                features, health_condition, recommended_split = prepare_feature_data(df, 'Camera Types')
                
                # Prepare data
                X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_data(
                    features, health_condition, recommended_split)
                
                # Check class distribution
                is_valid, class_dist = check_class_distribution(
                    health_condition, 'full dataset', 'Camera Types')
                if not is_valid:
                    print("Insufficient class distribution for camera types")
                    return pd.DataFrame()
                
                # Train model using manual grid search
                best_params, best_score = manual_grid_search(X_train, y_train)
                
                if best_params is None:
                    print("Grid search failed for multi-class model")
                    return pd.DataFrame()
                
                # Train final model
                final_model = train_final_model(X_train, X_val, y_train, y_val, best_params)
                
                # Evaluate model (with optional filtering)
                accuracy, report = evaluate_model(
                    final_model, X_test, y_test, 
                    key='multi_class', condition_label='Camera Types',
                    label_encoder=label_encoder,
                    filter_classes=filter_classes)
                
                # Update metrics DataFrame
                suffix = f" (Filtered: {', '.join(filter_classes)})" if filter_classes else ""
                metrics_df.loc[0] = [f'Camera Types{suffix}', accuracy, report]
                
                print(f"\nAccuracy: {accuracy:.3f}")
                if filter_classes:
                    print(f"Evaluated classes: {', '.join(filter_classes)}")
                print("\nClassification Report:")
                print(report)
                
            except Exception as e:
                print(f"Error in multi-class training: {str(e)}")
                return pd.DataFrame()
            
            # Return early for multi-class since we don't need to process conditions
            return metrics_df
        else:
            metrics_df = pd.DataFrame(columns=['condition', 'accuracy', 'auc'])
    else:
        metrics_df = pd.DataFrame(columns=['condition', 'r2'])

    # Only process conditions for binary/regression
    if condition_dict is None:
        print("Error: condition_dict is required for binary classification and regression")
        return pd.DataFrame()

    for index, key in enumerate(condition_dict.keys()):
        try:
            # Add relevant condition column to the dataset
            df, condition_label = add_condition_column(df, condition_df, key)

            # Select machine region
            features, health_condition, recommended_split = prepare_feature_data(df, condition_label)

            # Check initial class distribution
            if CLASSIFICATION:
                is_valid, class_dist = check_class_distribution(
                    health_condition, 'full dataset', condition_label)
                if not is_valid:
                    print(f"Skipping condition {condition_label} - insufficient class distribution")
                    continue
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_data(
                features, health_condition, recommended_split)
            
            # Check class distribution in all splits
            if CLASSIFICATION:
                skip_condition = False
                for data, name in [(y_train, 'training'), (y_val, 'validation'), 
                                 (y_test, 'testing')]:
                    is_valid, class_dist = check_class_distribution(
                        data, name, condition_label)
                    if not is_valid:
                        print(f"Skipping condition {condition_label} - insufficient distribution in {name}")
                        skip_condition = True
                        break
                
                if skip_condition:
                    continue

            # Train model using manual grid search
            best_params, best_score = manual_grid_search(X_train, y_train)
            
            if best_params is None:
                print(f"Skipping condition {condition_label} - grid search failed")
                continue

            # Train final model
            final_model = train_final_model(X_train, X_val, y_train, y_val, best_params)

            if CLASSIFICATION:
                if MULTI_CLASS:
                    accuracy, report = evaluate_model(
                        final_model, X_test, y_test, key=key, 
                        condition_label=condition_dict[key],
                        label_encoder=label_encoder)
                    
                    metrics_df.loc[condition_label] = [key, accuracy, report]
                    print(f"\nAccuracy: {accuracy:.3f}")
                    print("\nClassification Report:")
                    print(report)
                else:
                    accuracy, auc_score = evaluate_model(
                        final_model, X_test, y_test, key=key, 
                        condition_label=condition_dict[key],
                        label_encoder=None)
                    
                    metrics_df.loc[condition_label] = [key, accuracy, auc_score]
                    
                    if np.isnan(auc_score):
                        print(f"Accuracy: {accuracy:.3f}, AUC: Not available")
                    else:
                        print(f"Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
                        if auc_score > 0.75:
                            print("High performance model - confusion matrix generated")
            else:
                r2 = evaluate_model(final_model, X_test, y_test, 
                                  key=key, condition_label=condition_dict[key])
                metrics_df.loc[condition_label] = [key, r2]
                print(f"R²: {r2:.3f}")

        except Exception as e:
            print(f"\nError processing condition {condition_label}: {str(e)}")
            continue

    return metrics_df


def main(loop=False, filter_classes=None):
    """
    Main execution script for XGBoost model training and evaluation.
    Supports both binary classification and regression based on BINARY flag.
    """
    # Clear the log file at the start of the run
    clear_log_file()
    
    # Perform initial checks
    if not pre_run_check():
        return

    welcome_message()

    # Load datasets
    df = load_dataset(CSV_PATH)
    print("\nMain dataset shape:", df.shape)

    if not (CLASSIFICATION and MULTI_CLASS):
        # Only load condition/measurement data for binary and regression
        condition_df = pd.read_csv(CONDITION_PATH)
        print("Condition/Measurement dataset shape:", condition_df.shape)
        # Create condition dictionary for binary/regression
        condition_dict = create_condition_dict(condition_df)
    else:
        # For multi-class, we don't need condition data
        condition_df = None
        condition_dict = None

    # Create necessary directories
    create_directory(RESULTS_FOLDER)
    create_directory(PLOTS_FOLDER)

    if not start_or_quit():
        return

    if loop:
        # Run the model training and evaluation for all conditions 
        # (with filtering for multiclass)
        filtering_list = ["triton_3d_radial", "triton_macula_12x12", "triton_macula_6x6"]
        metrics_df = training_loop(df, condition_df, condition_dict, filter_classes=filter_classes)

        if not metrics_df.empty:
            # Update filenames to reflect filtering
            suffix = "_filtered" if filter_classes else ""
            # Save results to CSV with appropriate filename
            if CLASSIFICATION:
                if MULTI_CLASS:
                    filename = f"multi_class_classification_results{suffix}.csv"
                    plot_filename = f"multi_class_metrics_chart{suffix}.png"
                else:
                    filename = "binary_classification_results.csv"
                    plot_filename = "binary_metrics_chart.png"
            else:
                filename = "regression_results.csv"
                plot_filename = "r2_bar_chart.png"
            
            metrics_df.to_csv(os.path.join(RESULTS_FOLDER, filename), index=False)

            # Create appropriate visualization based on task type
            if CLASSIFICATION:
                if MULTI_CLASS:
                    plot_multi_class_metrics(metrics_df, 
                                        save_path=os.path.join(RESULTS_FOLDER, plot_filename),
                                        classes=filter_classes)
                else:
                    plot_binary_metrics(metrics_df, condition_dict, 
                                    save_path=os.path.join(RESULTS_FOLDER, plot_filename))
            else:
                plot_r2_bar_chart(metrics_df, condition_dict, cid=False, 
                                save_path=os.path.join(RESULTS_FOLDER, plot_filename))
            
            # Print results
            print(f"\nResults saved to: {os.path.join(RESULTS_FOLDER, filename)}")
            print(f"Plot saved to: {os.path.join(RESULTS_FOLDER, plot_filename)}")
    
    else:
        # Filter condition_dict for specific concept ID
        filtered_dict = {key: value for key, value in condition_dict.items() 
                        if key == 4182210}
        # measurement: 3004410 (HgbA1C%)
        # condition: 4317977 (Cataracts)

        # Run model for single condition
        metrics_df = training_loop(df, condition_df, filtered_dict)

        if metrics_df.empty:
            print("\nNo valid model could be trained for the selected condition")
            return

        # Print results based on task type
        if CLASSIFICATION:
            try:
                print(f"\nResults for selected condition:")
                print(f"Accuracy: {metrics_df['accuracy'].values[0]:.3f}")
                if MULTI_CLASS:
                    print("\nClassification Report:")
                    print(metrics_df['report'].values[0])
                else:
                    print(f"AUC: {metrics_df['auc'].values[0]:.3f}")
            except IndexError:
                print("\nNo valid results available for classification")
        else:
            try:
                print(f"\nR² value for selected condition: {metrics_df['r2'].values[0]:.3f}")
            except IndexError:
                print("\nNo valid R² value available for regression")

if __name__ == "__main__":
    filter_classes = ["maestro2_3d_macula", "triton_macula_12x12"]
    main(loop=True, filter_classes=None)
    
