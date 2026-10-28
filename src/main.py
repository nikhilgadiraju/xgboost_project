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
from paired_analyses import run_paired_analysis
import argparse
import json
import time

def run_analysis_selection(df, condition_df, analysis_name):
    """Run the requested analysis without invoking the training pipeline."""
    if analysis_name in {"pairing", "all"}:
        run_paired_analysis(df, condition_df)
    if analysis_name in {"camera", "all"}:
        from paired_analyses import paired_camera_classification_summary
        paired_camera_classification_summary(df, os.path.join(PAIRED_RESULTS_FOLDER, "paired_camera_summary.csv"))
    if analysis_name in {"diabetes", "all"}:
        if condition_df is None:
            condition_df = pd.read_csv(CONDITION_PATH)
        from paired_analyses import downstream_diabetes_summary
        downstream_diabetes_summary(df, condition_df, save_path=os.path.join(PAIRED_RESULTS_FOLDER, "downstream_diabetes_summary.csv"))


def training_loop(df, condition_df, condition_dict, train_filter_classes=None, eval_filter_classes=None):
    # Initialize the output metrics value dataframe
    if CLASSIFICATION:
        if MULTI_CLASS:
            metrics_df = pd.DataFrame(columns=['condition', 'accuracy', 'report'])
            # For multi-class, process single task
            try:
                # Filter classes at data preparation stage
                features, health_condition, recommended_split = prepare_feature_data(
                    df, 'Camera Types', filter_classes=train_filter_classes)
                
                # Rest of processing uses filtered data
                X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_data(
                    features, health_condition, recommended_split)
                
                # Preserve row-aligned metadata from the filtered test set for prediction exports.
                metadata_df = df.copy()
                if train_filter_classes is not None:
                    mask = np.isin(metadata_df['protocol'], train_filter_classes)
                    metadata_df = metadata_df.loc[mask].copy()
                test_metadata = metadata_df.loc[metadata_df['recommended_split'] == 'test', ['id', 'eye', 'filename', 'recommended_split']].copy()
                if len(test_metadata) != len(y_test):
                    raise ValueError(f"Filtered test metadata rows ({len(test_metadata)}) do not match evaluation rows ({len(y_test)}).")
                if test_metadata['filename'].duplicated().any():
                    raise ValueError("Filtered test metadata contains duplicate filenames.")
                
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
                
                # Evaluate model with optional additional filtering
                accuracy, report = evaluate_model(
                    final_model, X_test, y_test, 
                    key='multi_class', 
                    condition_label='Camera Types',
                    label_encoder=label_encoder,
                    filter_classes=eval_filter_classes,
                    sample_metadata=test_metadata,
                )
                
                # Update metrics DataFrame
                train_suffix = f" (Trained on: {', '.join(train_filter_classes)})" if train_filter_classes else ""
                eval_suffix = f" (Evaluated on: {', '.join(eval_filter_classes)})" if eval_filter_classes else ""
                metrics_df.loc[0] = [f'Camera Types{train_suffix}{eval_suffix}', accuracy, report]
                
                print(f"\nAccuracy: {accuracy:.3f}")
                if train_filter_classes:
                    print(f"Trained on classes: {', '.join(train_filter_classes)}")
                if eval_filter_classes:
                    print(f"Evaluated on classes: {', '.join(eval_filter_classes)}")
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
            
            # Keep row-aligned metadata for the test set so predictions can include identifiers.
            metadata_df = df.copy()
            test_metadata = metadata_df.loc[metadata_df['recommended_split'] == 'test', ['id', 'eye', 'filename', 'recommended_split']].copy()
            if len(test_metadata) != len(y_test):
                raise ValueError(f"Test metadata rows ({len(test_metadata)}) do not match evaluation rows ({len(y_test)}).")
            if test_metadata['filename'].duplicated().any():
                raise ValueError("Test metadata contains duplicate filenames.")
            
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
                        label_encoder=label_encoder,
                        sample_metadata=test_metadata)
                    
                    metrics_df.loc[condition_label] = [key, accuracy, report]
                    print(f"\nAccuracy: {accuracy:.3f}")
                    print("\nClassification Report:")
                    print(report)
                else:
                    accuracy, auc_score = evaluate_model(
                        final_model, X_test, y_test, key=key, 
                        condition_label=condition_dict[key],
                        label_encoder=None,
                        sample_metadata=test_metadata)
                    
                    metrics_df.loc[condition_label] = [key, accuracy, auc_score]
                    
                    if np.isnan(auc_score):
                        print(f"Accuracy: {accuracy:.3f}, AUC: Not available")
                    else:
                        print(f"Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
                        if auc_score > 0.75:
                            print("High performance model - confusion matrix generated")
            else:
                r2 = evaluate_model(final_model, X_test, y_test, 
                                  key=key, condition_label=condition_dict[key],
                                  sample_metadata=test_metadata)
                metrics_df.loc[condition_label] = [key, r2]
                print(f"R²: {r2:.3f}")

        except Exception as e:
            print(f"\nError processing condition {condition_label}: {str(e)}")
            continue

    return metrics_df


def main(loop=False, train_filter_classes=None, eval_filter_classes=None):
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

    #if not start_or_quit():
        #return

    if loop:
        # Run the model training and evaluation for all conditions 
        # (with filtering for multiclass)
        filtering_list = ["triton_3d_radial", "triton_macula_12x12", "triton_macula_6x6"]
        metrics_df = training_loop(df, condition_df, condition_dict, 
                                 train_filter_classes=train_filter_classes,
                                 eval_filter_classes=eval_filter_classes)

        if not metrics_df.empty:
            # Update filenames to reflect filtering
            train_suffix = "_train_filtered" if train_filter_classes else ""
            eval_suffix = "_eval_filtered" if eval_filter_classes else ""
            suffix = f"{train_suffix}{eval_suffix}"

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
            record_run_data("results", {
                "metrics_file": os.path.join(RESULTS_FOLDER, filename),
                "plot_file": os.path.join(RESULTS_FOLDER, plot_filename),
                "metrics": metrics_df.to_dict(orient="records"),
            })

            # Create appropriate visualization based on task type
            if CLASSIFICATION:
                if MULTI_CLASS:
                    plot_multi_class_metrics(
                        metrics_df, 
                        save_path=os.path.join(RESULTS_FOLDER, plot_filename),
                        classes=eval_filter_classes if eval_filter_classes else train_filter_classes)
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
        # measurement: 3004410 (HgbA1C%)
        # condition: 4317977 (Cataracts)
        filtered_dict = {key: value for key, value in condition_dict.items() 
                        if key == 4182210}

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
    parser = argparse.ArgumentParser(description="XGBoost camera classification pipeline")
    parser.add_argument("--analysis", choices=["training", "pairing", "camera", "diabetes", "all"], default="training")
    parser.add_argument("--train-filter-classes", nargs="*", default=None)
    parser.add_argument("--eval-filter-classes", nargs="*", default=None)
    args = parser.parse_args()

    if args.analysis != "training":
        df = load_dataset(CSV_PATH)
        condition_df = pd.read_csv(CONDITION_PATH) if os.path.exists(CONDITION_PATH) else None
        os.makedirs(PAIRED_RESULTS_FOLDER, exist_ok=True)
        run_analysis_selection(df, condition_df, args.analysis)
        raise SystemExit(0)

    # Example 1: Filter both training and evaluation
    # Removed classes: "triton_macula_12x12", "eidon_mosaic"

    # Example 1: All classes present
    train_classes_1 = [
        "uwf_central",
        "uwf_nasal",
        "macula",
        "wide",
        "mac_or_disk",
        "radial",
        "macula_6x6",
    ]

    # Example 2: macula-centered classes
    train_classes_2 = [
        "uwf_central",
        "macula",
        "mac_or_disk",
        "macula_6x6"
    ]

    # Example 3: maestro2 vs triton (topcon brand)
    train_classes_3 = [
        "macula",
        "macula_6x6"
    ]

    eval_classes = args.eval_filter_classes
    if args.train_filter_classes is not None:
        train_classes = args.train_filter_classes
    else:
        train_classes = train_classes_3

    if not args.train_filter_classes and not args.eval_filter_classes:
        run_with_logging(
            main,
            loop=True,
            train_filter_classes=train_classes,
            eval_filter_classes=eval_classes)
    else:
        run_with_logging(
            main,
            loop=True,
            train_filter_classes=train_classes,
            eval_filter_classes=eval_classes)
