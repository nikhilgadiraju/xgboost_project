"""
Paired-eye analysis and camera-classification utilities.

This module implements the manuscript Step 1 pairing logic and the Step 2 paired-eye
camera classification workflow.

Key requirements enforced here:
- same-eye pairing, not participant-level OD/OS pairing;
- only the target protocols are kept: Maestro2 (Macula) and Triton (Macula 6x6);
- diabetes labels are derived as a binary participant-level OMOP concept 201826 flag;
- paired-image model training uses the same underlying latent vectors as the rest of the project;
- participant-level bootstrap 95% CIs are computed across test-set participants.
"""

import os
import sys
import time
import argparse
from contextlib import redirect_stdout
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import model_training as mt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model_training import manual_grid_search, train_final_model

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CONDITION_PATH,
    CSV_PATH,
    DIABETES_CONCEPT_ID,
    LEARNING_RATE,
    NUM_ROUND,
    MAX_DEPTH,
    MIN_CHILD_WEIGHT,
    GAMMA,
    REG_LAMBDA,
    REG_ALPHA,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    COLSAMPLE_BYLEVEL,
    PAIRED_BOOTSTRAP_REPLICATES,
    PAIRED_MIN_VALIDATION_AUC,
    PAIRED_RANDOM_SEED,
    PAIRED_RESULTS_FOLDER,
    PAIRING_RULE,
    SAVED_RUNS_FOLDER,
)
from utils import run_with_logging

# ---------------------------------------------------------------------
# Paired-eye analysis utilities
# ---------------------------------------------------------------------
# Build same-eye camera pairs and run paired camera classification with
# validation-driven early stopping and bootstrap confidence intervals.
# ---------------------------------------------------------------------

TARGET_PROTOCOLS = {
    "macula": "Maestro2 (Macula)",
    "macula_6x6": "Triton (Macula 6x6)",
}


def _rename_latest_saved_run() -> str:
    """Rename the most recent saved run directory with a paired_analyses prefix."""
    if not os.path.exists(SAVED_RUNS_FOLDER):
        os.makedirs(SAVED_RUNS_FOLDER, exist_ok=True)
    run_dirs = [
        os.path.join(SAVED_RUNS_FOLDER, name)
        for name in os.listdir(SAVED_RUNS_FOLDER)
        if os.path.isdir(os.path.join(SAVED_RUNS_FOLDER, name))
    ]
    if not run_dirs:
        return ""
    latest_dir = max(run_dirs, key=os.path.getmtime)
    latest_name = os.path.basename(latest_dir)
    if latest_name.startswith("paired_analyses_"):
        return latest_dir
    new_dir = os.path.join(SAVED_RUNS_FOLDER, f"paired_analyses_{latest_name}")
    os.rename(latest_dir, new_dir)
    return new_dir


def run_paired_analyses_cli() -> pd.DataFrame:
    """Run the paired-analysis workflow and mirror the output to the terminal and saved run logs."""
    result = run_with_logging(_run_paired_analysis_pipeline)
    _rename_latest_saved_run()
    print("\nSaved paired analysis run with paired_analyses prefix in saved_runs.")
    return result


def _run_paired_analysis_pipeline() -> pd.DataFrame:
    """Load datasets, run the pairing summary, then train the paired camera models."""
    print("-------------------------------------")
    print("Paired-eye analysis pipeline")
    print("-------------------------------------")
    print("Load paired-eye and condition datasets.")
    df = pd.read_csv(CSV_PATH)
    condition_df = pd.read_csv(CONDITION_PATH)
    print(f"CFP dataset shape: {df.shape}")
    print(f"Condition dataset shape: {condition_df.shape}")

    print("\nStep 1: build and summarize same-eye camera pairs.")
    paired_summary = run_paired_analysis(df, condition_df)
    print(paired_summary)

    print("\nStep 2: train Maestro2 vs Triton paired-eye camera classifier.")
    paired_camera_summary = run_step_2_classifier()
    print(paired_camera_summary.to_string(index=False))
    return paired_camera_summary


def _ensure_results_dir() -> str:
    """Create the paired results folder if it does not already exist."""
    os.makedirs(PAIRED_RESULTS_FOLDER, exist_ok=True)
    return PAIRED_RESULTS_FOLDER


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that all required columns exist before processing paired-eye data."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required paired-eye columns: " + ", ".join(missing)
        )


def _image_id_column(row: pd.Series) -> str:
    """Choose the most stable image identifier for a row when preparing pair selection."""
    for candidate in ["filename", "image_id", "imageId", "id"]:
        value = row.get(candidate)
        if pd.notna(value):
            return str(value)
    return str(row.name)


def _coerce_eye_key(series: pd.Series) -> pd.Series:
    """Normalize laterality values to OD/OS strings for eye-level pairing."""
    return series.astype(str).str.upper().str.strip()


def _print_device_protocol_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Print a device-by-protocol table to confirm the camera mapping before filtering."""
    _validate_required_columns(df, ["device", "protocol"])
    crosstab = pd.crosstab(df["device"], df["protocol"]).sort_index(axis=0).sort_index(axis=1)
    print("\nDevice × protocol crosstab for QC-passed images")
    print("=" * 60)
    print(crosstab.to_string())
    return crosstab


def build_paired_eyes(df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """Build one row per qualifying same-eye camera pair for Maestro2 vs Triton."""
    _validate_required_columns(df, ["id", "eye", "device", "protocol", "recommended_split", "filename"])

    qc_df = df.copy()
    qc_df["participant_id"] = pd.to_numeric(qc_df["id"], errors="coerce")
    qc_df["laterality"] = _coerce_eye_key(qc_df["eye"])
    qc_df["image_id"] = qc_df.apply(_image_id_column, axis=1)
    qc_df = qc_df[qc_df["participant_id"].notna() & qc_df["laterality"].notna()].copy()

    _print_device_protocol_crosstab(qc_df)

    kept = qc_df[qc_df["protocol"].isin(list(TARGET_PROTOCOLS.keys()))].copy()
    kept = kept[
        ((kept["protocol"] == "macula") & (kept["device"].isin(["M"])))
        | ((kept["protocol"] == "macula_6x6") & (kept["device"].isin(["T"])))
    ].copy()

    kept = kept.sort_values(["participant_id", "laterality", "protocol", "image_id"], kind="mergesort")
    kept = kept.drop_duplicates(subset=["participant_id", "laterality", "protocol"], keep="first")

    pivot = kept.pivot_table(
        index=["participant_id", "laterality", "recommended_split"],
        columns="protocol",
        values="image_id",
        aggfunc="first",
    ).reset_index()

    paired = pivot[(pivot["macula"].notna()) & (pivot["macula_6x6"].notna())].copy()
    paired["pairing_rule"] = PAIRING_RULE
    paired["selected_camera_pair_rule"] = "first"
    paired["participant_id"] = paired["participant_id"].astype(int)

    if save_path is None:
        save_path = os.path.join(_ensure_results_dir(), "paired_eyes.csv")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    paired.to_csv(save_path, index=False)
    print(f"\nSaved paired eye table to: {save_path}")
    return paired


def summarize_paired_eyes(df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """Summarize paired-eye counts at the image, eye, and participant level."""
    qc_df = df.copy()
    _validate_required_columns(qc_df, ["id", "eye", "device", "protocol", "recommended_split", "filename"])
    qc_df["participant_id"] = pd.to_numeric(qc_df["id"], errors="coerce")
    qc_df["laterality"] = _coerce_eye_key(qc_df["eye"])
    qc_df = qc_df[qc_df["participant_id"].notna() & qc_df["laterality"].notna()].copy()
    qc_df = qc_df[qc_df["protocol"].isin(list(TARGET_PROTOCOLS.keys()))].copy()
    qc_df = qc_df[
        ((qc_df["protocol"] == "macula") & (qc_df["device"].isin(["M"])))
        | ((qc_df["protocol"] == "macula_6x6") & (qc_df["device"].isin(["T"])))
    ].copy()

    paired_df = build_paired_eyes(df, save_path=os.path.join(PAIRED_RESULTS_FOLDER, "paired_eyes.csv"))
    paired_df = paired_df.copy()
    paired_df["participant_id"] = paired_df["participant_id"].astype(int)

    summary_rows = []
    summary_rows.append({"unit": "images", "analysis": "qc_pass_images_total", "value": int(len(df))})
    summary_rows.append({"unit": "images", "analysis": "qc_pass_images_after_camera_filter", "value": int(len(qc_df))})
    summary_rows.append({"unit": "eyes", "analysis": "unique_eyes_after_camera_filter", "value": int(qc_df[["participant_id", "laterality"]].drop_duplicates().shape[0])})
    summary_rows.append({"unit": "eyes", "analysis": "paired_eyes_total", "value": int(paired_df.shape[0])})
    summary_rows.append({"unit": "participants", "analysis": "participants_with_paired_eyes", "value": int(paired_df["participant_id"].nunique())})

    for split in ["train", "val", "test"]:
        split_paired = paired_df[paired_df["recommended_split"] == split]
        summary_rows.append({"unit": "eyes", "analysis": f"paired_eyes_{split}", "value": int(split_paired.shape[0])})
        summary_rows.append({"unit": "participants", "analysis": f"participants_with_paired_eyes_{split}", "value": int(split_paired["participant_id"].nunique())})

        only_macula = qc_df[(qc_df["recommended_split"] == split) & (qc_df["protocol"] == "macula")][["participant_id", "laterality"]].drop_duplicates()
        only_triton = qc_df[(qc_df["recommended_split"] == split) & (qc_df["protocol"] == "macula_6x6")][["participant_id", "laterality"]].drop_duplicates()
        paired_keys = set(map(tuple, paired_df[paired_df["recommended_split"] == split][["participant_id", "laterality"]].to_numpy()))
        only_macula = only_macula[~only_macula.set_index(["participant_id", "laterality"]).index.isin(pd.MultiIndex.from_tuples(list(paired_keys)))]
        only_triton = only_triton[~only_triton.set_index(["participant_id", "laterality"]).index.isin(pd.MultiIndex.from_tuples(list(paired_keys)))]
        summary_rows.append({"unit": "eyes", "analysis": f"eyes_dropped_only_macula_{split}", "value": int(only_macula.shape[0])})
        summary_rows.append({"unit": "eyes", "analysis": f"eyes_dropped_only_triton_{split}", "value": int(only_triton.shape[0])})

    summary_rows.append({"unit": "setting", "analysis": "paired_random_seed", "value": int(PAIRED_RANDOM_SEED)})
    summary_rows.append({"unit": "setting", "analysis": "paired_bootstrap_replicates", "value": int(PAIRED_BOOTSTRAP_REPLICATES)})
    summary_rows.append({"unit": "setting", "analysis": "pairing_rule", "value": PAIRING_RULE})

    summary_df = pd.DataFrame(summary_rows)
    if save_path is None:
        save_path = os.path.join(_ensure_results_dir(), "paired_eyes_summary.csv")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"\nSaved paired eye summary to: {save_path}")
    return summary_df


def report_missing_condition_occurrence(paired_df: pd.DataFrame, condition_df: pd.DataFrame) -> pd.DataFrame:
    """Report participants with no condition_occurrence records and return the binary OMOP concept-201826 map."""
    if condition_df is None:
        raise ValueError("condition_df is required to compute the binary OMOP concept 201826 label.")

    paired_participants = set(paired_df["participant_id"].astype(int).unique())
    condition_lookup = condition_df.copy()
    condition_lookup["person_id"] = pd.to_numeric(condition_lookup["person_id"], errors="coerce")
    condition_lookup["condition_concept_id"] = pd.to_numeric(condition_lookup["condition_concept_id"], errors="coerce")
    condition_lookup = condition_lookup[condition_lookup["person_id"].notna()].copy()
    condition_persons = set(condition_lookup["person_id"].astype(int).unique())
    missing = sorted(paired_participants - condition_persons)
    print(f"\nPaired-eye participants with zero condition_occurrence records: {len(missing)}")
    if missing:
        print("First 20 missing participant IDs:", missing[:20])

    eligible_people = sorted(paired_participants & condition_persons)
    label_map = (
        condition_lookup
        .groupby("person_id")["condition_concept_id"]
        .apply(lambda values: int((values == DIABETES_CONCEPT_ID).any()))
        .astype(int)
        .to_dict()
    )
    records = [{"person_id": person_id, "diabetes_201826": int(label_map.get(person_id, -1))} for person_id in eligible_people]
    return pd.DataFrame(records)


def _build_paired_camera_dataset(latent_df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    """Stack the paired-eye Maestro2 and Triton latent vectors into one classification dataset."""
    _validate_required_columns(latent_df, ["filename", "recommended_split"])
    feature_cols = [column for column in latent_df.columns if column.startswith("feature_")]
    records = []

    for _, row in paired_df.iterrows():
        participant_id = int(row["participant_id"])
        laterality = str(row["laterality"])
        split = str(row["recommended_split"])

        for camera_name, label_value, column_name in [
            ("Maestro2", 0, "macula"),
            ("Triton", 1, "macula_6x6"),
        ]:
            image_name = str(row[column_name])
            match = latent_df[latent_df["filename"].astype(str) == image_name]
            if match.empty:
                continue
            feature_row = match.iloc[0]
            feature_values = feature_row[feature_cols].astype(float).to_numpy()
            record = {
                "participant_id": participant_id,
                "laterality": laterality,
                "recommended_split": split,
                "camera_name": camera_name,
                "camera_label": int(label_value),
                "image_filename": image_name,
            }
            for idx, feature_name in enumerate(feature_cols):
                record[feature_name] = float(feature_values[idx])
            records.append(record)

    if not records:
        raise ValueError("No paired-eye camera rows were matched to the latent-vector dataset.")
    return pd.DataFrame(records)


def _bootstrap_participant_metrics(y_true: np.ndarray, y_prob: np.ndarray, participant_ids: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Compute participant-level nonparametric bootstrap 95% confidence intervals."""
    unique_participants = np.unique(participant_ids)
    rng = np.random.default_rng(seed)
    acc_samples = []
    auc_samples = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_participants, size=len(unique_participants), replace=True)
        sample_idx = []
        for participant_id in sampled:
            sample_idx.extend(np.where(participant_ids == participant_id)[0].tolist())
        if not sample_idx:
            continue
        sampled_y_true = y_true[sample_idx]
        sampled_y_prob = y_prob[sample_idx]
        sampled_pred = (sampled_y_prob >= 0.5).astype(int)
        acc_samples.append(accuracy_score(sampled_y_true, sampled_pred))
        auc_samples.append(roc_auc_score(sampled_y_true, sampled_y_prob))

    if not acc_samples:
        raise ValueError("Bootstrap sample generation failed for the paired-camera test set.")

    return {
        "accuracy_ci_low": float(np.percentile(acc_samples, 2.5)),
        "accuracy_ci_high": float(np.percentile(acc_samples, 97.5)),
        "auc_ci_low": float(np.percentile(auc_samples, 2.5)),
        "auc_ci_high": float(np.percentile(auc_samples, 97.5)),
    }


def run_paired_camera_classification(
    latent_csv_path: str,
    paired_df: Optional[pd.DataFrame] = None,
    save_dir: Optional[str] = None,
    bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    """Train and evaluate the paired-eye Maestro2-vs-Triton camera classifier for one latent-vector source."""
    if paired_df is None:
        paired_df = pd.read_csv(os.path.join(PAIRED_RESULTS_FOLDER, "paired_eyes.csv"))
    latent_df = pd.read_csv(latent_csv_path)

    paired_camera_df = _build_paired_camera_dataset(latent_df, paired_df)
    feature_cols = [column for column in paired_camera_df.columns if column.startswith("feature_")]
    if not feature_cols:
        raise ValueError(f"No feature columns found in the latent-vector CSV: {latent_csv_path}")

    train_df = paired_camera_df[paired_camera_df["recommended_split"] == "train"].copy()
    val_df = paired_camera_df[paired_camera_df["recommended_split"] == "val"].copy()
    test_df = paired_camera_df[paired_camera_df["recommended_split"] == "test"].copy()

    X_train = train_df[feature_cols].astype(float)
    X_val = val_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_train = train_df["camera_label"].astype(int)
    y_val = val_df["camera_label"].astype(int)
    y_test = test_df["camera_label"].astype(int)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"Camera labels are not present in all splits for {latent_csv_path}.")

    original_classification = mt.CLASSIFICATION
    original_multi_class = mt.MULTI_CLASS
    try:
        mt.CLASSIFICATION = True
        mt.MULTI_CLASS = False
        best_params, best_score = mt.manual_grid_search(X_train, y_train, cv_folds=3)
        if best_params is None:
            raise ValueError("manual_grid_search returned no valid parameters for the paired camera classifier.")
        print(f"Using the paired-camera hyperparameters from the config-driven grid search: {best_params}")
        print(f"Best CV AUC from the underlying grid search: {abs(best_score):.4f}")
        model = mt.train_final_model(X_train, X_val, y_train, y_val, best_params)
    finally:
        mt.CLASSIFICATION = original_classification
        mt.MULTI_CLASS = original_multi_class

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": os.path.basename(latent_csv_path),
        "run_type": "grid_search_tuned",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }

    boot = _bootstrap_participant_metrics(y_test.to_numpy(), y_prob, test_df["participant_id"].to_numpy(), n_boot=bootstrap_replicates, seed=PAIRED_RANDOM_SEED)
    metrics.update({
        "accuracy_ci_95_low": boot["accuracy_ci_low"],
        "accuracy_ci_95_high": boot["accuracy_ci_high"],
        "auc_ci_95_low": boot["auc_ci_low"],
        "auc_ci_95_high": boot["auc_ci_high"],
        "bootstrap_replicates": bootstrap_replicates,
    })

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["actual_Maestro2", "actual_Triton"], columns=["pred_Maestro2", "pred_Triton"])

    folder = save_dir or PAIRED_RESULTS_FOLDER
    os.makedirs(folder, exist_ok=True)
    subtag = "tuned_grid_search"
    metrics_path = os.path.join(folder, f"paired_camera_{os.path.splitext(os.path.basename(latent_csv_path))[0]}_{subtag}_metrics.csv")
    confusion_path = os.path.join(folder, f"paired_camera_{os.path.splitext(os.path.basename(latent_csv_path))[0]}_{subtag}_confusion_matrix.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    cm_df.to_csv(confusion_path)

    print(f"\nPaired-camera classification metrics for {os.path.basename(latent_csv_path)} [{subtag}]")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix")
    print("=" * 60)
    print(cm_df.to_string())
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_path}")

    misclassified = []
    for image_filename, participant_id, true_label, pred_label, prob in zip(
        test_df["image_filename"].tolist(),
        test_df["participant_id"].tolist(),
        y_test.to_numpy(),
        y_pred,
        y_prob,
    ):
        if true_label != pred_label:
            misclassified.append({
                "participant_id": int(participant_id),
                "image_filename": str(image_filename),
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "pred_probability": float(prob),
            })
    if misclassified:
        misclassified_df = pd.DataFrame(misclassified)
        print("\nMisclassified test image IDs and participant IDs:")
        print(misclassified_df.to_string(index=False))
        print(f"Distinct participants among misclassified images: {misclassified_df['participant_id'].nunique()}")

    return metrics_df


def run_step_2_classifier() -> pd.DataFrame:
    """Run the paired-eye Maestro2 vs Triton classifier on the pretrained and fine-tuned latent vectors."""
    paired_df = pd.read_csv(os.path.join(PAIRED_RESULTS_FOLDER, "paired_eyes.csv"))
    condition_df = pd.read_csv(CONDITION_PATH)
    report = report_missing_condition_occurrence(paired_df, condition_df)
    print(report.head().to_string(index=False))

    results = []
    for latent_csv in [
        "/home/s440308/Documents/20_github/xgboost_project/data/results_pre-trained_20250703.csv",
        "/home/s440308/Documents/20_github/xgboost_project/data/results_fine-tuned_20250703.csv",
    ]:
        results.append(run_paired_camera_classification(latent_csv, paired_df=paired_df, save_dir=PAIRED_RESULTS_FOLDER))

    combined = pd.concat(results, ignore_index=True)
    summary_path = os.path.join(PAIRED_RESULTS_FOLDER, "paired_camera_summary.csv")
    combined.to_csv(summary_path, index=False)
    print(f"\nSaved combined step-2 summary to: {summary_path}")
    return combined


def run_paired_analysis(df: pd.DataFrame, condition_df: Optional[pd.DataFrame] = None) -> dict:
    """Run the corrected same-eye pairing summary and print the binary OMOP 201826 diabetes handling."""
    paired_df = build_paired_eyes(df)
    summary_df = summarize_paired_eyes(df)

    if condition_df is not None:
        label_df = report_missing_condition_occurrence(paired_df, condition_df)
        print("\nBinary diabetes label map for concept 201826")
        print("=" * 60)
        if label_df.empty:
            print("No eligible paired-eye participants with condition_occurrence records.")
        else:
            print(label_df.head().to_string(index=False))
            print(f"count_positive_201826 = {int((label_df['diabetes_201826'] == 1).sum())}")
            print(f"count_negative_201826 = {int((label_df['diabetes_201826'] == 0).sum())}")

    counts = {
        "paired_eyes_total": int(summary_df[summary_df["analysis"] == "paired_eyes_total"]["value"].iloc[0]),
        "participants_with_paired_eyes": int(summary_df[summary_df["analysis"] == "participants_with_paired_eyes"]["value"].iloc[0]),
        "paired_eyes_train": int(summary_df[summary_df["analysis"] == "paired_eyes_train"]["value"].iloc[0]),
        "participants_with_paired_eyes_train": int(summary_df[summary_df["analysis"] == "participants_with_paired_eyes_train"]["value"].iloc[0]),
        "paired_eyes_val": int(summary_df[summary_df["analysis"] == "paired_eyes_val"]["value"].iloc[0]),
        "participants_with_paired_eyes_val": int(summary_df[summary_df["analysis"] == "participants_with_paired_eyes_val"]["value"].iloc[0]),
        "paired_eyes_test": int(summary_df[summary_df["analysis"] == "paired_eyes_test"]["value"].iloc[0]),
        "participants_with_paired_eyes_test": int(summary_df[summary_df["analysis"] == "participants_with_paired_eyes_test"]["value"].iloc[0]),
    }
    print("\nPaired-eye analysis summary")
    print("=" * 60)
    for key in [
        "paired_eyes_total",
        "participants_with_paired_eyes",
        "paired_eyes_train",
        "participants_with_paired_eyes_train",
        "paired_eyes_val",
        "participants_with_paired_eyes_val",
        "paired_eyes_test",
        "participants_with_paired_eyes_test",
    ]:
        print(f"{key}: {counts[key]}")
    return counts


# ---------------------------------------------------------------------
# Downstream condition-analysis helpers (Stage 1 only)
# ---------------------------------------------------------------------

DEFAULT_DOWNSTREAM_CONCEPT_ID = 4317977
DEFAULT_DOWNSTREAM_LABEL = "cataracts"

DOWNSTREAM_CONDITION_MAP = {
    "cataract": {
        "concept_id": 4317977,
        "concept_name": "Cataracts (in one or both eyes)",
        "aliases": {"cataract", "cataracts"},
    },
    "t2dm": {
        "concept_id": 201826,
        "concept_name": "Type II Diabetes",
        "aliases": {"t2dm", "type2diabetes", "diabetes", "type_ii_diabetes"},
    },
}


def _normalize_condition_label(label: str) -> str:
    """Normalize a downstream condition label to the canonical option names used here."""
    normalized = str(label or "").strip().lower().replace(" ", "_")
    for key, metadata in DOWNSTREAM_CONDITION_MAP.items():
        if normalized in metadata["aliases"] or normalized == key:
            return key
    raise ValueError(f"Unsupported downstream label '{label}'. Supported labels: {sorted(DOWNSTREAM_CONDITION_MAP)}")


def _config_param_grid():
    """Construct the config-driven XGBoost hyperparameter grid used for downstream tuning."""
    values = []
    for learning_rate in (LEARNING_RATE if isinstance(LEARNING_RATE, list) else [LEARNING_RATE]):
        for num_round in (NUM_ROUND if isinstance(NUM_ROUND, list) else [NUM_ROUND]):
            for max_depth in (MAX_DEPTH if isinstance(MAX_DEPTH, list) else [MAX_DEPTH]):
                values.append(
                    {
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(num_round),
                        "max_depth": int(max_depth),
                        "min_child_weight": MIN_CHILD_WEIGHT,
                        "gamma": GAMMA,
                        "reg_lambda": REG_LAMBDA,
                        "reg_alpha": REG_ALPHA,
                        "subsample": SUBSAMPLE,
                        "colsample_bytree": COLSAMPLE_BYTREE,
                        "colsample_bylevel": COLSAMPLE_BYLEVEL,
                        "objective": "binary:logistic",
                        "tree_method": "hist",
                        "device": "cuda",
                        "eval_metric": "logloss",
                    }
                )
    return values


def _downstream_experiment_mask(frame: pd.DataFrame, experiment: str) -> pd.Series:
    """Return the boolean index for a downstream experiment subset."""
    if experiment == "A_both":
        return np.ones(len(frame), dtype=bool)
    if experiment == "B_maestro2_only":
        return frame["camera_name"].eq("Maestro2")
    if experiment == "C_triton_only":
        return frame["camera_name"].eq("Triton")
    raise ValueError(f"Unknown experiment '{experiment}'.")


def _build_downstream_label_frame(
    paired_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    concept_id: int,
    condition_name: str,
) -> pd.DataFrame:
    """Build the participant-level binary label map for a downstream condition concept."""
    if condition_df is None:
        raise ValueError("condition_df is required to compute the downstream condition label.")

    paired_participants = set(pd.to_numeric(paired_df["participant_id"], errors="coerce").dropna().astype(int).unique())
    condition_lookup = condition_df[["person_id", "condition_concept_id"]].copy()
    condition_lookup["person_id"] = pd.to_numeric(condition_lookup["person_id"], errors="coerce")
    condition_lookup["condition_concept_id"] = pd.to_numeric(condition_lookup["condition_concept_id"], errors="coerce")
    condition_lookup = condition_lookup[condition_lookup["person_id"].notna()].copy()
    condition_persons = set(condition_lookup["person_id"].astype(int).unique())
    eligible_people = sorted(paired_participants & condition_persons)
    missing = sorted(paired_participants - condition_persons)

    print(f"\nPaired-eye participants with zero condition_occurrence records for {condition_name}: {len(missing)}")
    if missing:
        print("First 20 missing participant IDs:", missing[:20])

    label_map = (
        condition_lookup.groupby("person_id")["condition_concept_id"]
        .apply(lambda values: int((values == concept_id).any()))
        .astype(int)
        .to_dict()
    )

    records = [
        {
            "person_id": int(person_id),
            "condition_concept_id": int(concept_id),
            "condition_name": str(condition_name),
            f"{condition_name}_label": int(label_map.get(person_id, 0)),
        }
        for person_id in eligible_people
    ]
    return pd.DataFrame(records)


def _run_downstream_stage_1(
    condition_concept_id: int = 4317977,
    condition_label: str = "cataract",
    stage: int = 1,
) -> pd.DataFrame:
    """Run the paired-eye downstream stage-1 condition prediction workflow without evaluating the test set."""
    if stage != 1:
        raise ValueError("This file only implements downstream condition Stage 1. Use --stage 1.")

    effective_label = _normalize_condition_label(condition_label)
    effective_concept_id = int(condition_concept_id)
    condition_name = DOWNSTREAM_CONDITION_MAP[effective_label]["concept_name"]
    if effective_concept_id != DOWNSTREAM_CONDITION_MAP[effective_label]["concept_id"]:
        effective_concept_id = DOWNSTREAM_CONDITION_MAP[effective_label]["concept_id"]

    print("-------------------------------------")
    print(f"Stage 1 downstream condition analysis: {effective_label} (concept {effective_concept_id})")
    print(f"Verified concept name: {condition_name}")
    print("-------------------------------------")

    paired_df = pd.read_csv(os.path.join(PAIRED_RESULTS_FOLDER, "paired_eyes.csv"))
    condition_df = pd.read_csv(CONDITION_PATH)
    condition_df = condition_df.copy()
    condition_df["person_id"] = pd.to_numeric(condition_df["person_id"], errors="coerce")
    condition_df["condition_concept_id"] = pd.to_numeric(condition_df["condition_concept_id"], errors="coerce")
    condition_df = condition_df[condition_df["person_id"].notna()].copy()
    condition_df["person_id"] = condition_df["person_id"].astype(int)

    label_df = _build_downstream_label_frame(
        paired_df,
        condition_df,
        concept_id=effective_concept_id,
        condition_name=effective_label,
    )
    if label_df.empty:
        raise ValueError(f"No eligible paired-eye participants were found for concept {effective_concept_id}.")

    paired_kept = paired_df.merge(
        label_df[["person_id", f"{effective_label}_label"]].rename(columns={"person_id": "participant_id"}),
        on="participant_id",
        how="inner",
    )

    print("\n=== Stage 1: label counts per split after excluding participants without condition records ===")
    for split_name in ["train", "val", "test"]:
        split_df = paired_kept[paired_kept["recommended_split"] == split_name].copy()
        positive = int(split_df[f"{effective_label}_label"].sum())
        negative = int((split_df[f"{effective_label}_label"] == 0).sum())
        print(
            f"split={split_name} paired_eyes={len(split_df)} participants={split_df['participant_id'].nunique()} "
            f"positive={positive} negative={negative}"
        )

    latent_paths = {
        "pretrained": "/home/s440308/Documents/20_github/xgboost_project/data/results_pre-trained_20250703.csv",
        "fine_tuned": "/home/s440308/Documents/20_github/xgboost_project/data/results_fine-tuned_20250703.csv",
    }
    param_grid = _config_param_grid()
    print(f"\nHyperparameter combinations: {len(param_grid)}")

    all_results = []
    for vector_name, latent_path in latent_paths.items():
        latent_df = pd.read_csv(latent_path)
        camera_df = _build_paired_camera_dataset(latent_df, paired_kept[["participant_id", "laterality", "recommended_split", "macula", "macula_6x6"]])
        camera_df = camera_df.merge(
            label_df[["person_id", f"{effective_label}_label"]].rename(columns={"person_id": "participant_id"}),
            on="participant_id",
            how="inner",
        )
        camera_df["target_label"] = camera_df[f"{effective_label}_label"].astype(int)

        feature_cols = [column for column in camera_df.columns if column.startswith("feature_")]
        print(f"\n=== Vector set: {vector_name} ===")

        for experiment_name in ["A_both", "B_maestro2_only", "C_triton_only"]:
            exp_mask = _downstream_experiment_mask(camera_df, experiment_name)
            subset = camera_df[exp_mask].copy()
            train_df = subset[subset["recommended_split"] == "train"].copy()
            val_df = subset[subset["recommended_split"] == "val"].copy()

            if train_df.empty or val_df.empty:
                print(f"{experiment_name}: skipped (empty train/val split)")
                continue

            X_train = train_df[feature_cols].astype(float)
            X_val = val_df[feature_cols].astype(float)
            y_train = train_df["target_label"].astype(int).to_numpy()
            y_val = val_df["target_label"].astype(int).to_numpy()
            g_train = train_df["participant_id"].astype(int).to_numpy()

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                print(f"{experiment_name}: skipped (insufficient labels in train/val)")
                continue

            best_auc = -1.0
            best_params = None
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=PAIRED_RANDOM_SEED)

            for params in param_grid:
                fold_aucs = []
                valid = True
                for tr_idx, te_idx in cv.split(X_train, y_train, groups=g_train):
                    X_tr = X_train.iloc[tr_idx]
                    X_te = X_train.iloc[te_idx]
                    y_tr = y_train[tr_idx]
                    y_te = y_train[te_idx]

                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                        valid = False
                        break

                    model = xgb.XGBClassifier(
                        **params,
                        early_stopping_rounds=10,
                        random_state=PAIRED_RANDOM_SEED,
                        n_jobs=1,
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                    prob = model.predict_proba(X_te)[:, 1]
                    fold_aucs.append(roc_auc_score(y_te, prob))

                if not valid or not fold_aucs:
                    continue
                mean_auc = float(np.mean(fold_aucs))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = params.copy()

            if best_params is None:
                print(f"{experiment_name}: tuning failed (no valid parameter set)")
                continue

            final_model = xgb.XGBClassifier(
                **best_params,
                early_stopping_rounds=10,
                random_state=PAIRED_RANDOM_SEED,
                n_jobs=1,
            )
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = final_model.get_booster().best_iteration
            if best_iter is not None:
                final_model.set_params(n_estimators=best_iter + 1)
                final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            val_prob = final_model.predict_proba(X_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_prob))
            print(
                f"{experiment_name}: val_auc={val_auc:.4f} cv_best_auc={best_auc:.4f} best_iter={best_iter} "
                f"params={{learning_rate:{best_params['learning_rate']}, n_estimators:{best_params['n_estimators']}, max_depth:{best_params['max_depth']}}}"
            )

            all_results.append(
                {
                    "vector_set": vector_name,
                    "condition_name": effective_label,
                    "condition_concept_id": int(effective_concept_id),
                    "experiment": experiment_name,
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)),
                    "val_auc": val_auc,
                    "cv_best_auc": best_auc,
                    "best_params": best_params,
                    "concept_name": condition_name,
                }
            )

    summary = pd.DataFrame(all_results)
    save_path = os.path.join(PAIRED_RESULTS_FOLDER, f"downstream_stage1_{effective_label}_{effective_concept_id}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary.to_csv(save_path, index=False)
    print(f"\nSaved downstream Stage 1 summary to: {save_path}")
    print("\nStage 1 complete. No test-set metrics were computed.")
    return summary


def run_downstream_condition_stage_1(
    condition_concept_id: int = 4317977,
    condition_label: str = "cataract",
    stage: int = 1,
) -> pd.DataFrame:
    """CLI-friendly wrapper for the downstream Stage 1 condition analysis."""
    result = run_with_logging(_run_downstream_stage_1, condition_concept_id, condition_label, stage)
    print("\nSaved downstream Stage 1 analysis with paired_analyses prefix in saved_runs.")
    return result


def _handle_cli() -> None:
    """Parse CLI args for paired analyses and downstream condition prediction."""
    parser = argparse.ArgumentParser(description="Paired-eye analysis workflows")
    parser.add_argument("--analysis", choices=["pairing", "camera", "downstream"], default="pairing")
    parser.add_argument("--label", default="cataract", help="Downstream condition label: cataract or t2dm")
    parser.add_argument("--stage", type=int, default=1, help="Downstream stage to run; only stage 1 is implemented in this file")
    args = parser.parse_args()

    if args.analysis == "downstream":
        run_downstream_condition_stage_1(
            condition_concept_id=DOWNSTREAM_CONDITION_MAP[_normalize_condition_label(args.label)]["concept_id"],
            condition_label=_normalize_condition_label(args.label),
            stage=args.stage,
        )
        return

    if args.analysis == "pairing":
        run_paired_analyses_cli()
        return

    if args.analysis == "camera":
        run_step_2_classifier()
        return


if __name__ == "__main__":
    _handle_cli()
