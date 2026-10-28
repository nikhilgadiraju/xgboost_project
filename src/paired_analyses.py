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

import ast
import json
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
                    "best_iteration": int(best_iter) if best_iter is not None else None,
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


def _youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose the validation threshold that maximizes Youden's J statistic."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.unique(np.sort(y_prob))
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_j = -np.inf
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tn = np.sum((pred == 0) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)
    return float(best_threshold)


def _bootstrap_auc_with_resample(y_true: np.ndarray, y_prob: np.ndarray, participant_ids: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Participant-level bootstrap AUC CI with skipped replicates counted."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    participant_ids = np.asarray(participant_ids)
    if y_true.size == 0 or y_prob.size == 0 or participant_ids.size == 0:
        return float("nan"), float("nan"), 0, 0
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), 0, 0

    unique_participants = np.unique(participant_ids)
    rng = np.random.default_rng(seed)
    auc_values = []
    skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(unique_participants, size=len(unique_participants), replace=True)
        idx = []
        for pid in sampled:
            idx.extend(np.where(participant_ids == pid)[0].tolist())
        if not idx:
            skipped += 1
            continue
        sampled_y = np.asarray(y_true)[idx]
        sampled_p = np.asarray(y_prob)[idx]
        if len(np.unique(sampled_y)) < 2:
            skipped += 1
            continue
        try:
            auc_values.append(float(roc_auc_score(sampled_y, sampled_p)))
        except ValueError:
            skipped += 1
            continue

    if not auc_values:
        return float("nan"), float("nan"), 0, int(n_boot)

    auc_low, auc_high = np.percentile(auc_values, [2.5, 97.5])
    return float(auc_low), float(auc_high), len(auc_values), int(skipped)


def _bootstrap_camera_auc_difference(
    y_true_a: np.ndarray,
    y_prob_a: np.ndarray,
    participant_ids_a: np.ndarray,
    y_true_b: np.ndarray,
    y_prob_b: np.ndarray,
    participant_ids_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
):
    """Resample test participants with replacement, recompute both camera AUCs, and return the percentile CI for the same signed difference as the point estimate."""
    y_true_a = np.asarray(y_true_a).astype(int)
    y_prob_a = np.asarray(y_prob_a).astype(float)
    participant_ids_a = np.asarray(participant_ids_a)
    y_true_b = np.asarray(y_true_b).astype(int)
    y_prob_b = np.asarray(y_prob_b).astype(float)
    participant_ids_b = np.asarray(participant_ids_b)

    if y_true_a.size == 0 or y_true_b.size == 0:
        return float("nan"), float("nan"), 0, 0

    common_participants = np.intersect1d(np.unique(participant_ids_a), np.unique(participant_ids_b))
    if common_participants.size == 0:
        return float("nan"), float("nan"), 0, int(n_boot)

    rng = np.random.default_rng(seed)
    diffs = []
    skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(common_participants, size=len(common_participants), replace=True)
        idx_a = []
        idx_b = []
        for pid in sampled:
            idx_a.extend(np.where(participant_ids_a == pid)[0].tolist())
            idx_b.extend(np.where(participant_ids_b == pid)[0].tolist())
        if not idx_a or not idx_b:
            skipped += 1
            continue
        sampled_y_a = y_true_a[idx_a]
        sampled_p_a = y_prob_a[idx_a]
        sampled_y_b = y_true_b[idx_b]
        sampled_p_b = y_prob_b[idx_b]
        if len(np.unique(sampled_y_a)) < 2 or len(np.unique(sampled_y_b)) < 2:
            skipped += 1
            continue
        try:
            auc_a = float(roc_auc_score(sampled_y_a, sampled_p_a))
            auc_b = float(roc_auc_score(sampled_y_b, sampled_p_b))
            diffs.append(auc_a - auc_b)
        except ValueError:
            skipped += 1
            continue

    if not diffs:
        return float("nan"), float("nan"), 0, int(n_boot)

    diff_low, diff_high = np.percentile(diffs, [2.5, 97.5])
    return float(diff_low), float(diff_high), len(diffs), int(skipped)


def _coerce_camera_test_df(frame: pd.DataFrame, camera_name: str, experiment_name: str, vector_name: str, model_label: str) -> pd.DataFrame:
    """Subset the paired test dataset to a given camera and attach model metadata."""
    camera_df = frame[(frame["camera_name"] == camera_name) & (frame["recommended_split"] == "test")].copy()
    camera_df["experiment"] = experiment_name
    camera_df["vector_set"] = vector_name
    camera_df["model_label"] = model_label
    return camera_df


def _deserialize_best_params(value):
    """Deserialize Stage 1 hyperparameters from a CSV string or dict-like object."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _stage2_model_from_summary(row: pd.Series) -> xgb.XGBClassifier:
    """Construct the exact Stage 1 model from its saved hyperparameters and best iteration."""
    params = _deserialize_best_params(row.get("best_params"))
    best_iter = row.get("best_iteration")
    if best_iter is None:
        if "n_estimators" in params:
            best_iter = int(params["n_estimators"]) - 1
        else:
            raise ValueError(f"Stage 1 summary row is missing best_iteration and n_estimators: {row.to_dict()}")
    model_params = dict(params)
    if "n_estimators" in model_params:
        model_params["n_estimators"] = int(best_iter) + 1
    else:
        model_params["n_estimators"] = int(best_iter) + 1
    model_params["objective"] = "binary:logistic"
    model_params["tree_method"] = "hist"
    model_params["device"] = "cuda"
    model_params["eval_metric"] = "logloss"
    model = xgb.XGBClassifier(
        **model_params,
        random_state=PAIRED_RANDOM_SEED,
        n_jobs=1,
    )
    return model


def _validate_stage1_reproduction(stage1_summary: pd.DataFrame, expected_lookup: dict) -> None:
    """Verify the Stage 1 validation AUCs match the saved summary within tolerance before any test evaluation."""
    print("\nStage 1 reproduction check (validation AUCs and exact saved model specs)")
    mismatch_rows = []
    expected_order = [
        ("pretrained", "A_both"),
        ("pretrained", "B_maestro2_only"),
        ("pretrained", "C_triton_only"),
        ("fine_tuned", "A_both"),
        ("fine_tuned", "B_maestro2_only"),
        ("fine_tuned", "C_triton_only"),
    ]
    for vector_set, experiment in expected_order:
        row = stage1_summary[(stage1_summary["vector_set"] == vector_set) & (stage1_summary["experiment"] == experiment)].iloc[0]
        observed = float(row["val_auc"])
        expected = float(expected_lookup[(vector_set, experiment)])
        params = _deserialize_best_params(row.get("best_params"))
        best_iter = row.get("best_iteration") if row.get("best_iteration") is not None else (int(params.get("n_estimators", 0)) - 1 if "n_estimators" in params else None)
        model = _stage2_model_from_summary(pd.Series(row))
        actual_trees = int(model.get_params().get("n_estimators", 0))
        print(
            f"{vector_set:>10} {experiment:<18} val_auc={observed:.4f} expected={expected:.4f} "
            f"delta={abs(observed - expected):.6f} trees_used={actual_trees} stage1_best_iter={best_iter} params={params}"
        )
        if abs(observed - expected) > 1e-3:
            mismatch_rows.append({
                "vector_set": vector_set,
                "experiment": experiment,
                "observed": observed,
                "expected": expected,
                "delta": abs(observed - expected),
            })
    if mismatch_rows:
        print("\nStage 1 validation AUC mismatch detected before Stage 2 test evaluation:")
        print(pd.DataFrame(mismatch_rows).to_string(index=False))
        raise ValueError("Stage 1 validation AUCs do not match the required reference values; aborting before test set evaluation.")


def _print_stage2_table_1(summary_rows: list[dict]) -> pd.DataFrame:
    """Print the requested AUC table comparing Maestro2 vs Triton for each training regime."""
    table = []
    for item in summary_rows:
        table.append({
            "vector_set": item["vector_set"],
            "trained_on": item.get("trained_on", item.get("experiment_label", "unknown")),
            "tested_on_Maestro2_auc": item["maestro2_auc"],
            "tested_on_Maestro2_ci": item["maestro2_ci"],
            "tested_on_Triton_auc": item["triton_auc"],
            "tested_on_Triton_ci": item["triton_ci"],
            "difference_auc": item["difference_auc"],
            "difference_ci": item["difference_ci"],
        })
    table_df = pd.DataFrame(table)
    print("\nTable 1: test-set camera AUC comparison")
    print(table_df.to_string(index=False))
    return table_df


def _print_stage2_table_2(agreement_rows: list[dict]) -> pd.DataFrame:
    """Print the same-eye agreement table for experiment A by vector set."""
    table = pd.DataFrame(agreement_rows)
    print("\nTable 2: same-eye agreement (experiment A)")
    print(table.to_string(index=False))
    return table


def _run_downstream_stage_2(
    condition_concept_id: int = 4317977,
    condition_label: str = "cataract",
    stage: int = 2,
) -> pd.DataFrame:
    """Run the Stage 2 downstream test-set evaluation using the exact Stage 1 models and settings."""
    if stage != 2:
        raise ValueError("This file implements downstream condition Stage 2 via --stage 2.")

    normalized_label = _normalize_condition_label(condition_label)
    effective_concept_id = int(condition_concept_id)
    if effective_concept_id != DOWNSTREAM_CONDITION_MAP[normalized_label]["concept_id"]:
        effective_concept_id = DOWNSTREAM_CONDITION_MAP[normalized_label]["concept_id"]

    summary_path = os.path.join(PAIRED_RESULTS_FOLDER, f"downstream_stage1_{normalized_label}_{effective_concept_id}.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Stage 1 summary not found: {summary_path}")

    stage1_summary = pd.read_csv(summary_path)
    if stage1_summary.empty:
        raise ValueError("Stage 1 summary is empty; Stage 2 cannot proceed.")
    expected_lookup = {
        ("pretrained", "A_both"): 0.8412,
        ("pretrained", "B_maestro2_only"): 0.8260,
        ("pretrained", "C_triton_only"): 0.8464,
        ("fine_tuned", "A_both"): 0.8525,
        ("fine_tuned", "B_maestro2_only"): 0.8181,
        ("fine_tuned", "C_triton_only"): 0.8683,
    }
    _validate_stage1_reproduction(stage1_summary, expected_lookup)

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
        condition_name=normalized_label,
    )
    paired_kept = paired_df.merge(
        label_df[["person_id", f"{normalized_label}_label"]].rename(columns={"person_id": "participant_id"}),
        on="participant_id",
        how="inner",
    )

    experiment_map = {
        "A_both": "both",
        "B_maestro2_only": "Maestro2",
        "C_triton_only": "Triton",
    }
    model_rows = []
    for _, row in stage1_summary.iterrows():
        vector_set = str(row["vector_set"])
        experiment = str(row["experiment"])
        if vector_set not in {"pretrained", "fine_tuned"}:
            continue
        model_rows.append({
            "vector_set": vector_set,
            "experiment": experiment,
            "experiment_label": experiment_map.get(experiment, experiment),
            "best_params": row["best_params"],
            "best_iteration": row.get("best_iteration"),
            "val_auc": float(row["val_auc"]),
        })

    all_test_predictions = []
    all_summary_rows = []
    agreement_rows = []
    test_table_rows = []
    same_eye_results = {}

    for vector_name in ["pretrained", "fine_tuned"]:
        latent_path = "/home/s440308/Documents/20_github/xgboost_project/data/results_pre-trained_20250703.csv" if vector_name == "pretrained" else "/home/s440308/Documents/20_github/xgboost_project/data/results_fine-tuned_20250703.csv"
        latent_df = pd.read_csv(latent_path)
        camera_df = _build_paired_camera_dataset(latent_df, paired_kept[["participant_id", "laterality", "recommended_split", "macula", "macula_6x6"]])
        camera_df = camera_df.merge(
            label_df[["person_id", f"{normalized_label}_label"]].rename(columns={"person_id": "participant_id"}),
            on="participant_id",
            how="inner",
        )
        camera_df["target_label"] = camera_df[f"{normalized_label}_label"].astype(int)

        for experiment_name in ["A_both", "B_maestro2_only", "C_triton_only"]:
            model_row = next((m for m in model_rows if m["vector_set"] == vector_name and m["experiment"] == experiment_name), None)
            if model_row is None:
                continue
            train_df = camera_df[(camera_df["recommended_split"] == "train") & _downstream_experiment_mask(camera_df, experiment_name)].copy()
            val_df = camera_df[(camera_df["recommended_split"] == "val") & _downstream_experiment_mask(camera_df, experiment_name)].copy()
            test_df = camera_df[(camera_df["recommended_split"] == "test")].copy()
            if train_df.empty or val_df.empty or test_df.empty:
                continue

            feature_cols = [column for column in camera_df.columns if column.startswith("feature_")]
            X_train = train_df[feature_cols].astype(float)
            X_val = val_df[feature_cols].astype(float)
            y_train = train_df["target_label"].astype(int).to_numpy()
            y_val = val_df["target_label"].astype(int).to_numpy()
            g_train = train_df["participant_id"].astype(int).to_numpy()

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            model = _stage2_model_from_summary(pd.Series(model_row))
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_prob = model.predict_proba(X_val)[:, 1]
            threshold = _youden_j_threshold(y_val, val_prob)

            for camera_name in ["Maestro2", "Triton"]:
                camera_test_df = _coerce_camera_test_df(test_df, camera_name=camera_name, experiment_name=experiment_name, vector_name=vector_name, model_label=f"{vector_name}_{experiment_name}")
                if camera_test_df.empty:
                    continue
                X_test = camera_test_df[feature_cols].astype(float)
                y_test = camera_test_df["target_label"].astype(int).to_numpy()
                participant_ids = camera_test_df["participant_id"].astype(int).to_numpy()
                prob = model.predict_proba(X_test)[:, 1]
                pred = (prob >= threshold).astype(int)
                balanced_acc = float((pred == y_test).mean()) if len(y_test) else np.nan
                tp = np.sum((pred == 1) & (y_test == 1))
                fn = np.sum((pred == 0) & (y_test == 1))
                tn = np.sum((pred == 0) & (y_test == 0))
                fp = np.sum((pred == 1) & (y_test == 0))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                balanced_accuracy = (sensitivity + specificity) / 2.0 if (sensitivity + specificity) > 0 else 0.0
                accuracy = float((pred == y_test).mean()) if len(y_test) else np.nan
                auc_score = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) >= 2 else float("nan")
                auc_low, auc_high, n_valid_boot, skipped = _bootstrap_auc_with_resample(y_test, prob, participant_ids, n_boot=2000, seed=PAIRED_RANDOM_SEED)
                row_summary = {
                    "vector_set": vector_name,
                    "experiment": experiment_name,
                    "trained_on": experiment_map.get(experiment_name, experiment_name),
                    "camera": camera_name,
                    "n_test_eyes": int(len(camera_test_df)),
                    "n_participants": int(camera_test_df["participant_id"].nunique()),
                    "n_positive_participants": int(camera_test_df.loc[camera_test_df["target_label"] == 1, "participant_id"].nunique()),
                    "auc": auc_score,
                    "auc_ci_low": auc_low,
                    "auc_ci_high": auc_high,
                    "balanced_accuracy": balanced_accuracy,
                    "accuracy": accuracy,
                    "sensitivity": float(sensitivity),
                    "specificity": float(specificity),
                    "threshold": float(threshold),
                    "bootstrap_replicates_used": int(n_valid_boot),
                    "bootstrap_replicates_skipped": int(skipped),
                }
                all_summary_rows.append(row_summary)

                for idx, row in camera_test_df.iterrows():
                    all_test_predictions.append({
                        "id": row.get("image_filename") or row.get("filename") or row.name,
                        "participant_id": int(row["participant_id"]),
                        "eye": row.get("laterality"),
                        "filename": row.get("image_filename") or row.get("filename"),
                        "camera": camera_name,
                        "split": "test",
                        "label": int(row["target_label"]),
                        "probability": float(prob[list(camera_test_df.index).index(row.name)]),
                        "experiment": experiment_name,
                        "vector_set": vector_name,
                    })

                if experiment_name == "A_both":
                    pair_df = camera_test_df[["participant_id", "laterality", "target_label"]].copy()
                    pair_df["probability"] = prob
                    pair_df["camera"] = camera_name
                    if camera_name == "Maestro2":
                        maestro_pair = pair_df.rename(columns={"probability": "maestro_prob", "target_label": "maestro_label"})
                    else:
                        triton_pair = pair_df.rename(columns={"probability": "triton_prob", "target_label": "triton_label"})

                    if "maestro_pair" in locals() and "triton_pair" in locals():
                        same_eye_df = maestro_pair.merge(triton_pair[["participant_id", "laterality", "triton_prob", "triton_label"]], on=["participant_id", "laterality"], how="inner")
                        if not same_eye_df.empty:
                            pearson_r = same_eye_df[["maestro_prob", "triton_prob"]].corr(method="pearson").iloc[0, 1]
                            spearman_r = same_eye_df[["maestro_prob", "triton_prob"]].corr(method="spearman").iloc[0, 1]
                            mad = float(np.abs(same_eye_df["maestro_prob"] - same_eye_df["triton_prob"]).mean())
                            same_class_valid = float((same_eye_df["maestro_prob"] >= threshold).astype(int).eq((same_eye_df["triton_prob"] >= threshold).astype(int)).mean())
                            same_class_05 = float((same_eye_df["maestro_prob"] >= 0.5).astype(int).eq((same_eye_df["triton_prob"] >= 0.5).astype(int)).mean())
                            same_eye_results[vector_name] = {
                                "vector_set": vector_name,
                                "pearson_r": float(pearson_r),
                                "spearman_r": float(spearman_r),
                                "mean_abs_diff": mad,
                                "same_predicted_class_at_validation_threshold_pct": same_class_valid,
                                "same_predicted_class_at_0_5_pct": same_class_05,
                                "n_eyes": int(len(same_eye_df)),
                                "validation_threshold": float(threshold),
                            }

    for vector_name in ["pretrained", "fine_tuned"]:
        for experiment_name in ["A_both", "B_maestro2_only", "C_triton_only"]:
            exp_rows = [r for r in all_summary_rows if r["vector_set"] == vector_name and r["experiment"] == experiment_name]
            if len(exp_rows) < 2:
                continue
            if experiment_name == "A_both":
                maestro = next((r for r in exp_rows if r["camera"] == "Maestro2"), None)
                triton = next((r for r in exp_rows if r["camera"] == "Triton"), None)
                if maestro and triton:
                    diff = float(triton["auc"]) - float(maestro["auc"])
                    m_a = np.asarray([row["probability"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Maestro2"], dtype=float)
                    y_a = np.asarray([row["label"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Maestro2"], dtype=int)
                    p_a = np.asarray([row["participant_id"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Maestro2"], dtype=int)
                    m_b = np.asarray([row["probability"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Triton"], dtype=float)
                    y_b = np.asarray([row["label"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Triton"], dtype=int)
                    p_b = np.asarray([row["participant_id"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == "Triton"], dtype=int)
                    diff_low, diff_high, diff_n, diff_skipped = _bootstrap_camera_auc_difference(y_b, m_b, p_b, y_a, m_a, p_a, n_boot=2000, seed=PAIRED_RANDOM_SEED)
                    test_table_rows.append({
                        "vector_set": vector_name,
                        "trained_on": "both",
                        "maestro2_auc": maestro["auc"],
                        "maestro2_ci": f"[{maestro['auc_ci_low']:.4f}, {maestro['auc_ci_high']:.4f}]" if pd.notna(maestro.get("auc_ci_low")) and pd.notna(maestro.get("auc_ci_high")) else "n/a",
                        "triton_auc": triton["auc"],
                        "triton_ci": f"[{triton['auc_ci_low']:.4f}, {triton['auc_ci_high']:.4f}]" if pd.notna(triton.get("auc_ci_low")) and pd.notna(triton.get("auc_ci_high")) else "n/a",
                        "difference_auc": diff,
                        "difference_ci": f"[{diff_low:.4f}, {diff_high:.4f}]",
                        "difference_bootstrap_replicates_used": int(diff_n),
                        "difference_bootstrap_replicates_skipped": int(diff_skipped),
                    })
            else:
                camera_a = next((r for r in exp_rows if r["camera"] == ("Maestro2" if experiment_name == "B_maestro2_only" else "Triton")), None)
                camera_b = next((r for r in exp_rows if r["camera"] == ("Triton" if experiment_name == "B_maestro2_only" else "Maestro2")), None)
                if camera_a and camera_b:
                    diff = float(camera_a["auc"]) - float(camera_b["auc"])
                    camera_a_name = "Maestro2" if experiment_name == "B_maestro2_only" else "Triton"
                    camera_b_name = "Triton" if experiment_name == "B_maestro2_only" else "Maestro2"
                    m_a = np.asarray([row["probability"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_a_name], dtype=float)
                    y_a = np.asarray([row["label"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_a_name], dtype=int)
                    p_a = np.asarray([row["participant_id"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_a_name], dtype=int)
                    m_b = np.asarray([row["probability"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_b_name], dtype=float)
                    y_b = np.asarray([row["label"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_b_name], dtype=int)
                    p_b = np.asarray([row["participant_id"] for row in all_test_predictions if row["vector_set"] == vector_name and row["experiment"] == experiment_name and row["camera"] == camera_b_name], dtype=int)
                    diff_low, diff_high, diff_n, diff_skipped = _bootstrap_camera_auc_difference(y_a, m_a, p_a, y_b, m_b, p_b, n_boot=2000, seed=PAIRED_RANDOM_SEED)
                    test_table_rows.append({
                        "vector_set": vector_name,
                        "trained_on": experiment_map.get(experiment_name, experiment_name),
                        "maestro2_auc": camera_a["auc"] if camera_a["camera"] == "Maestro2" else camera_b["auc"],
                        "maestro2_ci": f"[{(camera_a if camera_a['camera']=='Maestro2' else camera_b)['auc_ci_low']:.4f}, {(camera_a if camera_a['camera']=='Maestro2' else camera_b)['auc_ci_high']:.4f}]" if pd.notna((camera_a if camera_a['camera']=='Maestro2' else camera_b).get('auc_ci_low')) and pd.notna((camera_a if camera_a['camera']=='Maestro2' else camera_b).get('auc_ci_high')) else "n/a",
                        "triton_auc": camera_b["auc"] if camera_b["camera"] == "Triton" else camera_a["auc"],
                        "triton_ci": f"[{(camera_b if camera_b['camera']=='Triton' else camera_a)['auc_ci_low']:.4f}, {(camera_b if camera_b['camera']=='Triton' else camera_a)['auc_ci_high']:.4f}]" if pd.notna((camera_b if camera_b['camera']=='Triton' else camera_a).get('auc_ci_low')) and pd.notna((camera_b if camera_b['camera']=='Triton' else camera_a).get('auc_ci_high')) else "n/a",
                        "difference_auc": diff,
                        "difference_ci": f"[{diff_low:.4f}, {diff_high:.4f}]",
                        "difference_bootstrap_replicates_used": int(diff_n),
                        "difference_bootstrap_replicates_skipped": int(diff_skipped),
                    })

    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_path = os.path.join(PAIRED_RESULTS_FOLDER, f"cataract_{effective_concept_id}_stage2_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved downstream Stage 2 summary to: {summary_path}")
        print("\nStage 2 per-camera summary")
        print(summary_df[[
            "vector_set",
            "experiment",
            "trained_on",
            "camera",
            "n_test_eyes",
            "n_participants",
            "n_positive_participants",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "balanced_accuracy",
            "accuracy",
            "sensitivity",
            "specificity",
            "threshold",
            "bootstrap_replicates_used",
            "bootstrap_replicates_skipped",
        ]].to_string(index=False))

    if all_test_predictions:
        pred_path = os.path.join(PAIRED_RESULTS_FOLDER, f"cataract_{effective_concept_id}_stage2_test_predictions.csv")
        pd.DataFrame(all_test_predictions).to_csv(pred_path, index=False)
        print(f"Saved stage-2 per-image predictions to: {pred_path}")

    if same_eye_results:
        agreement_table = pd.DataFrame(list(same_eye_results.values()))
        print("\nTable 2: same-eye agreement (experiment A)")
        print(agreement_table[[
            "vector_set",
            "pearson_r",
            "spearman_r",
            "mean_abs_diff",
            "same_predicted_class_at_validation_threshold_pct",
            "same_predicted_class_at_0_5_pct",
            "n_eyes",
            "validation_threshold",
        ]].to_string(index=False))

    if test_table_rows:
        table_df = pd.DataFrame(test_table_rows)
        print("\nTable 1: test-set camera AUC comparison")
        print(table_df[[
            "vector_set",
            "trained_on",
            "maestro2_auc",
            "maestro2_ci",
            "triton_auc",
            "triton_ci",
            "difference_auc",
            "difference_ci",
            "difference_bootstrap_replicates_used",
            "difference_bootstrap_replicates_skipped",
        ]].to_string(index=False))

    return pd.DataFrame(all_summary_rows)


def run_downstream_condition_stage_2(
    condition_concept_id: int = 4317977,
    condition_label: str = "cataract",
    stage: int = 2,
) -> pd.DataFrame:
    """Run the Stage 2 downstream test-set evaluation with logging and saved-run manifests."""
    result = run_with_logging(_run_downstream_stage_2, condition_concept_id, condition_label, stage)
    print("\nSaved downstream Stage 2 analysis with paired_analyses prefix in saved_runs.")
    return result


def _handle_cli() -> None:
    """Parse CLI args for paired analyses and downstream condition prediction."""
    parser = argparse.ArgumentParser(description="Paired-eye analysis workflows")
    parser.add_argument("--analysis", choices=["pairing", "camera", "downstream"], default="pairing")
    parser.add_argument("--label", default="cataract", help="Downstream condition label: cataract or t2dm")
    parser.add_argument("--stage", type=int, default=1, help="Downstream stage to run; only stage 1 is implemented in this file")
    args = parser.parse_args()

    if args.analysis == "downstream":
        if args.stage == 1:
            run_downstream_condition_stage_1(
                condition_concept_id=DOWNSTREAM_CONDITION_MAP[_normalize_condition_label(args.label)]["concept_id"],
                condition_label=_normalize_condition_label(args.label),
                stage=args.stage,
            )
            return
        if args.stage == 2:
            run_downstream_condition_stage_2(
                condition_concept_id=DOWNSTREAM_CONDITION_MAP[_normalize_condition_label(args.label)]["concept_id"],
                condition_label=_normalize_condition_label(args.label),
                stage=args.stage,
            )
            return
        raise ValueError("This file only supports downstream Stage 1 and Stage 2.")

    if args.analysis == "pairing":
        run_paired_analyses_cli()
        return

    if args.analysis == "camera":
        run_step_2_classifier()
        return


if __name__ == "__main__":
    _handle_cli()
