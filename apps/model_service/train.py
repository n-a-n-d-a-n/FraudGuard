import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Configure MLflow SQLite tracking backend to enable MLflow Model Registry
DB_PATH = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(DB_PATH)


FEATURE_COLUMNS = [
    "customer_txn_count_60m",
    "amount_deviation_ratio",
    "is_new_device",
    "shared_device_account_count"
]


def load_data(file_path: str = "dummy_data/train_dummy.csv"):
    """
    Load synthetic transaction training data from CSV matching Member 3 feature schema.
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Map legacy columns if present for backwards compatibility
    if "velocity_1h" in df.columns:
        df = df.rename(columns={
            "velocity_1h": "customer_txn_count_60m",
            "amount_deviation": "amount_deviation_ratio",
            "new_device": "is_new_device",
            "shared_device_count": "shared_device_account_count"
        })

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]
    return X, y


def run_training_pipeline(data_path: str = "dummy_data/train_dummy.csv"):
    """
    Executes end-to-end model training, evaluation, probability calibration,
    MLflow Model Registry registration, and local joblib export.
    """
    # 1. Load data
    X, y = load_data(data_path)

    # 2. Split train/test (80/20 stratified on target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split complete: Train shape {X_train.shape}, Test shape {X_test.shape}")

    # Set up MLflow experiment
    mlflow.set_experiment("fraudguard360_detection")

    # Define model configurations
    models_config = {
        "baseline_logreg": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "log_fn": mlflow.sklearn.log_model,
            "params": {"model_type": "LogisticRegression", "max_iter": 1000, "random_state": 42}
        },
        "xgboost_v1": {
            "model": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            ),
            "log_fn": mlflow.xgboost.log_model,
            "params": {
                "model_type": "XGBClassifier",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42
            }
        },
        "lightgbm_v1": {
            "model": lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            "log_fn": mlflow.lightgbm.log_model,
            "params": {
                "model_type": "LGBMClassifier",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "random_state": 42
            }
        }
    }

    results = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("STARTING BENCHMARK MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 3. Train and evaluate initial uncalibrated candidate models
    for run_name, config in models_config.items():
        print(f"\n---> Training Model: {run_name}")
        model = config["model"]

        # Fit model on training data
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred)

        print(f"Metrics for {run_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"\nClassification Report ({run_name}):\n{report}")

        results.append({
            "model_name": run_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })
        trained_models[run_name] = model

        # Log individual candidate runs to MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(config["params"])
            mlflow.log_metrics({
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            })
            config["log_fn"](model, artifact_path="model")

    # Print candidate summary table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                    UNCALIBRATED CANDIDATE SUMMARY                      ")
    print("=" * 80)
    print(results_df.to_string(index=False, formatters={
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1": "{:.4f}".format,
        "roc_auc": "{:.4f}".format
    }))
    print("=" * 80)

    # 4. Best Model Selection & Probability Calibration
    best_row = results_df.loc[results_df["roc_auc"].idxmax()]
    best_model_name = best_row["model_name"]
    uncal_roc_auc = best_row["roc_auc"]
    uncalibrated_best_model = trained_models[best_model_name]

    print(f"\nSelected Best Candidate by ROC AUC: '{best_model_name}' (ROC AUC = {uncal_roc_auc:.4f})")
    print("\n" + "=" * 60)
    print("APPLYING PROBABILITY CALIBRATION (CalibratedClassifierCV, sigmoid, cv=5)")
    print("=" * 60)

    # Wrap best model in Platt Scaling calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=uncalibrated_best_model,
        method="sigmoid",
        cv=5
    )
    calibrated_model.fit(X_train, y_train)

    # Re-evaluate calibrated model on test set
    y_pred_cal = calibrated_model.predict(X_test)
    y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

    cal_precision = precision_score(y_test, y_pred_cal)
    cal_recall = recall_score(y_test, y_pred_cal)
    cal_f1 = f1_score(y_test, y_pred_cal)
    cal_roc_auc = roc_auc_score(y_test, y_proba_cal)

    print("\n--- UNCALIBRATED vs CALIBRATED MODEL METRIC COMPARISON ---")
    print(f"Base Model:             {best_model_name}")
    print(f"Uncalibrated Metrics -> Precision: {best_row['precision']:.4f} | Recall: {best_row['recall']:.4f} | F1: {best_row['f1']:.4f} | ROC AUC: {uncal_roc_auc:.4f}")
    print(f"Calibrated Metrics   -> Precision: {cal_precision:.4f} | Recall: {cal_recall:.4f} | F1: {cal_f1:.4f} | ROC AUC: {cal_roc_auc:.4f}")

    # 5. MLflow Model Registry Logging & Transition
    REGISTERED_MODEL_NAME = "fraudguard360-detector"

    print("\n" + "=" * 60)
    print(f"LOGGING CALIBRATED MODEL TO MLFLOW REGISTRY: '{REGISTERED_MODEL_NAME}'")
    print("=" * 60)

    with mlflow.start_run(run_name=f"{best_model_name}_calibrated") as run:
        mlflow.log_params({
            "base_model": best_model_name,
            "calibration_method": "sigmoid",
            "cv_folds": 5,
            "is_calibrated": True
        })
        mlflow.log_metrics({
            "precision": cal_precision,
            "recall": cal_recall,
            "f1_score": cal_f1,
            "roc_auc": cal_roc_auc
        })

        # CalibratedClassifierCV wraps the base estimator in sklearn-internal calibration classes
        # that aren't on skops' default trusted-type allowlist. Since we generated this model
        # ourselves in this training run, it is safe to explicitly trust these specific types.
        skops_trusted_types = [
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration"
        ]

        # Register model artifact with MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=skops_trusted_types
        )

    # Transition registered model version to 'Staging' using MlflowClient
    client = MlflowClient()
    latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
    latest_version_info = latest_versions[-1]
    version_num = latest_version_info.version

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        stage="Staging"
    )

    print(f"\n[MLflow Registry Success]")
    print(f"  Registered Model Name: {REGISTERED_MODEL_NAME}")
    print(f"  Version:               {version_num}")
    print(f"  Stage:                 Staging")

    # 6. Save Calibrated Model locally as Fallback
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(calibrated_model, joblib_path)
    print(f"\nSaved calibrated fallback model locally to '{joblib_path}'.")


if __name__ == "__main__":
    run_training_pipeline()
