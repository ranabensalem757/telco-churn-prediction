import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mlruns_path = args.mlflow_uri or f"file://{project_root}mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        df= load_data(args.input)
        print(f"data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality", int(is_valid))
        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow")

        df = preprocess_data(df)
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        df.to_csv(processed_path, index=False)
        print(f"shape : {df.shape}")

        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        #building features
        df_enc = build_features(df, target_col=target)
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        with open(os.path.join(artifacts_dir, "feature_columns.json"),"w") as f:
            json.dump(feature_cols, f)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        preprocessing_artifact ={
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir,"preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir,"preprocessing.pkl"))
        #training
        X=df_enc.drop(columns=[target])
        y=df_enc[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X ,y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"train : {X_train.shape[0]} samples, test {X_test.shape[0]} samples")
        scale_pos_weight = (y_train == 0).sum() / (y_train ==1).sum()
        print(f"class imbalance ratio to positive class : {scale_pos_weight}")
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"model trained in {train_time:.2f} seconds")
        
        #evaluation
        t1 = time.time()
        proba = model.predict_proba(X_test)[:,1] #extracts only the churn probability for each customer
        y_pred = (proba >= args.threshold) .astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)
        precision = precision_score(y_test, y_pred) # how many actually positive among those predicted positive
        recall = recall_score(y_test, y_pred) # how many did we catch among the actual positive
        f1 = f1_score(y_test, y_pred) #harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, y_pred) #area under the ROC curve
        mlflow.log_metrics("precision", precision)
        mlflow.log_metrics("recall", recall)
        mlflow.log_metrics("f1", f1)
        mlflow.log_metrics("roc_auc", roc_auc)
        print(f"precision : {precision:.4f}, recall : {recall:.4f}")
        print(f" f1 : {f1:.4f}, roc_auc : {roc_auc:.4f}")
        #saving the model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("model saved to MLflow")
        




