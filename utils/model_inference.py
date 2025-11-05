import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

def proportions_from_edges(arr, edges):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.full(len(edges)-1, 1.0/(len(edges)-1))
    counts, _ = np.histogram(arr, bins=np.array(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return np.full_like(counts, 1.0 / len(counts), dtype=float)
    return counts / total

def psi_from_expected(expected, actual):
    eps = 1e-8
    expected = np.clip(np.asarray(expected, dtype=float), eps, 1)
    actual   = np.clip(np.asarray(actual,   dtype=float), eps, 1)
    return float(np.sum((actual - expected) * np.log(actual / expected)))

def main(snapshotdate, modeluri=None, modelname=None, demo=True):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "/opt/airflow/model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"] if modelname else config["model_bank_directory"] + modeluri
    
    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:/opt/mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Resolve model URI
    model_uri = modeluri
    print(f"Using MLflow model_uri = {model_uri}")
    model = None
    deploy_mode = None
    # --- load model artefact from model bank ---
    if model_uri:
        try:
            print(f"Loading model from MLflow: {modeluri}")
            model = mlflow.sklearn.load_model(modeluri)  # pipeline (scaler + xgb) logged in training
            deploy_mode = "mlflow"
        except Exception as e:
            print(f"MLflow load failed: {e}")

    if model is None:
        # fallback to local pickle
        if not modelname:
            raise ValueError("No MLflow model and no --modelname provided for pickle fallback.")
        print(f'Loading model from pickle: {config["model_artefact_filepath"]}')
        with open(config["model_artefact_filepath"], "rb") as file:
            model_artefact = pickle.load(file)
        model = model_artefact["model"]  # pipeline
        deploy_mode = "pickle"

    # resolve monitoring baselines (bins + expected shares) 
    monitoring_bins = None
    if deploy_mode == "pickle":
        monitoring_bins = model_artefact.get("monitoring_bins")
    else:
        # try to fetch psi_bins.json from the producing run via Model Registry
        try:
            client = MlflowClient()
            if model_uri.startswith("models:/"):
                parts = model_uri.split("/")
                name = parts[1]
                last = parts[2]
                if last.isdigit():
                    # models:/name/<version>
                    mv = client.get_model_version(name=name, version=last)
                    run_id = mv.run_id
                else:
                    # models:/name/<stage>
                    vers = client.get_latest_versions(name=name, stages=[last])
                    run_id = vers[0].run_id if vers else None
                if run_id:
                    tmp_dir = "/tmp/_bins_artifacts"
                    os.makedirs(tmp_dir, exist_ok=True)
                    local_path = client.download_artifacts(run_id, "psi_bins.json", tmp_dir)
                    import json
                    with open(local_path, "r") as f:
                        monitoring_bins = json.load(f)
        except Exception as err:
            print(f"Could not load psi_bins.json from MLflow: {err}")
    
    #==========================
    # Feature store
    #==========================
    
    features_folder_path = "/opt/airflow/datamart/gold/feature_store/cust_fin_risk/"
    features_files_list = [features_folder_path+os.path.basename(f) for f in glob.glob(os.path.join(features_folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*features_files_list)
    # for demo
    if demo:
        label_folder_path = "/opt/airflow/datamart/gold/label_store/"
        label_files_list = [label_folder_path+os.path.basename(f) for f in glob.glob(os.path.join(label_folder_path, '*'))]
        label_store_sdf = spark.read.option("header", "true").parquet(*label_files_list)
        date_ID_to_join = label_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
        date_ID_to_join = date_ID_to_join.select('Customer_ID','snapshot_date')
        features_sdf = date_ID_to_join.join(features_store_sdf, "Customer_ID", "left")
    else:
        features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    
    # extract feature store
    print("extracted features_sdf rows:", features_sdf.count(), config["snapshot_date"])
    features_pdf = features_sdf.toPandas()

    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = [col for col in features_pdf.columns if col not in ('Customer_ID', 'snapshot_date', 'label', 'label_def', 'loan_id')]
    X_inference = features_pdf[feature_cols]    
    print('X_inference', X_inference.shape[0])

    # --- model prediction inference ---    
    # predict using model pipeline
    y_inference = model.predict_proba(X_inference)[:, 1]
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID"]].copy()
    y_inference_pdf["model_predictions"] = y_inference
    if deploy_mode == "mlflow":
        y_inference_pdf["model_name"] = model_uri #for traceability (registry URI)
    else:
        y_inference_pdf["model_name"] = modelname

    # --- save model inference to datamart gold table ---
    # create gold datalake
    if deploy_mode == "pickle":
        model_id = config["model_name"][:-4]
    else:
        model_id = model_uri.replace("models:/", "").replace("/", "_")

    gold_directory = f"/opt/airflow/datamart/gold/model_predictions/{model_id}/" 

    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = model_id + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    # MLflow run and push to Prometheus for metric storage
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_default_baseline")
    mlflow.set_experiment(experiment_name)
    run_name = f"infer_{config['snapshot_date_str']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("snapshot_date", config["snapshot_date_str"])
        mlflow.log_param("deploy_mode", deploy_mode)
        mlflow.log_param("model_identifier", model_uri if deploy_mode=="mlflow" else config["model_name"])
        mlflow.log_metric("n_scored", int(X_inference.shape[0]))

        # Score summary
        q = np.percentile(y_inference, [1,5,10,50,90,95,99]).tolist()
        mlflow.log_metrics({
            "score_mean": float(np.mean(y_inference)),
            "score_std":  float(np.std(y_inference)),
            "score_min":  float(np.min(y_inference)),
            "score_p01":  q[0], "score_p05": q[1], "score_p10": q[2],
            "score_p50":  q[3], "score_p90": q[4], "score_p95": q[5], "score_p99": q[6],
            "score_max":  float(np.max(y_inference)),
        })

        # Score PSI & Feature CSI (vs training baseline bins)
        psi_score = None
        per_feature_psi = None
        if monitoring_bins:
            sbins = monitoring_bins.get("score_bins")
            sexp  = monitoring_bins.get("score_expected")
            if sbins and sexp:
                sact = proportions_from_edges(y_inference, sbins).tolist()
                psi_score = psi_from_expected(sexp, sact)
                mlflow.log_metric("psi_score_train_vs_infer", psi_score)

            ncols = monitoring_bins.get("numeric_cols", [])
            fbins = monitoring_bins.get("feature_bins", {})
            fexp  = monitoring_bins.get("feature_expected", {})
            vals = []
            if ncols and fbins and fexp:
                infer_pdf = features_pdf
                for c in ncols:
                    if c in infer_pdf.columns and c in fbins and c in fexp:
                        a = proportions_from_edges(infer_pdf[c].values, fbins[c]).tolist()
                        e = fexp[c]
                        vals.append(psi_from_expected(e, a))
                if vals:
                    per_feature_psi = vals
                    mlflow.log_metric("csi_mean_train_vs_infer", float(np.nanmean(vals)))
                    mlflow.log_metric("csi_max_train_vs_infer",  float(np.nanmax(vals)))

                    psi_breach = (psi_score is not None and psi_score >= 0.25) #if value exceeds 0.25
                    csi_breach = (per_feature_psi is not None and float(np.nanmax(per_feature_psi)) >= 0.25) #for per feature

                    mlflow.set_tag("psi_breach", str(psi_breach))
                    mlflow.set_tag("csi_breach", str(csi_breach))
                    
                    snapshot_fmt = config["snapshot_date_str"].replace("-", "_")
                    drift_payload = {
                        "snapshot_date": config["snapshot_date_str"],
                        "model_id": model_id,
                        "psi_score": float(psi_score) if psi_score is not None else None,
                        "csi_max": float(np.nanmax(per_feature_psi)) if per_feature_psi else None,
                        "psi_breach": bool(psi_breach),
                        "csi_breach": bool(csi_breach),
                        "psi_threshold": float(os.getenv("MONITOR_PSI_MAX", "0.25")),
                        "csi_threshold": float(os.getenv("MONITOR_CSI_MAX", "0.25")),
                    }
                    # save next to predictions so Airflow can read without MLflow APIs
                    drift_file = os.path.join(gold_directory, f"drift_{snapshot_fmt}.json")
                    with open(drift_file, "w") as f:
                        import json as _json
                        _json.dump(drift_payload, f)
                    print("Wrote drift sidecar:", drift_file)
            
        # Supervised metrics (if label present now)
        auc = None; gini = None
        if "label" in features_pdf.columns and features_pdf["label"].notna().any():
            y_true = features_pdf["label"].astype(int).values
            if y_true.shape[0] == y_inference.shape[0]:
                auc = roc_auc_score(y_true, y_inference)
                gini = 2*auc - 1
                mlflow.log_metric("auc_infer", float(auc))
                mlflow.log_metric("gini_infer", float(gini))

        # Push to Prometheus Pushgateway for visualisation on Grafana (Pushgateway for short-lived batch jobs)
        PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "http://pushgateway:9091")
        reg = CollectorRegistry()
        labels = {"snapshot": config["snapshot_date_str"], "model": model_id}
        g_score_mean = Gauge("ml_score_mean", "Mean model score (monthly batch)", labels.keys(), registry=reg)
        g_psi_score  = Gauge("ml_psi_score", "Score PSI vs training", labels.keys(), registry=reg)
        g_csi_max    = Gauge("ml_csi_max", "Max feature CSI vs training", labels.keys(), registry=reg)
        g_auc        = Gauge("ml_auc", "AUC this month (if labels available)", labels.keys(), registry=reg)
        g_gini       = Gauge("ml_gini", "Gini this month (if labels available)", labels.keys(), registry=reg)
        g_psi_thr    = Gauge("ml_psi_threshold", "PSI threshold", labels.keys(), registry=reg)
        g_csi_thr    = Gauge("ml_csi_threshold", "CSI threshold", labels.keys(), registry=reg)
        psi_thr = float(os.getenv("MONITOR_PSI_MAX", "0.25"))
        csi_thr = float(os.getenv("MONITOR_CSI_MAX", "0.25"))
        g_psi_thr.labels(**labels).set(psi_thr)
        g_csi_thr.labels(**labels).set(csi_thr)   
        
        g_score_mean.labels(**labels).set(float(np.mean(y_inference)))
        if psi_score is not None:
            g_psi_score.labels(**labels).set(float(psi_score))
        if per_feature_psi:
            g_csi_max.labels(**labels).set(float(np.nanmax(per_feature_psi)))
        if auc is not None:
            g_auc.labels(**labels).set(float(auc))
        if gini is not None:
            g_gini.labels(**labels).set(float(gini))

        # push replaces prior sample for same job+labels
        push_to_gateway(PUSHGATEWAY_ADDR, job="loan_default_infer", registry=reg)
    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--modeluri", type=str, help="MLflow URI, e.g. models:/loan_default_pipeline/Production")
    group.add_argument("--modelname", type=str, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modeluri , args.modelname)
