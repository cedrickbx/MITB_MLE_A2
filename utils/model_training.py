import os
import glob
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
import argparse
from urllib.parse import urlparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.datasets import make_classification

#For MLFlow
import json
import tempfile
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

def proportions(arr, bins):
    counts, _ = np.histogram(arr, bins=bins)
    total = counts.sum()
    if total == 0:
        # avoid zero division and return uniform tiny mass
        return np.full_like(counts, 1.0 / len(counts), dtype=float)
    return counts / total

def psi_numeric(base, target, bins=10, use_quantiles=True):
    """
    base: 1D numpy array (train)
    target: 1D numpy array (test or oot)
    bins: number of bins
    use_quantiles: True -> derive cutpoints from base quantiles
    """
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)

    # drop missing values for PSI computation; keep it minimal here
    base = base[~np.isnan(base)]
    target = target[~np.isnan(target)]

    if base.size == 0 or target.size == 0:
        return np.nan

    if use_quantiles:
        quartile_bins = np.linspace(0, 1, bins + 1)
        cutpoints = np.unique(np.quantile(base, quartile_bins))
        if len(cutpoints) < 2:
            return 0.0
        bins_edges = cutpoints
    else:
        # use min/max with equal-width bins
        low = min(base.min(), target.min())
        high = max(base.max(), target.max())
        if low == high:
            return 0.0
        bins_edges = np.linspace(low, high, bins + 1)

    expected_prop = proportions(base, bins_edges) #expected proportion per bin 
    actual_prop = proportions(target, bins_edges)

    # clamp tiny values to avoid log(0)
    eps = 1e-8
    expected_prop = np.clip(expected_prop, eps, 1)
    actual_prop = np.clip(actual_prop, eps, 1)

    return (np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop)))

def csi_dataframe(train_df, target_df, numeric_cols, bins=10):
    """
    Compute Characteristic Stability Index between train_df and target_df and return mean & max.
    """
    psis = []
    for c in numeric_cols: #only factor in numeric columns
        try:
            psis.append(psi_numeric(train_df[c].values, target_df[c].values, bins=bins))
        except Exception:
            psis.append(np.nan)
    psis = np.array(psis, dtype=float)
    return float(np.nanmean(psis)), float(np.nanmax(psis))

def model_training(snapshotdate):
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    model_train_date_str = snapshotdate #"2024-09-01" should be the first model training date
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    config["random_state"] = 42

    random_state = config["random_state"]

    #==========================
    # MLFlow setup
    #==========================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)  # local dir or HTTP server
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_default_baseline")
    mlflow.set_experiment(experiment_name)
    # Autolog for sklearn pipelines/CV --> log params, metrics, model, artifacts
    mlflow.sklearn.autolog(log_models=True) #False to disable logging

    #==========================
    # Label store
    #==========================
    label_folder_path = "/opt/airflow/datamart/gold/label_store/"
    label_files_list = [label_folder_path+os.path.basename(f) for f in glob.glob(os.path.join(label_folder_path, '*'))]
    label_store_sdf = spark.read.option("header", "true").parquet(*label_files_list)
    # extract label store
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
    #==========================
    # Feature store
    #==========================
    features_folder_path = "/opt/airflow/datamart/gold/feature_store/cust_fin_risk/"
    features_files_list = [features_folder_path+os.path.basename(f) for f in glob.glob(os.path.join(features_folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*features_files_list)

    #====================================
    # Data Processing for Model Training
    #====================================
    # prepare data for modeling
    data_pdf = labels_sdf.join(features_store_sdf, on=["Customer_ID"], how="left").toPandas()
    # split data into train - test - oot
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]
    feature_cols = [col for col in data_pdf.columns if col not in ('Customer_ID', 'snapshot_date', 'label', 'label_def', 'loan_id')]

    #================================
    # Training and Test data
    #================================
    # split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size= 1 - config["train_test_ratio"],
        random_state=random_state,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )

    #=========================
    # OOT data
    #=========================
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]

    # set up standard scalar preprocessing - make each feature contribute equally
    scaler = StandardScaler()

    #================================
    # Model Training
    #================================
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state)

    #Define pipeline
    xg_pipeline = Pipeline([("standard_scalar", scaler),
                            ("xgclf", xgb_clf)])

    # Define the hyperparameter space to search
    param_dist = {
        'xgclf__n_estimators': [10, 25, 50],
        'xgclf__max_depth': [2, 3, 4],  # lower max_depth to simplify the model
        'xgclf__learning_rate': [0.01, 0.1, 1],
        'xgclf__subsample': [0.6, 0.8],
        'xgclf__colsample_bytree': [0.6, 0.8],
        'xgclf__gamma': [0, 0.1],
        'xgclf__min_child_weight': [1, 3, 5],
        'xgclf__reg_alpha': [0, 0.1, 1],
        'xgclf__reg_lambda': [1, 1.5, 2]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # ===============================================
    # Start MLFlow
    # ===============================================

    run_name = f"train_{config['model_train_date_str']}"
    with mlflow.start_run(run_name=run_name):
        # Log static config as params
        mlflow.log_param("model_train_date_str", config["model_train_date_str"])
        mlflow.log_param("train_test_period_months", config["train_test_period_months"])
        mlflow.log_param("oot_period_months", config["oot_period_months"])
        mlflow.log_param("train_test_ratio", config["train_test_ratio"])
        mlflow.log_param("random_state", config["random_state"])

        # Set up the random search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=xg_pipeline,
            param_distributions=param_dist,
            scoring=auc_scorer,
            n_iter=100,  # Number of iterations for random search
            cv=cv,       # Cross-validation in 3 folds using StratifiedKFold - ensure similar class proportion for each train-validation split 
            verbose=1,
            random_state=42,
            n_jobs=-1,   # Use all available cores
            refit=True
        )
        # Perform the random search
        random_search.fit(X_train, y_train)

        # Output the best parameters and best score
        print("Best parameters found: ", random_search.best_params_)
        print("Best AUC score: ", random_search.best_score_)

        # Evaluate the model on the train set
        best_model = random_search.best_estimator_
        #log hyperparameters of best estimators
        mlflow.log_params({k.replace("xgclf__", "xgb_"): v for k, v in random_search.best_params_.items()})

        # Evaluate the model on the train set
        train_proba = best_model.predict_proba(X_train)[:, 1]
        train_auc_score = roc_auc_score(y_train, train_proba)

        # Evaluate the model on the test set
        test_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc_score = roc_auc_score(y_test, test_proba)
        
        # Evaluate the model on the oot set
        oot_proba = best_model.predict_proba(X_oot)[:, 1]
        oot_auc_score = roc_auc_score(y_oot, oot_proba)

        #Compute gini score
        gini_train = round(2*train_auc_score-1,3)
        gini_test = round(2*test_auc_score-1,3)
        gini_oot = round(2*oot_auc_score-1,3)

        print("Train AUC score: ", train_auc_score)
        print("Test AUC score: ", test_auc_score)
        print("OOT AUC score: ", oot_auc_score)
        print("TRAIN GINI score: ", gini_train)
        print("Test GINI score: ", gini_test)
        print("OOT GINI score: ", gini_oot)

        #log performance metrics to MLFlow
        mlflow.log_metric("auc_train", train_auc_score)
        mlflow.log_metric("auc_test", test_auc_score)
        mlflow.log_metric("auc_oot", oot_auc_score)
        mlflow.log_metric("gini_train", gini_train)
        mlflow.log_metric("gini_test", gini_test)
        mlflow.log_metric("gini_oot", gini_oot)

        # log stability metrics (PSI/CSI) to MLFlow
        # PSI (train vs test and train vs oot), 10 quantile bins from train
        score_bins = np.unique(np.quantile(train_proba, np.linspace(0, 1, 11)))
        psi_score_train_test = psi_numeric(train_proba, test_proba, bins=len(score_bins)-1, use_quantiles=True)
        psi_score_train_oot  = psi_numeric(train_proba, oot_proba,  bins=len(score_bins)-1, use_quantiles=True)
        mlflow.log_metric("psi_score_train_vs_test", float(psi_score_train_test))
        mlflow.log_metric("psi_score_train_vs_oot",  float(psi_score_train_oot))

        # CSI (feature PSI) for numeric features only, mean/max across features
        numeric_cols = [c for c in feature_cols if np.issubdtype(X_train[c].dtype, np.number)]
        csi_mean_tt, csi_max_tt = csi_dataframe(X_train[numeric_cols], X_test[numeric_cols], numeric_cols, bins=10)
        csi_mean_to, csi_max_to = csi_dataframe(X_train[numeric_cols], X_oot[numeric_cols],  numeric_cols, bins=10)
        mlflow.log_metric("csi_mean_train_vs_test", float(csi_mean_tt))
        mlflow.log_metric("csi_max_train_vs_test",  float(csi_max_tt))
        mlflow.log_metric("csi_mean_train_vs_oot",  float(csi_mean_to))
        mlflow.log_metric("csi_max_train_vs_oot",   float(csi_max_to))

        #Logging config and metafeatures
        mlflow.log_dict({k: str(v) for k, v in config.items()}, "config.json")
        mlflow.log_dict({
            "feature_cols": feature_cols,
            "data_stats": {
                "X_train": X_train.shape[0],
                "X_test": X_test.shape[0],
                "X_oot": X_oot.shape[0],
                "y_train": round(y_train.mean(), 2),
                "y_test": round(y_test.mean(), 2),
                "y_oot": round(y_oot.mean(), 2),
                    }
                    }, "meta.json")

        mlflow.sklearn.log_model(sk_model=best_model,artifact_path="model",registered_model_name="loan_default_pipeline")
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id

        # Find the model version that was just created
        versions = client.search_model_versions(f"name='loan_default_pipeline'")
        this_ver = None
        for v in versions:
            if v.run_id == run_id:
                this_ver = v.version
                break

        if this_ver is not None:
            try:
                # Detect file-based registry and skip stage transition
                reg_uri = os.getenv("MLFLOW_REGISTRY_URI", os.getenv("MLFLOW_TRACKING_URI", "file:/opt/mlruns"))
                scheme = urlparse(reg_uri).scheme if reg_uri else "file"
                if scheme == "file":
                    print("Local file registry detected -> skipping stage transition, will select latest version at inference.")
                else:
                    client.transition_model_version_stage(
                        name="loan_default_pipeline",
                        version=this_ver,
                        stage="Production",
                        archive_existing_versions=True,
                    )
            except Exception as e:
                print(f"Skipping stage transition due to registry error: {e}")

        # persist monitoring baselines (bins + expected proportions)
        # Score bins (same as used above) + expected shares from train
        score_expected = proportions(train_proba, score_bins).tolist()

        # Feature bins (10-quantile from X_train) + expected shares from train
        feature_bins = {}
        feature_expected = {}
        for c in numeric_cols:
            colvals = X_train[c].values.astype(float)
            colvals = colvals[~np.isnan(colvals)]
            cuts = np.unique(np.quantile(colvals, np.linspace(0, 1, 11)))
            if len(cuts) < 2:
                # degenerate -> make a single bin edge pair to avoid errors downstream
                cuts = np.array([np.nanmin(colvals), np.nanmax(colvals)]) if colvals.size else np.array([0.0, 1.0])
            feature_bins[c] = cuts.tolist()
            feature_expected[c] = proportions(colvals, cuts).tolist()

        monitoring_payload = {
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "score_bins": score_bins.tolist(),
            "score_expected": score_expected,
            "feature_bins": feature_bins,
            "feature_expected": feature_expected,
        }
        mlflow.log_dict(monitoring_payload, "psi_bins.json") 
        
    # =================================
    # Saving model artefacts
    # =================================
    model_artefact = {}

    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscalar'] = best_model.named_steps.get('standard_scalar') #load data from pipeline
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = gini_train
    model_artefact['results']['gini_test'] = gini_test
    model_artefact['results']['gini_oot'] = gini_oot
    model_artefact['results']['psi_score_train_vs_test'] = psi_score_train_test
    model_artefact['results']['psi_score_train_vs_oot'] = psi_score_train_oot
    model_artefact['results']['csi_mean_train_vs_test'] = csi_mean_tt
    model_artefact['results']['csi_max_train_vs_test'] = csi_max_tt
    model_artefact['results']['csi_mean_train_vs_oot'] = csi_mean_to
    model_artefact['results']['csi_max_train_vs_oot'] = csi_max_to
    model_artefact['hp_params'] = random_search.best_params_
    model_artefact['monitoring_bins'] = monitoring_payload # monitoring baselines for inference-time PSI/CSI

    # =================================
    # Saving model
    # =================================
    # create model_bank dir
    model_bank_directory = "/opt/airflow/model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    model_training(args.snapshotdate)
