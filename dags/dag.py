import os, glob, re, json
import pandas as pd
from sklearn.metrics import roc_auc_score 
from mlflow.tracking import MlflowClient
import mlflow
from airflow import DAG
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator, PythonOperator, get_current_context, BranchPythonOperator
from datetime import datetime, timedelta, date 
from mlflow.tracking import MlflowClient
from airflow.models import TaskInstance, Variable
from airflow.sensors.filesystem import FileSensor
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
# ====================================
# Function for Data Pipeline
# ====================================
def should_build_if_missing(path, expect_dir=True):
    """
    True  -> run the downstream task (output missing for this ds)
    False -> skip downstream (output already present for this ds)
    """
    if expect_dir:
        status = os.path.isdir(path) and any(os.scandir(path))  # dir exists and has content
    else:
        status = os.path.isfile(path)

    if status:
        print(f"Output already exists at: {path}. Skipping data pipeline task.")
        return False
    else:
        print(f"Missing output: {path}. Proceeding with data pipeline task.")
        return True

# ====================================
# Function for Model Training Pipeline
# ====================================
def should_train_quarterly_if_enough_history(label_dir, min_months = 20, quarter_months=(3, 6, 9, 12)):
    """
    True  -> allow downstream training to run
    False -> skip training this run

    Conditions:
      1) logical_date month is a quarter in `quarter_months`
      2) at least `min_months` unique months of gold feature files exist
         (<= logical_date) under `label_dir`, named like:
         gold_feature_store_YYYY_MM_DD.parquet
    """
    ctx = get_current_context()
    ds = ctx["ds"] # '2024-09-01'
    run_date = datetime.strptime(ds, "%Y-%m-%d").date()

    # quarterly training
    if run_date.month not in quarter_months:
        print(f"Skipping training - {run_date} is not a training month.")
        return False

    # only proceed if there is enough historical data
    if not os.path.isdir(label_dir):
        print(f"Missing dir: {label_dir}")
        return False

    pat = re.compile(r"gold_label_store_(\d{4})_(\d{2})_(\d{2})\.parquet$")
    months_seen = set()
    for file in os.listdir(label_dir):
        match = pat.match(file)
        if not match:
            continue
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        filedate = date(year, month, day)
        if filedate <= run_date:
            months_seen.add(filedate.replace(day=1))

    if len(months_seen) < min_months:
        print(f" Skipping first model training - less than {min_months} months.")
        return False

    print("Starting model training.")
    return True

def mark_trained_this_ds():
    ds = get_current_context()["ds"]
    Variable.set(f"trained_{ds}", "1")

def skip_if_already_trained():
    ds = get_current_context()["ds"]
    # return True (proceed) only if not trained yet
    return Variable.get(f"trained_{ds}", default_var="0") == "0"
# =======================================
# Functions for Model Inference Pipeline
# =======================================
def select_model_for_inference(model_name="loan_default_pipeline", model_stage="Production", model_bank_dir="/opt/airflow/model_bank"):
    """
    Returns a dict:
      {
        "model_uri": "models:/loan_default_pipeline/Production"  # if found in MLflow
      }
    or
      {
        "pickle_path": "/opt/airflow/model_bank/credit_model_YYYY_MM_DD.pkl"  # newest pickle
      }
    or {} if nothing found.
    """
    # Look for trained model in MLflow
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:/opt/mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # list all model versions and pick the latest model
        versions = client.search_model_versions(f"name='{model_name}'")
        # fallback to latest version when no model is in the requested stage
        if versions:
            best_any = sorted(versions, key=lambda v: int(v.last_updated_timestamp), reverse=True)[0]
            # load by explicit version number
            uri = f"models:/{model_name}/{best_any.version}"
            return {"model_uri": uri}
    except Exception as err:
        print(f"MLflow registry not available or no model at stage: {err}")

    # Fallback - pick latest pickle under model_bank_dir
    try:
        if os.path.isdir(model_bank_dir):
            files = [os.path.join(model_bank_dir, f) for f in os.listdir(model_bank_dir) if f.endswith(".pkl")]
            if files:
                newest = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)[0]
                return {"pickle_path": newest}
    except Exception as err:
        print(f"pickle fallback lookup failed: {err}")

    # No model found
    return {}

# function to enter inference based on model availability
def run_inference_if_model_available(feature_dir):
    ctx = get_current_context()
    ds = ctx["ds"] # '2024-09-01'
    run_date = datetime.strptime(ds, "%Y-%m-%d").date()
    pat = re.compile(r"gold_ft_store_cust_fin_risk_(\d{4})_(\d{2})_(\d{2})\.parquet$")
    months_seen = set()
    for file in os.listdir(feature_dir):
        match = pat.match(file)
        if not match:
            continue
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        filedate = date(year, month, day)
        if filedate <= run_date:
            months_seen.add(filedate.replace(day=1))

    if len(months_seen) > 20: #only start inference if there are more than 20 months
        resolved = select_model_for_inference(
            model_name="loan_default_pipeline",
            model_stage="Production",
            model_bank_dir="/opt/airflow/model_bank"
        )
        ctx = get_current_context()
        ctx["ti"].xcom_push(key="resolved_model", value=resolved) 
        ok = bool(resolved)
        print(f"[inference gate] resolved={resolved} -> will_run={ok}")
        return ok

# =======================================
# Functions for Model Monitoring Pipeline
# =======================================
def _pick_prediction_file_for_snapshot(snapshot_fmt: str):
    pred_dir = "/opt/airflow/datamart/gold/model_predictions"
    model_dirs = [os.path.join(pred_dir, d) for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
    for d in sorted(model_dirs, key=os.path.getmtime, reverse=True):
        hits = glob.glob(os.path.join(d, f"*_{snapshot_fmt}.parquet"))
        if hits:
            return hits[0], os.path.basename(d)
    raise FileNotFoundError(f"No predictions parquet for snapshot {snapshot_fmt}")

def compute_supervised_metrics(ds, **_):
    snap_str = ds
    snap_fmt = snap_str.replace("-", "_")
    pred_file, model_id = _pick_prediction_file_for_snapshot(snap_fmt)
    y_pred = pd.read_parquet(pred_file)
    y = pd.read_parquet(f"/opt/airflow/datamart/gold/label_store/gold_label_store_{snap_fmt}.parquet")[["Customer_ID","label"]]
    df = y_pred.merge(y, on="Customer_ID", how="inner").dropna(subset=["model_predictions","label"])
    y_true = df["label"].astype(int).to_numpy()
    y_prob = df["model_predictions"].to_numpy()
    auc  = float(roc_auc_score(y_true, y_prob))
    gini = float(2*auc - 1)

    # log to MLflow (time-series charts)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/opt/mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_default_baseline"))
    with mlflow.start_run(run_name=f"monitor_{snap_str}"):
        mlflow.log_param("snapshot_date", snap_str)
        mlflow.log_param("model_id", model_id)
        mlflow.log_metrics({
            "auc_infer": auc,
            "gini_infer": gini,
        })
    # thresholds
    thr_auc_min = 0.70
    thr_gini_min = 0.30
    auc_breach = auc < thr_auc_min
    gini_breach = gini < thr_gini_min
    breach = auc_breach or gini_breach
    # Push XCom for branching
    ti = get_current_context()["ti"]
    ti.xcom_push(key="monitor_metrics", value=json.dumps({
        "auc": auc, "gini": gini, "auc_breach": auc_breach, "gini_breach": gini_breach, "breach": breach, "model_id": model_id
    }))
    #push metrics to prometheus for storage
    PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "http://pushgateway:9091")
    reg = CollectorRegistry()
    labels = {"snapshot": ds, "model": model_id}
    g_auc  = Gauge("ml_auc", "AUC", labels.keys(), registry=reg)
    g_gini = Gauge("ml_gini", "Gini", labels.keys(), registry=reg)
    g_auc.labels(**labels).set(float(auc))
    g_gini.labels(**labels).set(float(gini))
    g_auc_thr    = Gauge("ml_auc_threshold",  "AUC threshold",  labels.keys(), registry=reg)
    g_gini_thr   = Gauge("ml_gini_threshold", "Gini threshold", labels.keys(), registry=reg)
    g_auc_thr.labels(**labels).set(float(thr_auc_min))
    g_gini_thr.labels(**labels).set(float(thr_gini_min))
    g_auc_thr.labels(**labels).set(thr_auc_min)
    g_gini_thr.labels(**labels).set(thr_gini_min)
    push_to_gateway(PUSHGATEWAY_ADDR, job="loan_default_monitor", registry=reg)
    return True

def _drift_sidecar_path(ds_str: str, model_predictions_root: str = "/opt/airflow/datamart/gold/model_predictions"):
    snap_fmt = ds_str.replace("-", "_")
    # find the most recent model_id folder with this snapshot drift file
    dirs = [os.path.join(model_predictions_root, d) for d in os.listdir(model_predictions_root)
            if os.path.isdir(os.path.join(model_predictions_root, d))]
    for d in sorted(dirs, key=os.path.getmtime, reverse=True):
        candidate = os.path.join(d, f"drift_{snap_fmt}.json")
        if os.path.isfile(candidate):
            return candidate
    return None

def decide_unsupervised_retrain(ds, **_):
    """Branch on PSI/CSI breach before labels exist."""
    sidecar = _drift_sidecar_path(ds)
    if not sidecar:
        print("No drift sidecar found -> skip unsupervised retrain")
        return "unsupervised_gate_pass"

    with open(sidecar, "r") as f:
        payload = json.load(f)

    psi_score = payload.get("psi_score")
    csi_max   = payload.get("csi_max")
    psi_thr   = float(payload.get("psi_threshold", 0.25))
    csi_thr   = float(payload.get("csi_threshold", 0.25))
    psi_breach = (psi_score is not None and psi_score >= psi_thr)
    csi_breach = (csi_max   is not None and csi_max   >= csi_thr)

    print(f"[unsupervised drift] psi={psi_score} (thr {psi_thr}) csi_max={csi_max} (thr {csi_thr}) "
          f"-> breach={psi_breach or csi_breach}")

    return "model_retrain_from_stability_breach" if (psi_breach or csi_breach) else "stability_gate_pass"

def decide_retrain(**context):
    ti = context["ti"]
    payload = json.loads(ti.xcom_pull(task_ids="compute_supervised_metrics", key="monitor_metrics"))
    if payload.get("breach"):
        print("Threshold breach detected. Retrain branch.")
        return "model_retrain_from_performance_degradation"
    else:
        print("No breach. Model not retrained.")
        return "model_monitor_completed"

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    # end_date=datetime(2023, 2, 1),
    catchup=True,
) as dag:
    #=====================
    # Data Pipeline
    #=====================
    # --- label store ---

    # Check that the raw label source data exists before starting the label pipeline
    dep_check_source_label_data = BashOperator(
        task_id="dep_check_source_label_data",
        bash_command=(
            'cd /opt/airflow/data && '
            'test -f lms_loan_daily.csv'
        ),
    )

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bypass_bronze_label = DummyOperator(task_id="bypass_bronze_label")
    # Check bronze label output exists for this snapshot date
    dep_check_bronze_label_output = ShortCircuitOperator(
    task_id="check_bronze_label",
    python_callable=should_build_if_missing,
    op_kwargs={
        "path": '/opt/airflow/datamart/bronze/lms/bronze_loan_daily_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.csv',
        "expect_dir":False,
            },
    ignore_downstream_trigger_rules=False
        )
    
    after_bronze_label_gate = DummyOperator(task_id="after_bronze_label_gate",
                                      trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_silver_label = DummyOperator(task_id="bypass_silver_label")

    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
        retries=0,             # temporarily disable retries so you see the first failure
        retry_delay=timedelta(seconds=1),
    )

    # Check silver label output exists for this snapshot date
    dep_check_silver_label_output = ShortCircuitOperator(
        task_id="check_silver_label",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/silver/lms/silver_loan_daily_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )
    
    after_silver_label_gate = DummyOperator(task_id="after_silver_label_gate",
                                      trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_gold_label = DummyOperator(task_id="bypass_gold_label")
    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Check gold label store output exists for this snapshot date
    dep_check_gold_label_output = ShortCircuitOperator(
        task_id="check_gold_label",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/gold/label_store/gold_label_store_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )
    
    label_store_completed = DummyOperator(task_id="label_store_completed",
                                           trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
 
    # --- feature store ---
    # Check that raw clickstream source exists
    dep_check_source_data_bronze_clickstream = BashOperator(
        task_id="dep_check_source_data_bronze_clickstream",
        bash_command=(
            'cd /opt/airflow/data && '
            'test -f feature_clickstream.csv'
        ),
    )

    bronze_clickstream_store = BashOperator(
        task_id='run_bronze_clickstream_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_clickstream_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    bypass_bronze_clickstream = DummyOperator(task_id="bypass_bronze_clickstream")
    # Check bronze clickstream output exists for this snapshot date
    dep_check_bronze_clickstream_output = ShortCircuitOperator(
        task_id="dep_check_bronze_clickstream_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/bronze/clks/bronze_clks_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.csv',
            "expect_dir":False,
                },
        ignore_downstream_trigger_rules=False
            )
    after_bronze_clickstream_gate = DummyOperator(task_id="after_bronze_clickstream_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_silver_clickstream = DummyOperator(task_id="bypass_silver_clickstream")

    silver_clickstream_store = BashOperator(
        task_id='run_silver_clickstream_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_clickstream_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    # Check silver clickstream output exists for this snapshot date
    dep_check_silver_clickstream_output = ShortCircuitOperator(
        task_id="dep_check_silver_clickstream_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/silver/clks/silver_clks_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )
    after_silver_clickstream_gate = DummyOperator(task_id="after_silver_clickstream_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_gold_clickstream = DummyOperator(task_id="bypass_gold_clickstream")
    
    after_gold_eng_gate = DummyOperator(task_id="after_gold_eng_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    
    # Check that raw attributes source exists
    dep_check_source_data_bronze_attr = BashOperator(
        task_id="dep_check_source_data_bronze_attr",
        bash_command=(
            'cd /opt/airflow/data && '
            'test -f features_attributes.csv'
        ),
    )


    bronze_attr_store = BashOperator(
        task_id='run_bronze_attr_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_attr_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Check bronze attributes output exists for this snapshot date
    dep_check_bronze_attr_output = ShortCircuitOperator(
        task_id="dep_check_bronze_attr_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/bronze/attr/bronze_attr_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.csv',
            "expect_dir":False,
                },
        ignore_downstream_trigger_rules=False
            )

    after_bronze_attr_gate = DummyOperator(task_id="after_bronze_attr_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_bronze_attr = DummyOperator(task_id="bypass_bronze_attr")

    silver_attr_store = BashOperator(
        task_id='run_silver_attr_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_attr_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    after_silver_attr_gate = DummyOperator(task_id="after_silver_attr_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_silver_attr = DummyOperator(task_id="bypass_silver_attr")

    # Check silver attributes output exists for this snapshot date
    dep_check_silver_attr_output = ShortCircuitOperator(
        task_id="dep_check_silver_attr_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/silver/attr/silver_attr_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )



    # Check that raw financials source exists
    dep_check_source_data_bronze_fin = BashOperator(
        task_id="dep_check_source_data_bronze_fin",
        bash_command=(
            'cd /opt/airflow/data && '
            'test -f features_financials.csv'
        ),
    )

    bronze_fin_store = BashOperator(
        task_id='run_bronze_fin_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_fin_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    after_bronze_fin_gate = DummyOperator(task_id="after_bronze_fin_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_bronze_fin = DummyOperator(task_id="bypass_bronze_fin")
    # Check bronze financials output exists for this snapshot date
    dep_check_bronze_fin_output = ShortCircuitOperator(
        task_id="dep_check_bronze_fin_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/bronze/fin/bronze_fin_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.csv',
            "expect_dir":False,
                },
        ignore_downstream_trigger_rules=False
            )
    
    after_silver_fin_gate = DummyOperator(task_id="after_silver_fin_gate",
                                    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_silver_fin = DummyOperator(task_id="bypass_silver_fin")

    silver_fin_store = BashOperator(
        task_id='run_silver_fin_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_fin_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Check silver financials output exists for this snapshot date
    dep_check_silver_fin_output = ShortCircuitOperator(
        task_id="dep_check_silver_fin_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/silver/fin/silver_fin_mthly_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )


    gold_cust_eng_store = BashOperator(
        task_id='run_gold_cust_eng_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_cust_eng_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    after_gold_cust_risk_gate = DummyOperator(task_id="after_gold_cust_risk_gate",
                                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    bypass_gold_cust_risk = DummyOperator(task_id="bypass_gold_cust_risk")
    # Check gold engagement feature output exists for this snapshot date
    dep_check_gold_eng_output = ShortCircuitOperator(
        task_id="dep_check_gold_eng_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/gold/feature_store/eng/gold_ft_store_engagement_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )


    gold_cust_risk_store = BashOperator(
        task_id='run_gold_cust_fin_risk_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_cust_fin_risk_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Check gold customer financial risk feature output exists for this snapshot date
    dep_check_gold_cust_risk_output = ShortCircuitOperator(
        task_id="dep_check_gold_cust_risk_output",
        python_callable=should_build_if_missing,
        op_kwargs={
            "path": '/opt/airflow/datamart/gold/feature_store/cust_fin_risk/gold_ft_store_cust_fin_risk_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
            "expect_dir":True,
                },
        ignore_downstream_trigger_rules=False
            )

    feature_store_completed = DummyOperator(task_id="feature_store_completed", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
 
     # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> [dep_check_bronze_label_output,bypass_bronze_label] 
    dep_check_bronze_label_output >> bronze_label_store 
    bronze_label_store >> after_bronze_label_gate
    bypass_bronze_label >> after_bronze_label_gate
    after_bronze_label_gate >> [dep_check_silver_label_output, bypass_silver_label]
    dep_check_silver_label_output >> silver_label_store
    silver_label_store >> after_silver_label_gate
    bypass_silver_label >> after_silver_label_gate
    after_silver_label_gate >> [dep_check_gold_label_output, bypass_gold_label]
    dep_check_gold_label_output >> gold_label_store
    gold_label_store >> label_store_completed
    bypass_gold_label >> label_store_completed


    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_clickstream >> [dep_check_bronze_clickstream_output,bypass_bronze_clickstream]
    dep_check_bronze_clickstream_output >> bronze_clickstream_store 
    bronze_clickstream_store >> after_bronze_clickstream_gate
    bypass_bronze_clickstream >> after_bronze_clickstream_gate
    after_bronze_clickstream_gate >> [dep_check_silver_clickstream_output, bypass_silver_clickstream]
    dep_check_silver_clickstream_output >> silver_clickstream_store
    silver_clickstream_store >> after_silver_clickstream_gate
    bypass_silver_clickstream >> after_silver_clickstream_gate
    after_silver_clickstream_gate >> [dep_check_gold_eng_output, bypass_gold_clickstream]
    dep_check_gold_eng_output >> gold_cust_eng_store
    gold_cust_eng_store >> after_gold_eng_gate 
    bypass_gold_clickstream >> after_gold_eng_gate
    after_gold_eng_gate >> feature_store_completed


    dep_check_source_data_bronze_attr >>  [dep_check_bronze_attr_output, bypass_bronze_attr]
    dep_check_bronze_attr_output >> bronze_attr_store
    bronze_attr_store >> after_bronze_attr_gate
    bypass_bronze_attr >> after_bronze_attr_gate
    after_bronze_attr_gate >> [dep_check_silver_attr_output, bypass_silver_attr]
    dep_check_silver_attr_output >> silver_attr_store
    silver_attr_store >> after_silver_attr_gate
    bypass_silver_attr >> after_silver_attr_gate

    dep_check_source_data_bronze_fin >> [dep_check_bronze_fin_output, bypass_bronze_fin] 
    dep_check_bronze_fin_output >> bronze_fin_store
    bronze_fin_store >> after_bronze_fin_gate
    bypass_bronze_fin >> after_bronze_fin_gate
    after_bronze_fin_gate >> [dep_check_silver_fin_output, bypass_silver_fin]
    dep_check_silver_fin_output >> silver_fin_store
    silver_fin_store >> after_silver_fin_gate
    bypass_silver_fin >> after_silver_fin_gate
    # Gold features depend on all silver outputs being present
    after_silver_attr_gate >> [dep_check_gold_cust_risk_output, bypass_gold_cust_risk]
    after_silver_fin_gate >> [dep_check_gold_cust_risk_output, bypass_gold_cust_risk]
    dep_check_gold_cust_risk_output >> gold_cust_risk_store
    gold_cust_risk_store >> after_gold_cust_risk_gate
    bypass_gold_cust_risk >> after_gold_cust_risk_gate
    after_gold_cust_risk_gate >> feature_store_completed

    # ============================================
    # Model Auto Training Pipeline
    # ============================================
    model_automl_start = DummyOperator(task_id="model_automl_start")
    train_quarterly = ShortCircuitOperator(
            task_id="quarterly_train_branch",
            python_callable=should_train_quarterly_if_enough_history,
            op_kwargs={
                "label_dir": '/opt/airflow/datamart/gold/label_store',
                "min_months": 20, #only start first training when at least 20 months of data is present
                "quarter_months": (3, 6, 9, 12),
            },
            ignore_downstream_trigger_rules=False
        )
    bypass_model_training = DummyOperator(task_id="bypass_model_training")
    model_automl = BashOperator(
                    task_id="loan_default_model_automl",
                    bash_command=(
                    'cd /opt/airflow/scripts && '
                    'python3 model_training.py '
                    '--snapshotdate "{{ ds }}"'
                ),)
    
    mark_trained = PythonOperator(
        task_id="mark_trained_this_ds",
        python_callable=mark_trained_this_ds
    )

    model_automl_completed = DummyOperator(task_id="model_automl_completed", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    [feature_store_completed, label_store_completed] >> model_automl_start
    model_automl_start >> [train_quarterly, bypass_model_training] 
    train_quarterly >> model_automl >> mark_trained >> model_automl_completed
    bypass_model_training >> model_automl_completed

    # ============================================
    # Model Inference
    # ============================================
    model_inference_start = DummyOperator(task_id="model_inference_start")
    
    checking_for_model = ShortCircuitOperator(
                    task_id="check_and_select_model_for_inference",
                    python_callable=run_inference_if_model_available,
                    op_kwargs={
                        "feature_dir": '/opt/airflow/datamart/gold/feature_store/cust_fin_risk',
                    },
                    ignore_downstream_trigger_rules=False)

    model_inference = BashOperator(
                        task_id="model_inference",
                        bash_command=(
                            'cd /opt/airflow/scripts && '
                            '{% set r = ti.xcom_pull(task_ids="check_and_select_model_for_inference", key="resolved_model") %}'
                            'python3 model_inference.py --snapshotdate "{{ ds }}" '
                            '{% if r and r.get("model_uri") %} --modeluri "{{ r.get("model_uri") }}" '
                            '{% elif r and r.get("pickle_path") %} --modelname "{{ r.get("pickle_path").split("/")[-1] }}" '
                            '{% else %} && echo "ERROR: No model found for inference" && exit 1 {% endif %}'
                        ),
                    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # # Define task dependencies to run scripts sequentially
    model_automl_completed >> model_inference_start >> checking_for_model >> model_inference >> model_inference_completed

    # ============================================
    # Model Monitoring
    # ============================================
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    #decide when performance degrades
    decide_to_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
        provide_context=True
    )

    stability_check = BranchPythonOperator(
        task_id="stability_drift_branch",
        python_callable=decide_unsupervised_retrain,
        provide_context=True,
    )

    gate_stability_retrain = ShortCircuitOperator(
        task_id="gate_stability_retrain",
        python_callable=skip_if_already_trained,
        ignore_downstream_trigger_rules=False
    )
    gate_perf_retrain = ShortCircuitOperator(
        task_id="gate_perf_retrain",
        python_callable=skip_if_already_trained,
        ignore_downstream_trigger_rules=False
    )

    #retrain when stability breach
    model_retrain_stability = BashOperator(
        task_id="model_retrain_from_stability_breach",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_training.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    wait_for_labels = FileSensor(
        task_id="wait_for_labels",
        filepath='/opt/airflow/datamart/gold/label_store/'
                 'gold_label_store_{{ macros.ds_format(ds, "%Y-%m-%d", "%Y_%m_%d") }}.parquet',
        poke_interval=300, timeout=60*60, mode="reschedule", fs_conn_id="fs_default",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    model_monitor = PythonOperator(
        task_id="compute_supervised_metrics",
        python_callable=compute_supervised_metrics,
        provide_context=True
    )
    #retrain when performance degrades
    model_retrain_performance = BashOperator(
        task_id="model_retrain_from_performance_degradation",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_training.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    post_retrain_infer = BashOperator(
    task_id="post_retrain_inference",
    bash_command=(
                    'cd /opt/airflow/scripts && '
                    '{% set r = ti.xcom_pull(task_ids="check_and_select_model_for_inference", key="resolved_model") %}'
                    'python3 model_inference.py --snapshotdate "{{ ds }}" '
                    '{% if r and r.get("model_uri") %} --modeluri "{{ r.get("model_uri") }}" '
                    '{% elif r and r.get("pickle_path") %} --modelname "{{ r.get("pickle_path").split("/")[-1] }}" '
                    '{% else %} && echo "ERROR: No model found for post-retrain inference" && exit 1 {% endif %}'
                ),
                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
            )


    #when there is no breach in stability
    stability_gate_pass = DummyOperator(task_id="stability_gate_pass")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start >> stability_check
    stability_check >> gate_stability_retrain >> model_retrain_stability >> post_retrain_infer >> model_monitor_completed
    stability_check >> stability_gate_pass >> wait_for_labels >> model_monitor >> decide_to_retrain
    decide_to_retrain >> gate_perf_retrain >> model_retrain_performance >> post_retrain_infer >> model_monitor_completed
    decide_to_retrain >> model_monitor_completed
