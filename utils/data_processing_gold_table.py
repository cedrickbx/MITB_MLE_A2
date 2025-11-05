import os
import glob
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_lms_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_fts_gold_engag_table(snapshot_date_str, silver_clks_directory, gold_clks_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    all_dfs = []

    for i in range(1, 7):  # from 1 to 6 months ago
        month_date = snapshot_date - relativedelta(months=i)
        partition_name = "silver_clks_mthly_" + month_date.strftime('%Y_%m_%d') + '.parquet'
        filepath = silver_clks_directory + partition_name

        try:
            df = spark.read.parquet(filepath)
            print(f'Loaded from: {filepath}, row count: {df.count()}')

            # Only use fe_1 for clicks since the behaviour is similar for all other fe_n
            # Add months_ago column to keep track
            df = df.select('Customer_ID', 'fe_1').withColumn('months_ago', F.lit(i))
            all_dfs.append(df)

        except Exception as e:
            print(f'No data for {i} months ago')

    if not all_dfs: #account for the 1st month (2023-01-01) without any data
        print("No data loaded for any of the previous 6 months; writing empty frame.")
        try:
            current_fp = silver_clks_directory + f"silver_clks_mthly_{snapshot_date:%Y_%m_%d}.parquet"
            base = spark.read.parquet(current_fp).select("Customer_ID").distinct()
        except Exception:
            base = spark.createDataFrame([], "Customer_ID string")
        df_final = base.withColumn("snapshot_date", F.lit(snapshot_date_str))\
            .select(["Customer_ID","snapshot_date"] + [F.lit(None).cast("int").alias(f"click_{i}m") for i in range(1,7)])
    
        filepath = gold_clks_directory + f"gold_ft_store_engagement_{snapshot_date_str.replace('-','_')}.parquet"
        df_final.write.mode("overwrite").parquet(filepath)
        print("Saved empty scaffold to:", filepath)
        return df_final
    
    # Union and pivot
    union_df = all_dfs[0] # start with first df in the list
    for df in all_dfs[1:]: # for remaining dfs in the list
        union_df = union_df.unionByName(df)

    pivot_df = union_df.groupBy('Customer_ID')\
        .pivot('months_ago', [1, 2, 3, 4, 5, 6])\
        .agg(F.first('fe_1')) # if there are duplicates take the first row
    
    for i in range(1, 7):
        if str(i) in pivot_df.columns:
            pivot_df = pivot_df.withColumnRenamed(str(i), f"click_{i}m")
        else:
            pivot_df = pivot_df.withColumn(f"click_{i}m", F.lit(None).cast("int"))
    
    # Rename columns or create cols (for missing data)
    for i in range(1, 7):
        pivot_df = pivot_df.withColumn(f'click_{i}m', F.col(f'click_{i}m').cast('int'))

    # Add snapshot_date column
    pivot_df = pivot_df.withColumn("snapshot_date", F.lit(snapshot_date_str))
    
    # Reorder columns for consistency
    ordered_cols = ['Customer_ID', 'snapshot_date'] + [f'click_{i}m' for i in range(1, 7)]
    df_final = pivot_df.select(ordered_cols)

    # Save to gold directory
    partition_name = "gold_ft_store_engagement_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_clks_directory + partition_name
    df_final.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)

    return df_final

def process_fts_gold_cust_risk_table(snapshot_date_str, silver_attr_directory, silver_fin_directory, gold_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_fin_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fin_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    # connect to silver attr table
    attr_files_list = [silver_attr_directory+os.path.basename(f) for f in glob.glob(os.path.join(silver_attr_directory, '*'))]
    attr_df = spark.read.option("header", "true").parquet(*attr_files_list)
    print('loaded from:', silver_attr_directory, 'row count:', attr_df.count())
    attr_df = attr_df.select("Customer_ID","Age","Occupation")

    # Feature Engineering
    df = df.withColumn("Loans_per_Credit_Item", F.col("Num_of_Loan") / (F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.lit(1)))
    df = df.withColumn("Debt_to_Salary", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))

    # extract columns for gold table (includes features engineered)
    df = df.select("Customer_ID", 'Credit_History_Age', 'Loans_per_Credit_Item', 'Debt_to_Salary', 'Loan_Extent', 'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date','Changed_Credit_Limit', 'Payment_Behaviour')    
    df = df.join(attr_df, "Customer_ID", "left")

    # save gold table - IRL connect to database to write
    partition_name = "gold_ft_store_cust_fin_risk_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
