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
import re

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType


def process_silver_loan_table(snapshot_date_str, bronze_lms_directory, silver_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    installment = F.when((F.col("due_amt").isNull()) | (F.col("due_amt") == 0), None).otherwise(F.col("overdue_amt") / F.col("due_amt"))
    df = df.withColumn("installments_missed", F.coalesce(F.ceil(installment), F.lit(0)).cast("int"))
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_clickstream_table(snapshot_date_str, bronze_clks_directory, silver_clks_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clks_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clks_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # clean data: enforce schema / data type
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_clks_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clks_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_attributes_table(snapshot_date_str, bronze_attr_directory, silver_attr_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attr_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attr_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean Name field
    def clean_name(name):
        if name is None:
            return None
        cleaned = re.sub(r"[^A-Za-z .'-]", '', name)  # remove invalid characters
        return cleaned.strip()  # strip whitespace
    
    clean_name_udf = F.udf(clean_name, StringType())
    
    # Apply my function to the Name column
    df = df.withColumn("Name", clean_name_udf(F.col("Name")))

    # Clean Age field
    # Remove non-digit characters from Age, then convert to int and remove negative and large values
    df = df.withColumn("Age", F.regexp_replace(F.col("Age").cast(StringType()), r"\D", ""))
    df = df.withColumn("Age", F.col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((F.col("Age") >= 0) & (F.col("Age") <= 120), F.col("Age")).otherwise(None))

    # Clean SSN
    valid_ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("SSN", F.when(F.col("SSN").rlike(valid_ssn_pattern), F.col("SSN")).otherwise(None))

    # CLean Occupation
    df = df.withColumn("Occupation", F.when(F.col("Occupation") == '_______', None).otherwise(F.col("Occupation")))
    mapping_expr = F.when(F.col("Occupation") == "Accountant", 0)\
                   .when(F.col("Occupation") == "Architect", 1)\
                   .when(F.col("Occupation") == "Developer", 2)\
                   .when(F.col("Occupation") == "Doctor", 3)\
                   .when(F.col("Occupation") == "Engineer", 4)\
                   .when(F.col("Occupation") == "Entrepreneur", 5)\
                   .when(F.col("Occupation") == "Journalist", 6)\
                   .when(F.col("Occupation") == "Lawyer", 7)\
                   .when(F.col("Occupation") == "Manager", 8)\
                   .when(F.col("Occupation") == "Mechanic", 9)\
                   .when(F.col("Occupation") == "Media_Manager", 10)\
                   .when(F.col("Occupation") == "Musician", 11)\
                   .when(F.col("Occupation") == "Scientist", 12)\
                   .when(F.col("Occupation") == "Teacher", 13)\
                   .when(F.col("Occupation") == "Writer", 14)\
                   .otherwise(15) #For '_________'
    df = df.withColumn("Occupation", mapping_expr.cast("int"))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_attr_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_financials_table(snapshot_date_str, bronze_fin_directory, silver_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_fin_mthly_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fin_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Decimal currency columns (3dp based on ISO standard)
    cols_decimal3 = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for c in cols_decimal3:
        cleaned = F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", "")
        double_value = F.when(cleaned == "", None).otherwise(cleaned.cast("double"))
        rounded_value = F.when(double_value.isNotNull(), F.round(double_value, 3)).otherwise(None)
        df = df.withColumn(c, F.when(rounded_value < 0, None).otherwise(rounded_value))

    # Integer columns
    cols_integer = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries', 'Interest_Rate'
    ]
    for c in cols_integer:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^\d]", ""))
        df = df.withColumn(c, F.when(F.col(c) == "", None).otherwise(F.col(c).cast("int")))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    # Convert Credit_History_Age to months
    def convert_to_months(s):
        if s is None:
            return None
        m = re.match(r"(\d+) Years and (\d+) Months", s)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return None

    convert_udf = F.udf(convert_to_months, IntegerType())
    df = df.withColumn("Credit_History_Age", convert_udf(F.col("Credit_History_Age")))

    # Float columns
    cols_float = ['Changed_Credit_Limit', 'Credit_Utilization_Ratio']
    for c in cols_float:
        cleaned = F.regexp_replace(F.col(c).cast("string"), r"[^\d.]", "")
        df = df.withColumn(c, F.when(cleaned == "", None).otherwise(cleaned.cast("double")))

    # Remove negative credit utilization ratios
    df = df.withColumn("Credit_Utilization_Ratio", F.when(F.col("Credit_Utilization_Ratio") < 0, None).otherwise(F.col("Credit_Utilization_Ratio")))

    # Cap outliers
    outlier_caps = [
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries',
    ]
    for colname in outlier_caps: #cap at 97th percentile
        percentile_value = df.approxQuantile(colname, [0.97], 0.01)[0]
        df = df.withColumn(colname, F.when(F.col(colname) > percentile_value, percentile_value).otherwise(F.col(colname)))

    # Encode Credit_Mix
    credit_mix_mapping = ['Bad', 'Standard', 'Good']
    # Remove invalid values first
    df = df.withColumn("Credit_Mix", F.when(F.col("Credit_Mix").isin(credit_mix_mapping), F.col("Credit_Mix")).otherwise(None))
    mapping_expr = F.when(F.col("Credit_Mix") == "Bad", 0)\
                   .when(F.col("Credit_Mix") == "Standard", 1)\
                   .when(F.col("Credit_Mix") == "Good", 2)
    df = df.withColumn("Credit_Mix", mapping_expr.cast("int"))

    # Encode Payment_of_Min_Amount
    df = df.withColumn("Payment_of_Min_Amount", 
        F.when(F.col("Payment_of_Min_Amount") == "Yes", True)
        .when(F.col("Payment_of_Min_Amount") == "No", False)
        .otherwise(None))

    # Encode Payment_Behaviour
    valid_pb_enums = [
        'High_spent_Small_value_payments', 'High_spent_Medium_value_payments',
        'High_spent_Large_value_payments', 'Low_spent_Small_value_payments',
        'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments'
    ]

    # Remove invalid values first
    df = df.withColumn("Payment_Behaviour",
        F.when(F.col("Payment_Behaviour").isin(valid_pb_enums), F.col("Payment_Behaviour")).otherwise(None))

    mapping_expr_pb = F.when(F.col("Payment_Behaviour") == "Low_spent_Small_value_payments", 0)\
        .when(F.col("Payment_Behaviour") == "Low_spent_Medium_value_payments", 1)\
        .when(F.col("Payment_Behaviour") == "Low_spent_Large_value_payments", 2)\
        .when(F.col("Payment_Behaviour") == "High_spent_Small_value_payments", 3)\
        .when(F.col("Payment_Behaviour") == "High_spent_Medium_value_payments", 4)\
        .when(F.col("Payment_Behaviour") == "High_spent_Large_value_payments", 5)
    df = df.withColumn("Payment_Behaviour", mapping_expr_pb.cast("int"))

    # Feature Engineering
    df = df.withColumn("Loans_per_Credit_Item", F.col("Num_of_Loan") / (F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.lit(1)))
    df = df.withColumn("Debt_to_Salary", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("Loan_Extent", F.col("Delay_from_due_date") * F.col("Num_of_Loan"))
    
    # save silver table - IRL connect to database to write
    partition_name = "silver_fin_mthly_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df