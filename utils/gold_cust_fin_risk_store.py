import argparse
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import data_processing_bronze_table
import data_processing_silver_table
import data_processing_gold_table

# to call this script: python bronze_clickstream_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create gold datalake
    silver_fin_directory = "../datamart/silver/fin/"
    silver_attr_directory = "../datamart/silver/attr/"
    gold_fin_directory = "../datamart/gold/feature_store/cust_fin_risk/"

    if not os.path.exists(gold_fin_directory):
        os.makedirs(gold_fin_directory)
    #run data processing
    data_processing_gold_table.process_fts_gold_cust_risk_table(date_str, silver_attr_directory, silver_fin_directory, gold_fin_directory, spark)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
