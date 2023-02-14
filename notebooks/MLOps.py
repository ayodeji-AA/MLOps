# Databricks notebook source
from mlflow.recipes import Recipe
import pyspark.sql.functions as F

# COMMAND ----------

def ingest_data(file_path):
    """
    Ingest data from different file formats and determine the format
    """
    
    if file_path.endswith(".csv"):
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print("The data is in CSV format.")
    elif file_path.endswith(".parquet"):
        df = spark.read.parquet(file_path)
        print("The data is in Parquet format.")
    elif file_path.endswith(".json"):
        df = spark.read.json(file_path)
        print("The data is in JSON format.")
    else:
        print("The format of the data is not supported.")
        return
    
    return df
    
    raise NotImplementedError


def count_missings(spark_df):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) |\
                                         F.col(c).isNull() |\
                                         (F.col(c) == '') |\
                                         F.col(c). contains ('None'), c)).alias(c) for c in spark_df.columns])

    col_counts = [F.sum(F.when(F.col(col) != 0, 1).otherwise(0)).alias(col) for col in df.columns]
    result = df.agg(*col_counts).collect()[0].asDict()
    for col, count in result.items():
        if count != 0:
            print(f"Column '{col}' has null values")
            df.filter(F.col(col).isNull()).show()
    if all(val == 0 for val in result.values()):
        print("There are no null values")



def check_duplicates(df):
    """
    Check for duplicate rows in a Spark dataframe
    """
    count = df.count()
    distinct_count = df.distinct().count()
    if count == distinct_count:
        print("There are no duplicate rows in the dataframe.")
    else:
        duplicates = df.groupBy().agg(F.count("*").alias("count")) \
            .filter(F.col("count") > 1)
        print("Duplicate rows detected in the following columns:")
        duplicates.show()
        
def replace_spaces(df):
    """
    Check for spaces in columns and replace them with underscores
    """
    has_space = False
    for col_name in df.columns:
        if " " in col_name:
            has_space = True
            df = df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

    if has_space:
        print("Spaces in column names have been replaced with underscores.")
    else:
        print("There are no spaces in column names.")
        
    return df


def detect_data_type(df):
    """
    Detects the data type of each column in a PySpark dataframe
    """
    # Get the schema of the dataframe
    schema = df.schema

    # Iterate over the schema to get the data type of each column
    for field in schema:
        print("Column: {}, Data Type: {}".format(field.name, field.dataType))


### path should be '/user/hive/warehouse/{project name}'
def create_delta_table(df, database, table, path):
    """
    Creates a Delta Lake database and reads a DataFrame into the database
    """
    
    # Write the DataFrame to a Delta Lake database
    df.write.format("delta").mode("overwrite").save(path)
    spark.sql(f'create database if not exists {database}')
    
    spark.sql("""
      CREATE TABLE IF NOT EXISTS {}.{}
      USING DELTA
      LOCATION '{}'
    """.format(database,table,path)
    )
    
    return None


# COMMAND ----------

# MAGIC %md
# MAGIC ###Ingest Dataset

# COMMAND ----------


