# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.types import DateType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor as rf_sp
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC #### Bash codes for creating centralised mlflow

# COMMAND ----------

#databricks configure --token 
#enter host (with worksapce id included) #databricks link of dev, QA, prod
#enter token of model dev workspace  
#databricks secrets create-scope --scope modelregistery
#databricks secrets put --scope modelregistery --key modelregistery-token --string-value dapi5d4a1a907559461e73117957709bfbb6-2 #mlfow dbx token
#databricks secrets put --scope modelregistery --key modelregistery-workspace-id --string-value 1440288414278528 #mlflow dbx workspace id
#databricks secrets put --scope modelregistery --key modelregistery-host --string-value https://adb-1440288414278528.8.azuredatabricks.net/ #mlflow dbx link

#databricks secrets delete --scope modelregistery --key modelregistery-host
# latest databricks cli not working use python -m pip install --upgrade pip setuptools wheel databricks-cli==0.11.0

# COMMAND ----------

registry_uri = f'databricks://modelregistery:modelregistery'
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

def count_missings(df):
    """
    Counts numbercount_missings of nulls and nans in each column
    """
    df = df.select([F.count(F.when(F.isnan(c) |\
                                         F.col(c).isNull() |\
                                         (F.col(c) == '') |\
                                         F.col(c). contains ('None'), c)).alias(c) for c in df.columns])

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

spark_df = spark.read.csv('/FileStore/MLOps/Training/kc_house_data.csv', header=True, inferSchema=True)
spark_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleaning

# COMMAND ----------

count_missings(spark_df)
check_duplicates(spark_df)
replace_spaces(spark_df)
detect_data_type(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Preprocessing

# COMMAND ----------

df = (
    spark_df
        .withColumn('date', F.to_date(F.col('date'), "yyyyMMdd'T'HHmmss"))
        .withColumn('date', F.year('date'))
        .withColumn('bathrooms', F.round(F.col('bathrooms')))
        .withColumn('floors', F.round(F.col('floors')))
)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Save as delta

# COMMAND ----------

create_delta_table(df, 'MLOps', 'housing_data', '/user/hive/warehouse/MLOps/housing_dataset')

# COMMAND ----------

Housing_df = spark.read.table('MLOps.housing_data')
Housing_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Modelling

# COMMAND ----------

# Start an MLflow experiment
experiment_name = "/MLOps_Housing/"
mlflow.set_experiment(experiment_name)

# Split the data into training and testing sets
(train_data, test_data) = Housing_df.randomSplit([0.7, 0.3], seed = 20)
print("Number of training set rows: %d" % train_data.count())
print("Number of test set rows: %d" % test_data.count())


# COMMAND ----------

# Train the model and log the results with MLflow
with mlflow.start_run(run_name="MLOps-Training-RF") as run:
    # Define the feature vector using VectorAssembler
    dropped_col = Housing_df.drop('id','price')
    feature_cols = dropped_col.columns
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Define the Random Forest regressor
    rf = rf_sp(featuresCol="features", labelCol="price", numTrees=10)

    # Build the ML pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Fit the pipeline on the training data
    model = pipeline.fit(train_data)
    
    # Make predictions on the testing data
    predictions = model.transform(test_data)
    
    # Evaluate the model using RMSE
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
   # rmse = evaluator.evaluate(predictions)
    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    r2 = evaluator.setMetricName("r2").evaluate(predictions)
    
    # Log the model parameters and evaluation metrics with MLflow
    mlflow.spark.log_model(model, "model")
    mlflow.log_param("numTrees", rf.getNumTrees())
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

# End the MLflow experiment
mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Moodel

# COMMAND ----------

# Register Model
model_name = "RF_MLOps_Training"
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Check Model Status
client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

client.update_registered_model(
  name=model_details.name,
  description="This model forecasts housing list prices based on various inputs."
)

# COMMAND ----------

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using random forest in pyspark and without hyperparameter tuning."
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hyperparameter Tuning

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow

def objective_function(params):    
  # set the hyperparameters that we want to tune
  max_depth = params["max_depth"]
  num_trees = params["num_trees"]

  # create a grid with our hyperparameters
  grid = (ParamGridBuilder()
    .addGrid(rf.maxDepth, [max_depth])
    .addGrid(rf.numTrees, [num_trees])
    .build())

  # cross validate the set of hyperparameters
  cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)
  cvModel = cv.fit(train_data)

  # get our average RMSE across all three folds
  rmse = cvModel.avgMetrics[0]

  return {"loss": rmse, "status": STATUS_OK}

# COMMAND ----------

from hyperopt import hp

search_space = {
  "max_depth": hp.randint("max_depth", 2, 5),
  "num_trees": hp.randint("num_trees", 10, 100)
}

# COMMAND ----------

from hyperopt import fmin, tpe, STATUS_OK, Trials
import numpy as np

# Creating a parent run
with mlflow.start_run():
  num_evals = 4
  trials = Trials()
  best_hyperparam = fmin(fn=objective_function, 
                         space=search_space,
                         algo=tpe.suggest, 
                         max_evals=num_evals,
                         trials=trials,
                         rstate=np.random.default_rng(42)
                        )
  
  # get optimal hyperparameter values
  best_max_depth = best_hyperparam["max_depth"]
  best_num_trees = best_hyperparam["num_trees"]
  
  # change RF to use optimal hyperparameter values (this is a stateful method)
  rf.setMaxDepth(best_max_depth)
  rf.setNumTrees(best_num_trees)
  
  # train pipeline on entire training data - this will use the updated RF values
  pipelineModel = pipeline.fit(train_data)
  
  # evaluate final model on test data
  predDF = pipelineModel.transform(test_data)
  rmse = evaluator.setMetricName("rmse").evaluate(predictions)
  r2 = evaluator.setMetricName("r2").evaluate(predictions)
  
  # Log param and metric for the final model
  mlflow.log_param("max_depth", best_max_depth)
  mlflow.spark.log_model(model, "model")
  mlflow.log_param("numTrees", rf.getNumTrees())
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)

 

# COMMAND ----------



# COMMAND ----------

# Register Model
model_name = "RF_MLOps_Training_hp"
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Check Model Status
client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

client.update_registered_model(
  name=model_details.name,
  description="This model forecasts housing list prices based on various inputs."
)

# COMMAND ----------

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using random forest in pyspark and with hyperparameter tuning."
)

# COMMAND ----------


