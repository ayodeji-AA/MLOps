# Databricks notebook source
!pip install kaggle

# COMMAND ----------

# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=ayodejiogunlami
# MAGIC export KAGGLE_KEY=29e8b22aa9f3b823a6b94fda463db46b
# MAGIC kaggle datasets download -d harlfoxem/housesalesprediction
# MAGIC unzip housesalesprediction

# COMMAND ----------

dbutils.fs.cp( 'file:/Workspace/Repos/ayodeji@advancinganalytics.co.uk/MLOps/notebooks/kc_house_data.csv' , '/FileStore/MLOps/Training/kc_house_data.csv' )

# COMMAND ----------


