// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import ml.dmlc.xgboost4j.scala.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.FloatType


// COMMAND ----------

val labelName = "fare_amount"

lazy val schema =
  StructType(Array(
    StructField("vendor_id", DoubleType),
    StructField("passenger_count", DoubleType),
    StructField("trip_distance", DoubleType),
    StructField("pickup_longitude", DoubleType),
    StructField("pickup_latitude", DoubleType),
    StructField("rate_code", DoubleType),
    StructField("store_and_fwd", DoubleType),
    StructField("dropoff_longitude", DoubleType),
    StructField("dropoff_latitude", DoubleType),
    StructField(labelName, DoubleType),
    StructField("hour", DoubleType),
    StructField("year", IntegerType),
    StructField("month", IntegerType),
    StructField("day", DoubleType),
    StructField("day_of_week", DoubleType),
    StructField("is_weekend", DoubleType)
  ))




// COMMAND ----------


val trainPath = "/FileStore/tables/taxi_tsmall-7ec29.csv"
val df = spark.read.option("inferSchema", "false").option("header", true).schema(schema).csv(trainPath)



// COMMAND ----------

val df2 = df.select($"hour", $"fare_amount", $"day_of_week").filter($"day_of_week" === "6.0" )

// COMMAND ----------

df2.show(3)


// COMMAND ----------

df2.take(1)

// COMMAND ----------

df2.explain(true)

// COMMAND ----------

val df3 = df2.groupBy("hour").count

df3.orderBy(asc("hour"))show(5)


// COMMAND ----------

df3.explain

// COMMAND ----------


df.show
df.cache
df.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")



// COMMAND ----------

display(df.select("passenger_count", "trip_distance","rate_code","fare_amount","hour","day_of_week"))

// COMMAND ----------

df.groupBy("hour").count.orderBy("hour").show()

// COMMAND ----------

df.select($"hour", $"fare_amount").filter($"hour" === "0.0" ).show(2)

// COMMAND ----------

df.select("passenger_count", "trip_distance","rate_code","fare_amount").describe().show

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from taxi

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance, rate_code, fare_amount, is_weekend, day_of_week from taxi

// COMMAND ----------

 df.select("trip_distance", "rate_code","fare_amount").show(5)

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance,avg(trip_distance),  avg(fare_amount)
// MAGIC from taxi
// MAGIC group by trip_distance order by avg(trip_distance) desc

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount), avg(trip_distance)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select rate_code, avg(fare_amount) , avg(trip_distance)
// MAGIC from taxi
// MAGIC group by rate_code order by rate_code

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance, fare_amount
// MAGIC from taxi
