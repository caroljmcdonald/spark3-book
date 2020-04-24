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
val evalPath  = "/FileStore/tables/taxi_esmall-98e87.csv"

val tdf = spark.read.option("inferSchema", "false").option("header", true).schema(schema).csv(trainPath)
val edf = spark.read.option("inferSchema", "false").option("header", true).schema(schema).csv(evalPath)


// COMMAND ----------

tdf.take(1)

// COMMAND ----------

tdf.show

// COMMAND ----------

val tdf = tdf0.withColumn("fare_amount", round( tdf0("fare_amount"),3)).withColumn("trip_distance", round( tdf0("trip_distance"),3))
val edf = edf0.withColumn("fare_amount", round( edf0("fare_amount"),3)).withColumn("trip_distance", round( edf0("trip_distance"),3))

// COMMAND ----------

edf.select("fare_amount", "trip_distance").show

// COMMAND ----------


tdf.show
//edf.show
tdf.cache
tdf.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")



// COMMAND ----------

display(tdf.select("passenger_count", "trip_distance","rate_code","fare_amount","hour","day_of_week"))

// COMMAND ----------

tdf.groupBy("hour").count.orderBy("hour").show(4)

// COMMAND ----------

tdf.select($"hour", $"fare_amount").filter($"hour" === "0.0" ).show(2)

// COMMAND ----------

tdf.select("passenger_count", "trip_distance","rate_code","fare_amount").describe().show

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from taxi

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance, rate_code, fare_amount, is_weekend, day_of_week from taxi

// COMMAND ----------

 tdf.select("trip_distance", "rate_code","fare_amount").show(5)

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

tdf.groupBy("hour").avg("fare_amount").orderBy("hour").show(5)

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

// COMMAND ----------

// MAGIC %sql
// MAGIC select day_of_week, avg(fare_amount), avg(trip_distance)
// MAGIC from taxi
// MAGIC group by day_of_week order by day_of_week

// COMMAND ----------

//display(
  tdf.select("trip_distance", "rate_code","fare_amount").describe().show

// COMMAND ----------

display(tdf.select("orig_ltv", "msa", "interest_rate", "current_actual_upb", "orig_loan_term").describe())


// COMMAND ----------

val featureNames = Array("passenger_count","trip_distance", "pickup_longitude","pickup_latitude","rate_code","dropoff_longitude", "dropoff_latitude", "hour", "day_of_week","is_weekend")

//val featureNames = schema.filter(_.name != labelName).map(_.name)

// COMMAND ----------

object Vectorize {
  def apply(df: DataFrame, featureNames: Array[String], labelName: String): DataFrame = {
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    new VectorAssembler()
      .setInputCols(featureNames)
      .setOutputCol("features")
      .transform(df.select(toFloat:_*))
      .select(col("features"), col(labelName))
  }
}



// COMMAND ----------


var trainSet = Vectorize(tdf, featureNames, labelName)
var evalSet = Vectorize(edf, featureNames, labelName)


// COMMAND ----------

trainSet.take(1)

// COMMAND ----------

lazy val paramMap = Map(
  "learning_rate" -> 0.05,
  "max_depth" -> 8,
  "subsample" -> 0.8,
  "gamma" -> 1,
  "num_round" -> 500
)
val xgbParamFinal = paramMap ++ Map("tree_method" -> "hist", "num_workers" -> 2)



// COMMAND ----------

val xgbRegressor = new XGBoostRegressor(xgbParamFinal)
      .setLabelCol(labelName)
      .setFeaturesCol("features")




// COMMAND ----------

object Benchmark {
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("Elapsed time [" + phase + "]: " + ((t1 - t0).toFloat / 1000) + "s")
    (result, (t1 - t0).toFloat / 1000)
  }
}




// COMMAND ----------

// Start training
println("\n------ Training ------")
// start training
val (model, _) = Benchmark.time("train") {
  xgbRegressor.fit(trainSet)
}

// COMMAND ----------

import org.apache.spark.ml.evaluation._

val (prediction, _) = Benchmark.time("transform") {
  val ret = model.transform(evalSet).cache()
  ret.foreachPartition(_ => ())
  ret
}
prediction.select( labelName, "prediction").show(10)
val evaluator = new RegressionEvaluator().setLabelCol(labelName)
val (rmse, _) = Benchmark.time("evaluation") {
  evaluator.evaluate(prediction)
}
println(s"RMSE == $rmse")
