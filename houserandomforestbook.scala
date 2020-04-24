// Databricks notebook source

import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline


// COMMAND ----------


val schema = StructType(Array(
    StructField("longitude", FloatType,true),
    StructField("latitude", FloatType, true),
    StructField("medage", FloatType, true),
    StructField("totalrooms", FloatType, true),
    StructField("totalbdrms", FloatType, true),
    StructField("population", FloatType, true),
    StructField("houshlds", FloatType, true),
    StructField("medincome", FloatType, true),
    StructField("medhvalue", FloatType, true)
))  

                  

// COMMAND ----------

import spark.implicits._

 var file ="/FileStore/tables/cal_housing.csv"


 var df  = spark.read.format("csv").option("inferSchema", "false").schema(schema).load(file) 
 
 df.show

// COMMAND ----------

df.describe("totalrooms","houshlds", "population" , "totalbdrms").show

// COMMAND ----------


df = df.withColumn("roomsPhouse", col("totalrooms")/col("houshlds"))
df = df.withColumn("popPhouse", col("population")/col("houshlds"))
df = df.withColumn("bedrmsPRoom", col("totalbdrms")/col("totalrooms"))

df.describe("roomsPhouse","popPhouse", "bedrmsPRoom" , "medhvalue").show

// COMMAND ----------

df=df.drop("totalrooms","houshlds", "population" , "totalbdrms")
df.show

// COMMAND ----------

df.describe("medincome","medhvalue","roomsPhouse","popPhouse").show

// COMMAND ----------

df.cache
df.createOrReplaceTempView("house")
spark.catalog.cacheTable("house")

// COMMAND ----------


df.select(corr("medhvalue","medincome")).show()


// COMMAND ----------

df.select(corr("medhvalue","popPhouse")).show()

// COMMAND ----------

df.select(corr("medhvalue","medage")).show()

// COMMAND ----------

df.select(corr("medhvalue","longitude")).show()

// COMMAND ----------

df.select(corr("medhvalue","bedrmsPRoom")).show()


// COMMAND ----------

df.select(corr("medhvalue","roomsPhouse")).show()

// COMMAND ----------

// MAGIC %sql 
// MAGIC select * from house

// COMMAND ----------

// MAGIC %sql 
// MAGIC select avg(medhvalue)  from house group by medhvalue 

// COMMAND ----------

// MAGIC %sql 
// MAGIC select * from house

// COMMAND ----------

// MAGIC %sql 
// MAGIC select medhvalue, medincome from house

// COMMAND ----------

// MAGIC %sql 
// MAGIC select longitude, latitude from house

// COMMAND ----------

val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), 1234)

// COMMAND ----------


//
val featureCols = Array("medage", "medincome", "roomsPhouse", "popPhouse", "bedrmsPRoom", "longitude", "latitude")

//put features into a feature vector column   
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("rawfeatures")
val scaler = new StandardScaler().setInputCol("rawfeatures").setOutputCol("features").setWithStd(true).setWithMean(true)

// COMMAND ----------

val rf = new RandomForestRegressor().setLabelCol("medhvalue").setFeaturesCol("features")   

// COMMAND ----------

val steps =  Array(assembler, scaler, rf)

val pipeline = new Pipeline().setStages(steps)

val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(50, 100, 200))
      .addGrid(rf.maxDepth, Array(2, 5, 10))
      .addGrid(rf.numTrees, Array(5, 10, 20))
      .build()

val evaluator = new RegressionEvaluator()
  .setLabelCol("medhvalue")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val crossvalidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

// COMMAND ----------

 val pipelineModel = crossvalidator.fit(trainingData)

// COMMAND ----------

val rfm = pipelineModel
      .bestModel.asInstanceOf[PipelineModel]
      .stages(2)
      .asInstanceOf[RandomForestRegressionModel]  

val featureImportances = rfm.featureImportances

  assembler.getInputCols
      .zip(featureImportances.toArray)
      .sortBy(-_._2)
      .foreach {
        case (feat, imp) =>
          println(s"feature: $feat, importance: $imp")
      }

// COMMAND ----------

rfm.toString()

// COMMAND ----------

val bestEstimatorParamMap = pipelineModel
      .getEstimatorParamMaps
      .zip(pipelineModel.avgMetrics)
      .maxBy(_._2)
      ._1
println(s"Best params:\n$bestEstimatorParamMap")


// COMMAND ----------

var predictions = pipelineModel.transform(testData)
predictions.select("prediction", "medhvalue").show(5)

// COMMAND ----------

predictions = predictions.withColumn("error", col("prediction")-col("medhvalue"))

// COMMAND ----------

predictions = predictions.withColumn("serror", col("error")-col("medhvalue"))   math.pow((actual - predicted), 2)
df.select("*", pow(col("col1"), col("col2")).alias("pow")).show()

// COMMAND ----------


val maevaluator = new RegressionEvaluator()
  .setLabelCol("medhvalue")
  .setPredictionCol("prediction")
  .setMetricName("mae")

// COMMAND ----------

val mae = maevaluator.evaluate(predictions)

// COMMAND ----------

predictions.select("prediction", "medhvalue", "error").show

// COMMAND ----------

predictions.describe("prediction", "medhvalue", "error").show

// COMMAND ----------

predictions.select("prediction", "medhvalue", "error").show

// COMMAND ----------

val rmse = evaluator.evaluate(predictions)
//52844
