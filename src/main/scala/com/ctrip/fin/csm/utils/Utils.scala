package com.ctrip.fin.csm.utils

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor, RandomForestRegressor}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._

/**
  * Created by zhongxl on 2016/10/19.
  */

object Utils extends Serializable{

  /**
    * Using data bricks to read csv file
    * print the data schema and head 5 data
    *
    * @param address CSV file address
    * @param sqlContext
    * @return DataFrame
    */
  def readCsv (address:String,sqlContext: SparkSession):DataFrame={

    val df= sqlContext.read
      .format("csv")//you should have installed the 'data bricks' csv package
      .option("header", "true")//if data have header
      .option("inferSchema", "true") //规范格式类型
      .load(address)

    println("Follows are the train data："+df.count())
    df.show(5)//print the first 5 rows
    println("-"*100)
    df.printSchema()
    //    println("-"*70)
    return df
  }

  /**
    * Read json file
    * print the data schema and head 5 data
    *
    * @param address the json file path;
    * @param sqlContext
    * @return Dataframe
    */
  def readJson(address:String,sqlContext: SQLContext):DataFrame={

    val df = sqlContext.read.json(address)
    println("Follows are the train data："+df.count())
    df.show(5)//print the first 5 rows
    println("-"*70)
    df.printSchema()
    println("-"*70)
    return df
  }

  def readHive(table: String, sc: SparkContext): DataFrame = {
    val hiveContext = new HiveContext(sc)
    val df = hiveContext.sql("FROM table SELECT *")
    return df
  }



  //-----------------------------------------------------------------//

  /**
    * Using the Random Forest model to regression the data
    *                 +-----+------+------+
    *                 |LABEL| COL1 | COL2 |
    *                 +-----+------+------+
    *                 | 1   |   2  |  3   |
    *                 +-----+------+------+
    * @param data data frame style
    * @return pipeline
    */
  def rfrModelling (data: DataFrame,flag:String):PipelineModel={
    /***
      * Algorithm Flow:
      * read data->
      * create pipeline (chain the data pre processing function)->
      * pipeline fit train data ->
      * pipeline transform test data ->
      * evaluate the predict result
      */

    /***--------------------------READ AND INDEXED DATA-----------------------------*/
    /***----------------------------------------------------------------------------*/

    // -----------------------------------------------------------------+
    // PAY ATTENTION!!!
    // THE DATA FRAME STYLE DATA SHOULD HAVE TWO COLUMNS LIKE THIS:
    // +-----+----------+-------------+
    // |LABEL| FEATURES | COL1 | COL2 |
    // +-----+----------+------+------+
    // | 0.0 |[0,2,1...]|  0   |  2   |
    // +-----+----------+------+------+
    // -----------------------------------------------------------------+

    // the flag column should be Double Type
    val df = data.withColumn("Label",data(flag)*1.0).drop(flag)


    // get the columns names (ARRAY)
    val args = data.drop(flag).columns

    // create the new columns in vectors
    //-----------TRANSFORM THE DATA TO 'LABEL FEATURES' STYLE-----------+
    val vectorAssembler = new VectorAssembler()
      .setInputCols(args)
      .setOutputCol("features")

    //if you read an 'libsvm' file as data frame,you should indexed the columns

    /***----------------------------SPLIT THE DATA----------------------------------*/
    /***----------------------------------------------------------------------------*/

    //random split the data
    val Array(trn_data,tst_data) = df.randomSplit(Array(0.7,0.3))

    /***-------------------TRAIN THE RF MODEL AND CREATE PIPELINE-------------------*/
    /***----------------------------------------------------------------------------*/

    // creat the random forest model
    val rfModel = new RandomForestRegressor()
      .setLabelCol("Label")
      .setFeaturesCol("features")
      .setNumTrees(30) //set the tree number
      .setMaxDepth(10) //set the tree depth
      .setImpurity("variance") //add noise to improve the robust

    // chain indexer and forest buy using pipeline
    val pipeline = new Pipeline().setStages(Array(vectorAssembler,rfModel))

    // train the model
    // the process included the indexing process and fit model process
    val modelRFR = pipeline.fit(trn_data)

    /***------------------USING PIPELINE PREDICT THE TEST DATA----------------------*/
    /***----------------------------------------------------------------------------*/

    // cross valid the test data
    // return data frame style result
    val prediction = modelRFR.transform(tst_data) //Dataframe

    //print the first 5 predict result if you need
    prediction.select("prediction","Label").show(5)

    /***---------------EVALUATE THE MODEL / COMPUTE THE TEST ERROR -----------------*/
    /***----------------------------------------------------------------------------*/


    // evaluate the regression result buy using
    // formula: RMSE=(prediction-label)^2
    // during this model we should use the calssification

    //create the evaluate function
    val evaluator = new RegressionEvaluator()
      .setLabelCol("Label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(prediction)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    /***-----------------PRINT THE FRAMEWORK OF THE MODEL---------------------------*/
    /***----------------------------------------------------------------------------*/


    // if you want to analysis the whole framework of the model, this may be useful
    // i don't suggest you to using this part

    //    val Model = model_RF.stages(1).asInstanceOf[RandomForestRegressionModel]
    //    println("Learned regression forest model:\n" + Model.toDebugString)

    /***----------------------------SAVE THE PIEPLINE-------------------------------*/
    /***----------------------------------------------------------------------------*/


    //    model_RF.save(model_path)//!!!
    // this feature is not support in spark 1.6 and will supported in spark 2.0

    return modelRFR
    /***----------------------------------END---------------------------------------*/
    /***----------------------------------------------------------------------------*/

  }

  /**
    * Using the GBDT model to regression the data
    *                 +-----+------+------+
    *                 |LABEL| COL1 | COL2 |
    *                 +-----+------+------+
    *                 | 1   |   2  |  3   |
    *                 +-----+------+------+
    * @param data data frame style
    * @return pipeline
    */
  def gbtrModelling(data: DataFrame,flag:String):PipelineModel={
    /***
      * Algorithm Flow:
      * read data->
      * create pipeline (chain the data pre processing function)->
      * pipeline fit train data ->
      * pipeline transform test data ->
      * evaluate the predict result
      */



    // input the label name
//    val flag="uid_flag"

    /***--------------------------READ AND INDEXED DATA-----------------------------*/
    /***----------------------------------------------------------------------------*/


    // -----------------------------------------------------------------+
    // PAY ATTENTION!!!
    // THE DATA FRAME STYLE DATA SHOULD HAVE TWO COLUMNS LIKE THIS:
    // +-----+----------+
    // |LABEL| FEATURES |
    // +-----+----------+
    // | 0.0 |[0,2,1...]|
    // +-----+----------+
    // OR THE DATA FRAME SHOULD BE USE 'VECTOR ASSEMBLER' TO TRANSFORM:
    // DF=
    // +-----+---------------------+
    // |LABEL|       features      |
    // +-----+---------------------+
    // |LABEL| COLUMN 0 | COLUMN 1 | THE COLUMN CAN BE OPERATED DIRECTLY
    // +-----+---------------------+
    // | 1.0 |   0      |     3    |
    // +-----+----------+----------+
    // -----------------------------------------------------------------+

    // the flag column should be Double Type
    val df=data.withColumn("Label",data(flag)*1.0).drop(flag)

    //-----------TRANSFORM THE DATA TO 'LABEL FEATURES' STYLE-----------+

    // get the columns names
    val args = data.drop(flag).columns

    // create the
    val vectorAssembler = new VectorAssembler()
      .setInputCols(args)
      .setOutputCol("features")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setMax(1.0).setMin(0.0) // this will scaler the data to range(0,1)

    /***----------------------------SPLIT THE DATA----------------------------------*/
    /***----------------------------------------------------------------------------*/


    //random split the data
    val Array(trn_data,tst_data) = df.randomSplit(Array(0.7,0.3))

    /***-------------------TRAIN THE RF MODEL AND CREATE PIPELINE-------------------*/
    /***----------------------------------------------------------------------------*/


    // creat the random forest model
    val gbt_model = new GBTRegressor()
      .setLabelCol("Label")
      .setFeaturesCol("scaledFeatures")
      .setMaxIter(30)//Param for maximum number of iterations
      .setMaxDepth(15) //set the tree depth
//      .setImpurity("variance") //add noise to improve the robust


    // chain indexer and forest buy using pipeline
    val pipeline = new Pipeline().setStages(Array(vectorAssembler,scaler,gbt_model))

    // train the model
    // the process included the indexing process and fit model process
    val model_GBT = pipeline.fit(trn_data)

    /***------------------USING PIPELINE PREDICT THE TEST DATA----------------------*/
    /***----------------------------------------------------------------------------*/


    // cross valid the test data
    // return data frame style result
    val prediction = model_GBT.transform(tst_data) //Dataframe

    //print the first 5 predict result if you need
    prediction.select("prediction","Label").show(5)

    /***---------------EVALUATE THE MODEL / COMPUTE THE TEST ERROR -----------------*/
    /***----------------------------------------------------------------------------*/


    // evaluate the regression result buy using
    // formula: RMSE=(prediction-label)^2

    //create the evaluate function
    val evaluator = new RegressionEvaluator()
      .setLabelCol("Label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(prediction)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    /***-----------------PRINT THE FRAMEWORK OF THE MODEL---------------------------*/
    /***----------------------------------------------------------------------------*/


    // if you want to analysis the whole framework of the model, this may be useful
    // i don't suggest you to using this part

//    val Model = model_GBT.stages(1).asInstanceOf[GBTRegressionModel]
//    println("Learned regression forest model:\n" + Model.toDebugString)

    /***----------------------------SAVE THE PIEPLINE-------------------------------*/
    /***----------------------------------------------------------------------------*/


    //    model_GBT.save(model_path)//!!!
    // this feature is not support in spark 1.6 and will supported in spark 2.0
    return model_GBT
  }

  /**
    * Using the random forest classification model to classify
    *                 +-----+------+------+
    *                 |LABEL| COL1 | COL2 |
    *                 +-----+------+------+
    *                 | 1   |   2  |  3   |
    *                 +-----+------+------+
    * @param data data frame
    * @return pipeline
    */
  def rfcModelling (data: DataFrame,flag:String):PipelineModel={
    /***
      * Algorithm Flow:
      * read data->
      * create pipeline (chain the data pre processing function)->
      * pipeline fit train data ->
      * pipeline transform test data ->
      * evaluate the predict result
      */



    // input the label name
//    val flag=label

    /***--------------------------READ AND INDEXED DATA-----------------------------*/
    /***----------------------------------------------------------------------------*/

    // -----------------------------------------------------------------+
    // PAY ATTENTION!!!
    // THE DATA FRAME STYLE DATA SHOULD HAVE TWO COLUMNS LIKE THIS:
    // +-----+----------+
    // |LABEL| FEATURES |
    // +-----+----------+
    // | 0.0 |[0,2,1...]|
    // +-----+----------+
    // -----------------------------------------------------------------+

    // the flag column should be Double Type
    val df=data.withColumn("label",data(flag)*1.0).drop(flag)


    // you can show the new data frame's head
        df.show(5)
    //    df.printSchema()

    //-----------TRANSFORM THE DATA TO 'LABEL FEATURES' STYLE-----------+

    // get the columns names
    val args = data.drop(flag).columns

    // create the vector features
    val vectorAssembler = new VectorAssembler()
      .setInputCols(args)
      .setOutputCol("features")


    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setMax(1.0).setMin(0.0) // this will scaler the data to range(0,1)


    // the random forest model need to indexed the label
    // REMEMBER TO USE IndexString to transform back the index label
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("index_label")

    //if you read an 'libsvm' file as data frame,you should indexed the columns

    /***----------------------------SPLIT THE DATA----------------------------------*/
    /***----------------------------------------------------------------------------*/


    //random split the data
    val Array(trn_data,tst_data) = df.randomSplit(Array(0.7,0.3))

    /***-------------------TRAIN THE RF MODEL AND CREATE PIPELINE-------------------*/
    /***----------------------------------------------------------------------------*/


    // creat the random forest model
    val rf_model = new RandomForestClassifier()
      .setLabelCol("index_label")
      .setFeaturesCol("scaledFeatures")
      .setNumTrees(30) //set the tree number
      .setMaxDepth(20) //set the tree depth
      .setImpurity("gini") //add noise to improve the robust
    //      .setProbabilityCol("probability")

    // Use the IndexToString to transform the index label
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexer.fit(df).labels)

    // chain indexer and forest buy using pipeline
    val pipeline = new Pipeline()
      .setStages(Array(vectorAssembler,scaler,indexer,rf_model,labelConverter))

    // train the model
    // the process included the indexing process and fit model process
    val modelRFC = pipeline.fit(trn_data)

    /***------------------USING PIPELINE PREDICT THE TEST DATA----------------------*/
    /***----------------------------------------------------------------------------*/


    // cross valid the test data
    // return data frame style result
    val prediction = modelRFC.transform(df) //Dataframe

    //print the first 5 predict result if you need
    prediction.show(10)

    /***---------------EVALUATE THE MODEL / COMPUTE THE TEST ERROR -----------------*/
    /***----------------------------------------------------------------------------*/

    classificationReport(prediction)

    /***-----------------PRINT THE FRAMEWORK OF THE MODEL---------------------------*/
    /***----------------------------------------------------------------------------*/


    // if you want to analysis the whole framework of the model, this may be useful
    // i don't suggest you to using this part

    //    val Model = model_RF.stages(1).asInstanceOf[RandomForestRegressionModel]
    //    println("Learned regression forest model:\n" + Model.toDebugString)

    /***----------------------------SAVE THE PIEPLINE-------------------------------*/
    /***----------------------------------------------------------------------------*/


    //    model_RF.save(model_path)//!!!
    // this feature is not support in spark 1.6 and will supported in spark 2.0

    return modelRFC
    /***----------------------------------END---------------------------------------*/
    /***----------------------------------------------------------------------------*/

  }

  /**
    * Using the GBDT model to classification the data
    *                 +-----+------+------+
    *                 |LABEL| COL1 | COL2 |
    *                 +-----+------+------+
    *                 | 1   |   2  |  3   |
    *                 +-----+------+------+
    * @param data data frame style
    * @return pipeline
    *         GBTC has no probability columns
    */
  def gbtcModelling(data: DataFrame,flag:String):PipelineModel={
    /***
      * Algorithm Flow:
      * read data->
      * create pipeline (chain the data pre processing function)->
      * pipeline fit train data ->
      * pipeline transform test data ->
      * evaluate the predict result
      */

    /***--------------------------READ AND INDEXED DATA-----------------------------*/
    /***----------------------------------------------------------------------------*/

    // -----------------------------------------------------------------+
    // PAY ATTENTION!!!
    // THE DATA FRAME STYLE DATA SHOULD HAVE TWO COLUMNS LIKE THIS:
    // +-----+----------+
    // |LABEL| FEATURES |
    // +-----+----------+
    // | 0.0 |[0,2,1...]|
    // +-----+----------+
    // -----------------------------------------------------------------+

    // the flag column should be Double Type
    val df=data.withColumn("label",data(flag)*1.0).drop(flag)

    //-----------TRANSFORM THE DATA TO 'LABEL FEATURES' STYLE-----------+

    // get the columns names
    val args = data.drop(flag).columns

    // create the
    val vectorAssembler = new VectorAssembler()
      .setInputCols(args)
      .setOutputCol("features")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setMax(1.0).setMin(0.0) // this will scaler the data to range(0,1)

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("index_label")

    /***----------------------------SPLIT THE DATA----------------------------------*/
    /***----------------------------------------------------------------------------*/

    //random split the data
    val Array(trn_data,tst_data) = df.randomSplit(Array(0.7,0.3))

    /***-------------------TRAIN THE RF MODEL AND CREATE PIPELINE-------------------*/
    /***----------------------------------------------------------------------------*/

    // creat the random forest model
    val gbt_model = new GBTClassifier()
      .setLabelCol("index_label")
      .setFeaturesCol("scaledFeatures")
      .setMaxIter(50)//Param for maximum number of iterations
      .setMaxDepth(10) //set the tree depth
//      .setImpurity("entropy") //add noise to improve the robust

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexer.fit(df).labels)

    // chain indexer and forest buy using pipeline
    val pipeline = new Pipeline().setStages(Array(vectorAssembler,scaler,indexer,gbt_model,labelConverter))

    // train the model
    // the process included the indexing process and fit model process
    val model_GBTC = pipeline.fit(trn_data)

    /***------------------USING PIPELINE PREDICT THE TEST DATA----------------------*/
    /***----------------------------------------------------------------------------*/

    // cross valid the test data
    // return data frame style result
    val prediction = model_GBTC.transform(tst_data) //Dataframe

    //print the first 5 predict result if you need
    prediction.select("predictedLabel","label").show(5)

    /***---------------EVALUATE THE MODEL / COMPUTE THE TEST ERROR -----------------*/
    /***----------------------------------------------------------------------------*/

    classificationReport(prediction)

    /***-----------------PRINT THE FRAMEWORK OF THE MODEL---------------------------*/
    /***----------------------------------------------------------------------------*/

    // if you want to analysis the whole framework of the model, this may be useful
    // i don't suggest you to using this part

    /***----------------------------SAVE THE PIEPLINE-------------------------------*/
    /***----------------------------------------------------------------------------*/
    //    model_GBT.save(model_path)//!!!
    // this feature is not support in spark 1.6 and will supported in spark 2.0

    return model_GBTC
  }

  //------------------------------------------- ----------------------//

  /**
    * Classification evaluate
    * @param prediction the result data frame (LABEL, PREDICTION)
    */
  def classificationReport(prediction:DataFrame)={

    /*** EVALUATE THE CLASSIFICATION RESULT */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("index_label")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(prediction)
    println("Test Error = " + (1.0 - accuracy))

    // TRANSFORM THE PREDICTION TO RDD
    //---------------------------------------------------------------
    val prediction_temp=prediction.select("index_label","prediction")

    val predictionRdd=prediction_temp.rdd.map(row=>(row.getDouble(0),row.getDouble(1)))

    val metrics=new BinaryClassificationMetrics(predictionRdd)
    //-------------------------------------------
//    val score=metrics.scoreAndLabels
//    score.foreach{ case (t,p) =>
//      println(s"label: $t,prediction: $p")
//    }
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }
    // Precision-Recall Curve
    //      val PRC = metrics.pr
    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-1 score: $f, Beta = 1")
    }
    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-0.5 score: $f, Beta = 0.5")
    }
    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)
    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)
    // ROC Curve
    //      val roc = metrics.roc
    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
    //--------------------------------------------
  }

  /**
    * USint the rdd to create the data frame to concat multi data frame
    *
    * @param data_1 Data frame
    * @param data_2 data frame LIST
    * @return the new contacted data frame
    */
  def concatDataFrame(data_1: DataFrame,data_2: DataFrame*):DataFrame={
    var resultDF: DataFrame = data_1
    for (df <- data_2) {
      val rows = resultDF.rdd.zip(df.rdd).map {
        case (rowLeft, rowRight) => Row.fromSeq(rowLeft.toSeq ++ rowRight.toSeq)
      }
      val schema = StructType(resultDF.schema.fields ++ df.schema.fields)
      resultDF = resultDF.sqlContext.createDataFrame(rows, schema)
    }
    return resultDF
  }

  /**
    * select and transform the prediction data frame
    *
    * @param prediction the prediction data
    * @return the positive probability data frame
    */
  def predictAndProbability(prediction: DataFrame):DataFrame={
    // show the prediction
    //--------------------------------------------------------------
//    prediction.select("predictedLabel","probability").show(5)
    //    prediction.printSchema()

    import prediction.sqlContext.implicits._
    //create data frame need to import this implicits
    // show the probability
    // the probability col is vector[n_p,p_p]
    // you should import the 'org.apache.spark.mllib.linalg.Vector' manually
    //------------------------------------------------
    val probability = prediction.select("uid_uid","probability")
    val proba = probability.map{case Row(row1:String,row:Vector)=>(row1,row(0),row(1))}
      .toDF("uid_uid","Proba_1","Proba_2")


    //show the result
//    proba.printSchema()
    proba.show()

    return proba// return the positive probability
  }

  def deleteHdfsPath(sc: SparkContext, path: String) = {
    val hadoopConf = sc.hadoopConfiguration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val hdfsPath = new Path(path)
    if (hdfs.exists(hdfsPath)) {
      //hdfs.delete(hdfsPath, true)
      System.exit(-1)
    }
  }

  def outliersBoxProcess(dataFrame: DataFrame): DataFrame = {
    var newdata = dataFrame
    for (flag <- dataFrame.columns.toList) {
      val quantiles = dataFrame.select(flag)
        .stat.approxQuantile(flag, Array(0.25, 0.5, 0.75), 0.1)
      val q1 = quantiles(0)
      val median = quantiles(1)
      val q3 = quantiles(2)
      val iqr = q3 - q1
      val lowerRange = q1 - 1.5 * iqr
      val upperRange = q3 + 1.5 * iqr
      println("FOLLOWS ARE THE OUTLIER DATA OF " + flag)
      dataFrame.select(flag).filter(s"$flag<$lowerRange or $flag > $upperRange").show(3)
      val compare = udf { (row: Int) =>
        if (row > upperRange || row < lowerRange) {Math.floor(median)}
        else {row}
      }
      val juge = udf { (row: Int) =>
        if (row == -1) {0}
        else {1}}
      newdata = newdata.withColumn(flag + "_new", compare(col(flag)))
        .withColumn(flag + "TF", juge(col(flag))).drop(flag)
      //注意这种更新DataFrame方式
    }
    return newdata
  }

  //    println("-" * 50)
  //    outliersBoxProcess(df.select("uid_age", "uid_grade")).show(5)

  def normalDataProcess(dataFrame: DataFrame): DataFrame = {
    var newData = dataFrame
    for (flag <- dataFrame.columns.toList) {
      val describes = dataFrame.describe(flag)
      //        describes.show()
      val meanVal = describes.take(3)(1).toSeq(1).toString.toDouble
      val stdVal = describes.take(3)(2).toSeq(1).toString.toDouble
      val maxVal = meanVal + 3 * stdVal
      val minVal = meanVal - 3 * stdVal
      val compare = udf { (row: Int) =>
        if (row > maxVal || row < minVal) {Math.floor(meanVal)}
        else {row}}
      val juge = udf { (row: Int) =>
        if (row == -1) {0}
        else {1}}
      newData = newData
        .withColumn(flag + "_new", compare(col(flag))).drop(flag)
    }
    return newData
  }

  //    normalDataProcess(df.select("uid_age", "uid_grade")).show(5)

  def outliersProcess(dataFrame: DataFrame, replace: Int = 95): DataFrame = {
    var newData = dataFrame
    for (flag <- dataFrame.columns.toList) {
      val quantiles = dataFrame.select(flag)
        .stat.approxQuantile(flag, Array(0.5, 0.95), 0.1)
      val q95 = quantiles(1)
      val q5 = quantiles(0)
      val compare95 = udf { (row: Int) =>
        if (row > q95) {Math.floor(q95)}
        else if (row == -1) {q5}
        else {row}}
      val compare5 = udf { (row: Int) =>
        if (row > q95) {Math.floor(q5)}
        else if (row == -1) {q5}
        else {row}}

      if (replace == 5) {
        newData = newData.withColumn(flag + "_new", compare5(col(flag))).drop(flag)
      }
      else {
        newData = newData.withColumn(flag + "_new", compare95(col(flag))).drop(flag)
      }
    }
    return newData
  }


}
