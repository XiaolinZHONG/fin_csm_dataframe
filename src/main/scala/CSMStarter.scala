package com.ctrip.fin.csm

import com.ctrip.fin.csm.utils.{Utils, _}
import com.ctrip.fin.csm.model._
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by zhongxl on 2016/10/13.
  */
object CSMStarter {

  var sc: SparkContext = null
  var sqlContext: SparkSession = null
  val baseOutputPath = "/user/bifinread/csm/"

  /**
    * 训练模型
    * 保存模型与Scaler对象
    *
    * @param moduleName
    * @return
    */
  def train(moduleName: String, dataPath: String) = {

    val startTime = System.currentTimeMillis()
    val data = Utils.readCsv(dataPath,sqlContext)
    moduleName match {
      case "consume"     => new ConsumeModel().dataTraining(data)
      case "finance"     => new FinanceModel().dataTraining(data)
      case "interaction" => new InteractionModel().dataTraining(data)
      case "people"      => new PeopleModel().dataTraining(data)
      case "relation"    => new RelationModel().dataTraining(data)
      case _ =>
    }
    val endTime = System.currentTimeMillis()
    println("TRAIN TIME COST： "+(endTime - startTime)+"ms")
  }

  /**
    * 预测结果
    *
    * @param moduleName
    * @param dataPath
    * @return
    */
  def predict(moduleName: String, dataPath: String) = {

    val startTime = System.currentTimeMillis()

    val data = Utils.readCsv(dataPath,sqlContext)
    data.printSchema()
    val predictDF: DataFrame = moduleName match {
      case "consume"     => new ConsumeModel().dataPredict(data)
      case "finance"     => new FinanceModel().dataPredict(data)
      case "interaction" => new InteractionModel().dataPredict(data)
      case "people"      => new PeopleModel().dataPredict(data)
      case "relation"    => new RelationModel().dataPredict(data)
    }

    val endTime = System.currentTimeMillis()
    println("PREDICT TIME COST： "+(endTime - startTime)+"ms")
    predictDF.show()
  }

  /**
    * Combine all of the score
    */
  def combineResult() = {
    val consumeDF     = new ConsumeModel().readPredictedData(sqlContext)
    val financeDF     = new FinanceModel().readPredictedData(sqlContext)
    val interactionDF = new InteractionModel().readPredictedData(sqlContext)
    val peopleDF      = new PeopleModel().readPredictedData(sqlContext)
    val relationDF    = new RelationModel().readPredictedData(sqlContext)

    consumeDF.printSchema()
    val consumeDF2 = consumeDF.filter("ScoreConsume IS NOT NULL")
    consumeDF.describe().show()
    consumeDF2.describe().show()

    consumeDF.registerTempTable("consumeScore")
    financeDF.registerTempTable("financeScore")
    interactionDF.registerTempTable("interactionScore")
    peopleDF.registerTempTable("peopleScore")
    relationDF.registerTempTable("relationScore")
//
//
    sqlContext.sql("SELECT a.uid_uid,a.ScoreConsume,b.ScoreFinance,c.ScoreInteraction,d.ScorePeople,e.ScoreRelation," +
      "   (a.ScoreConsume*0.25+b.ScoreFinance*0.2+c.ScoreInteraction*0.15+d.ScorePeople*0.3+e.ScoreRelation*0.1) AS ScoreAll " +
      " FROM consumeScore a " +
      "JOIN financeScore b     ON a.uid_uid = b.uid_uid  " +
      "JOIN interactionScore c ON a.uid_uid = c.uid_uid  " +
      "JOIN peopleScore d      ON a.uid_uid = d.uid_uid  " +
      "JOIN relationScore e    ON a.uid_uid = e.uid_uid").show()

  }


    //-----------------------MAIN FUNCTION----------------------------//
  def main(args: Array[String]): Unit = {

    val appName    = args(0)
    val appType    = args(1) // train,predict,combine
    val moduleName = args(2) // sub-model name
    val dataPath   = args(3)

    val spark = SparkSession.builder.master("local")
      .appName("spark_ml")
      .getOrCreate()

      val conf = new SparkConf().setAppName(appName).setMaster("local")
    sc = new SparkContext(conf)
    sqlContext = spark

    appType match {
      case "train"   => train(moduleName, dataPath)
      case "predict" => predict(moduleName, dataPath)
      case "combine" => combineResult()
    }
  }
  //---------------------------------------------------------------//
}
