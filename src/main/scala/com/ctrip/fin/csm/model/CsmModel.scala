package com.ctrip.fin.csm.model

import java.io.{FileOutputStream, ObjectOutputStream, _}

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.reflect.io.Path


/**
  * Created by zyong on 2016/10/20.
  */
abstract class CsmModel(module: String) extends Serializable {

  val baseOutputPath: String = "/home/bifinread/csm_2"
  val modulePath: String = s"${baseOutputPath}/${module}"
  val modelPath: String = s"${modulePath}/model.obj"

  val savedDataFormat: String = "parquet"
  val predictedDataPath: String = s"${modulePath}/predict"

  //FOLLOWS ARE USED IN THE MODULE PART TO SAVE THE PIPELINE
  def serializePipeline(pipeline: PipelineModel)={

    //clear the the output/save directory
    val outPath: Path = Path(modulePath)
//    //try (outPath.deleteRecursively)
    outPath.createDirectory(true, true)

    val oos = new ObjectOutputStream(new FileOutputStream(modelPath))
    oos.writeObject(pipeline)
    oos.flush()
    oos.close()
  }

  //FOLLOWS ARE USED IN THE MODULE PART TO DELIVER THE PIPELINE
  def deSerializePipeline[T]():PipelineModel={
    val ois = new ObjectInputStream(new FileInputStream(modelPath))
    ois.readObject().asInstanceOf[PipelineModel]
  }


  /**
    * 存储打过分的数据
    * @param sc
    * @param data
    */
  def savePredictedData(sc: SparkContext, data: DataFrame) = {
    Utils.deleteHdfsPath(sc, predictedDataPath)
    data.write.format(savedDataFormat).save(predictedDataPath)
  }

  /**
    * 读取打过分的数据
    * @param sqlContext
    * @return
    */
  def readPredictedData(sqlContext: SQLContext): DataFrame = {
    sqlContext.read.format(savedDataFormat).load(predictedDataPath)
  }
}
