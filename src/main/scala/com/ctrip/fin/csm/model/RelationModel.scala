package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Created by zhongxl on 2016/10/26.
  */
class RelationModel extends CsmModel("relation"){

  def dataTraining(trn_data:DataFrame)={

    /***--------GET THE RELATION DATA FROM TRAIN DATA------------*/

    // SELECT THE COLUMNS OF FINANCE PART
    //-----------------------------------
    val relation_trn = trn_data.select("uid_flag", "com_passenger_count",
      "com_idno_count","com_mobile_count", "ord_success_order_cmobile_count")

    val label= "uid_flag"
    println("This is the train data of finance_part:\n")
    relation_trn.show(5)

    /***--------PRE PROCESSING TRAIN DATA ----------------------*/

    //this part might be used i future

    /***--------------TRAIN THE MODEL---------------------------*/
    // import utils function

    val pipeline:PipelineModel = Utils.rfcModelling(relation_trn,label)

    serializePipeline(pipeline)

  }

  /**
    * Select the finance part data to predict
    *
    * @param tst_data test data (data frame)
    * @return the predict result
    */
  def dataPredict(tst_data:DataFrame):DataFrame={

    /***-------GET THE FINANCE DATA FROM TEST DATA---------------*/

    // SELECT THE COLUMNS OF FINANCE PART
    //--------------------------------
    val relation_tst = tst_data.select("uid_uid","com_passenger_count",
      "com_idno_count","com_mobile_count", "ord_success_order_cmobile_count")

    println("This is the tst data of finance_part:\n")
//    relation_tst.show(5)

    /***------------PRE PROCESSING TRAIN DATA -------------------*/

    /***------------MODELLING AND PREDICT------------------------*/
    // import the pipeline
    val pipelineModel = deSerializePipeline()

    val prediction = pipelineModel.transform(relation_tst)
    val result = Utils.predictAndProbability(prediction)
    val scoreRelation = result
      .withColumn("ScoreRelation",result("Proba_1")*500+350)
      .select("uid_uid","ScoreRelation")
    val sc = tst_data.sqlContext.sparkContext
//    savePredictedData(sc,scoreRelation)
    return scoreRelation
  }

}
