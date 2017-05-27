package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Created by zhongxl on 2016/10/26.
  */
class InteractionModel extends CsmModel("interaction"){

  /**
    * Select the interaction part data ,which should be data frame
    * do some features operation and value operation, then transform
    * to the model method and get the pipeline
    *
    * @param trn_data train data DataFrame
    *                 +-----+------+------+
    *                 |LABEL| COL1 | COL2 |
    *                 +-----+------+------+
    *                 | 1   |   2  |  3   |
    *                 +-----+------+------+
    * @param label the tain data label which is defeault as "uid_flag"
    * @return serialized pipeline
    *         pay attention to this !
    */
  def dataTraining(trn_data:DataFrame)={

    /***--------GET THE PEOPLE DATA FROM TRAIN DATA------------*/

    // SELECT THE COLUMNS OF PEOPLE PART
    //-----------------------------------
    val interaction_trn = trn_data.select("uid_flag", "voi_complaint_count", "voi_complrefund_count",
      "voi_comment_count", "acc_loginday_count", "pro_validpoints",
      "pro_base_active", "pro_ctrip_profits", "pro_customervalue")

    val label= "uid_flag"
    println("This is the train data of interaction_part:\n")
    interaction_trn.show(5)

    /***--------PRE PROCESSING TRAIN DATA ----------------------*/

    //this part might be used i future

    /***--------------TRAIN THE MODEL---------------------------*/
    // import utils function

    val pipeline:PipelineModel = Utils.gbtcModelling(interaction_trn,label)

    serializePipeline(pipeline)

  }

  /**
    * Select the interaction part data to predict
    *
    * @param tst_data test data (data frame)
    * @return the predict result
    */
  def dataPredict(tst_data:DataFrame):DataFrame={

    /***-------GET THE PEOPLE DATA FROM TEST DATA---------------*/

    // SELECT THE COLUMNS OF PEOPLE PART
    //--------------------------------
    val interaction_tst = tst_data.select("uid_uid","voi_complaint_count", "voi_complrefund_count",
      "voi_comment_count", "acc_loginday_count", "pro_validpoints",
      "pro_base_active", "pro_ctrip_profits", "pro_customervalue")

    println("This is the tst data of interaction_part:\n")
    interaction_tst.show(5)

    /***------------PRE PROCESSING TRAIN DATA -------------------*/

    /***------------MODELLING AND PREDICT------------------------*/
    // import the pipeline
    val pipelineModel = deSerializePipeline()

    val prediction = pipelineModel.transform(interaction_tst)
//    val result = Utils.predictAndProbability(prediction)



    // THE PROBA_1 IS THE REAL PROBA OF GOOD LABEL
    val scoreInteraction = prediction
      .withColumn("ScoreInteraction",prediction("predictedLabel")*500+350)
      .select("uid_uid","ScoreInteraction")

    val sc = tst_data.sqlContext.sparkContext
    savePredictedData(sc,scoreInteraction)
    return scoreInteraction
  }

}
