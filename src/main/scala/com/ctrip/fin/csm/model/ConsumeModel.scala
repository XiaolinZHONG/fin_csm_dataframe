package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Created by zhongxl on 2016/10/26.
  */
class ConsumeModel extends CsmModel("consume") {

  def dataTraining(trn_data: DataFrame) = {

    /** *--------GET THE CONSUME DATA AND PRE PROCESS------------ */

    // SELECT THE COLUMNS OF CONSUME PART
    //-----------------------------------
    val consume_trn = processData(trn_data, true)
    val label = "uid_flag"
    println("This is the train data of consume_part:\n" )
    consume_trn.show(5)


    /** *--------------TRAIN THE MODEL--------------------------- */
    // import utils function

    val pipeline: PipelineModel = Utils.rfcModelling(consume_trn, label)

    serializePipeline(pipeline)
  }

  def dataPredict(tst_data: DataFrame): DataFrame = {

    /** *-------GET THE PEOPLE DATA FROM TEST DATA--------------- */

    // SELECT THE COLUMNS OF CONSUME PART
    //--------------------------------
    val consume_tst = processData(tst_data, false)

    println("This is the tst data of people_part:\n")
    consume_tst.show(5)

    /** *------------PRE PROCESSING TRAIN DATA ------------------- */

    /** *------------MODELLING AND PREDICT------------------------ */
    // import the pipeline
    val pipelineModel = deSerializePipeline()

    val prediction = pipelineModel.transform(consume_tst)
    prediction.show(5)
    val result = Utils.predictAndProbability(prediction)

    // THE PROBA_1 IS THE REAL PROBA OF GOOD LABEL
    val scoreConsume = result
      .withColumn("ScoreConsume", result("Proba_1") * 500 + 350)
      .select("uid_uid","ScoreConsume")
    val sc = tst_data.sqlContext.sparkContext
//    savePredictedData(sc,scoreConsume)
    return scoreConsume
  }

  def processData(data: DataFrame, isUsedForTraining: Boolean): DataFrame = {

    //平均消费水平
    val data_1 = data.withColumn("ord_success_order_price",
      data("ord_success_order_amount") / data("ord_success_order_count"))

    //高星酒店消费
    val data_2 = data_1.withColumn("ord_success_first_class_order_price",
      data("ord_success_first_class_order_amount") / data("ord_success_first_class_order_count"))

    //海外酒店
    val data_3 = data_2.withColumn("ord_success_aboard_order_price",
      data("ord_success_aboard_order_amount") / data("ord_success_aboard_order_count"))

    //头等舱数据
    val data_4 = data_3.withColumn("ord_success_flt_first_class_order_price",
      data("ord_success_flt_first_class_order_amount") / data("ord_success_flt_first_class_order_count"))

    //机票海外订单
    val data_5 = data_4.withColumn("ord_success_flt_aboard_order_price",
      data("ord_success_flt_aboard_order_amount") / data("ord_success_flt_aboard_order_count"))

    //机票消费单价
    val data_6 = data_5.withColumn("ord_success_flt_order_price",
      data("ord_success_flt_order_amount") / data("ord_success_flt_order_count"))

    //高星酒店
    val data_7 = data_6.withColumn("ord_success_htl_first_class_order_price",
      data("ord_success_htl_first_class_order_amount") / data("ord_success_htl_first_class_order_count"))

    //海外酒店
    val data_8 = data_7.withColumn("ord_success_htl_aboard_order_price",
      data("ord_success_htl_aboard_order_amount") / data("ord_success_htl_aboard_order_count"))

    //酒店消费单价
    val data_9 = data_8.withColumn("ord_success_htl_order_price",
      data("ord_success_htl_order_amount") / data("ord_success_htl_order_count"))

    //火车票消费
    val data_10 = data_9.withColumn("ord_success_trn_order_price",
      data("ord_success_trn_order_amount") / data("ord_success_trn_order_count"))

    if (isUsedForTraining == true) {
      val data_new = data_10.na.fill(-1.0).select("uid_flag", "pro_advanced_date", "pro_htl_star_prefer", "pro_ctrip_profits",
        "ord_success_max_order_amount", "ord_success_avg_leadtime", "ord_cancel_order_count",
        "ord_success_order_type_count", "ord_success_order_acity_count", "ord_success_flt_last_order_days",
        "ord_success_flt_max_order_amount", "ord_success_flt_avg_order_pricerate",
        "ord_success_flt_order_acity_count", "ord_success_htl_last_order_days", "ord_success_htl_max_order_amount",
        "ord_success_htl_order_refund_ratio", "ord_success_htl_guarantee_order_count",
        "ord_success_htl_noshow_order_count", "ord_cancel_htl_order_count", "ord_success_trn_last_order_days",
        "ord_success_order_price", "ord_success_first_class_order_price", "ord_success_aboard_order_price",
        "ord_success_flt_first_class_order_price", "ord_success_flt_aboard_order_price",
        "ord_success_flt_order_price", "ord_success_htl_first_class_order_price", "ord_success_htl_aboard_order_price",
        "ord_success_htl_order_price", "ord_success_trn_order_price")
      return data_new
    }
    else {
      val data_new = data_10.na.fill(0).select("uid_uid","pro_advanced_date", "pro_htl_star_prefer", "pro_ctrip_profits",
        "ord_success_max_order_amount", "ord_success_avg_leadtime", "ord_cancel_order_count",
        "ord_success_order_type_count", "ord_success_order_acity_count", "ord_success_flt_last_order_days",
        "ord_success_flt_max_order_amount", "ord_success_flt_avg_order_pricerate",
        "ord_success_flt_order_acity_count", "ord_success_htl_last_order_days", "ord_success_htl_max_order_amount",
        "ord_success_htl_order_refund_ratio", "ord_success_htl_guarantee_order_count",
        "ord_success_htl_noshow_order_count", "ord_cancel_htl_order_count", "ord_success_trn_last_order_days",
        "ord_success_order_price", "ord_success_first_class_order_price", "ord_success_aboard_order_price",
        "ord_success_flt_first_class_order_price", "ord_success_flt_aboard_order_price",
        "ord_success_flt_order_price", "ord_success_htl_first_class_order_price", "ord_success_htl_aboard_order_price",
        "ord_success_htl_order_price", "ord_success_trn_order_price")
      return data_new
    }
  }

}