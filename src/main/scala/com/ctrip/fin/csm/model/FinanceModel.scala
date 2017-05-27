package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/**
  * Created by zhongxl on 2016/10/26.
  */
class FinanceModel extends CsmModel("finance"){

  def dataTraining(trn_data:DataFrame)={

    /***--------GET THE PEOPLE DATA FROM TRAIN DATA------------*/

    // SELECT THE COLUMNS OF FINANCE PART
    //-----------------------------------
    val finance_trn = processData(trn_data,true)
    val label= "uid_flag"
    println("This is the train data of finance_part:\n")
    finance_trn.show(5)

    /***--------PRE PROCESSING TRAIN DATA ----------------------*/

    //this part might be used i future

    /***--------------TRAIN THE MODEL---------------------------*/
    // import utils function

    val pipeline:PipelineModel = Utils.rfcModelling(finance_trn,label)

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
    val finance_tst = processData(tst_data,false)

    println("This is the tst data of finance_part:\n")
    finance_tst.show(5)

    /***------------PRE PROCESSING TRAIN DATA -------------------*/

    /***------------MODELLING AND PREDICT------------------------*/
    // import the pipeline
    val pipelineModel = deSerializePipeline()

    val prediction = pipelineModel.transform(finance_tst)
    val result = Utils.predictAndProbability(prediction)
    val scoreFinance = result
      .withColumn("ScoreFinance",result("Proba_1")*500+350)
      .select("uid_uid","ScoreFinance")
    val sc = tst_data.sqlContext.sparkContext
    savePredictedData(sc,scoreFinance)
    return scoreFinance
  }

  def processData(data: DataFrame, isUsedForTraining: Boolean): DataFrame = {

    //携程账户余额
    val data_1 = data.withColumn("cap_balance",
      data("cap_tmoney_balance") + data("cap_wallet_balance") + data("cap_wallet_balance"))

    //付款成功率
    val data_2 = data_1.withColumn("bil_pays_ratio",
      data("bil_paysord_count") / data("bil_payord_count"))

    //信用卡付款成功率
    val data_3 = data_2.withColumn("bil_pays_credit_ratio",
      data("bil_paysord_credit_count") / data("bil_payord_credit_count"))

    //借记卡付款成功率
    val data_4 = data_3.withColumn("bil_pays_debit_ratio",
      data("bil_paysord_debit_count") / data("bil_payord_debit_count"))

    //头等消费单价
    val data_5 = data_4.withColumn("ord_success_first_class_order_price",
      data("ord_success_first_class_order_amount") / data("ord_success_first_class_order_count"))

    //海外酒店消费单价
    val data_6 = data_5.withColumn("ord_success_htl_aboard_order_price",
      data("ord_success_htl_aboard_order_amount") / data("ord_success_htl_aboard_order_count"))

    if (isUsedForTraining == true) {
      val data_new = data_6.na.fill(0)
        .select("uid_flag", "voi_complrefund_count", "fai_lackbalance", "bil_refundord_count",
          "bil_ordertype_count", "bil_platform_count", "pro_htl_star_prefer",
          "pro_htl_consuming_capacity", "pro_phone_type", "ord_success_max_order_amount",
          "ord_total_order_amount", "ord_success_flt_first_class_order_count",
          "ord_success_trn_max_order_amount", "ord_success_htl_first_class_order_count",
          "ord_success_htl_max_order_amount", "ord_success_aboard_order_count",
          "cap_balance", "ord_success_htl_aboard_order_price", "ord_success_first_class_order_price",
          "bil_pays_debit_ratio", "bil_pays_credit_ratio", "bil_pays_ratio")
      return data_new
    }
    else {
      val data_new = data_6.na.fill(0).select("uid_uid","voi_complrefund_count", "fai_lackbalance", "bil_refundord_count",
        "bil_ordertype_count", "bil_platform_count", "pro_htl_star_prefer",
        "pro_htl_consuming_capacity", "pro_phone_type", "ord_success_max_order_amount",
        "ord_total_order_amount", "ord_success_flt_first_class_order_count",
        "ord_success_trn_max_order_amount", "ord_success_htl_first_class_order_count",
        "ord_success_htl_max_order_amount", "ord_success_aboard_order_count",
        "cap_balance", "ord_success_htl_aboard_order_price", "ord_success_first_class_order_price",
        "bil_pays_debit_ratio", "bil_pays_credit_ratio", "bil_pays_ratio")
      return data_new
    }
  }
}
