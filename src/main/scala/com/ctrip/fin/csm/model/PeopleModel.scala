package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.Utils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


/**
  * Created by zhongxl on 2016/10/19.
  */
class PeopleModel extends CsmModel("people"){


  /**
    * Select the people part data ,which should be data frame
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
    val people_trn = processData(trn_data,true)
    val label= "uid_flag"
    println("This is the train data of people_part:\n")
//    people_trn.printSchema()

    /***--------PRE PROCESSING TRAIN DATA ----------------------*/

    //this part might be used i future

    /***--------------TRAIN THE MODEL---------------------------*/
    // import utils function

    val pipeline:PipelineModel = Utils.rfcModelling(people_trn,label)

    serializePipeline(pipeline)

  }

  /**
    * Select the people part data to predict
    * @param tst_data test data (data frame)
    * @return the predict result
    */
  def dataPredict(tst_data:DataFrame):DataFrame={

    /***-------GET THE PEOPLE DATA FROM TEST DATA---------------*/

    // SELECT THE COLUMNS OF PEOPLE PART
    //--------------------------------
    val people_tst = processData(tst_data,false)

    println("This is the tst data of people_part:\n")
    people_tst.show(5)

    /***------------PRE PROCESSING TRAIN DATA -------------------*/

    /***------------MODELLING AND PREDICT------------------------*/
    // import the pipeline
    val pipelineModel = deSerializePipeline()

    val prediction = pipelineModel.transform(people_tst)
    val result = Utils.predictAndProbability(prediction)

    // THE PROBA_1 IS THE REAL PROBA OF GOOD LABEL
    val scorePeople = result
      .withColumn("ScorePeople",result("Proba_1")*500+350)
      .select("uid_uid","ScorePeople")
    val sc = tst_data.sqlContext.sparkContext
    savePredictedData(sc,scorePeople)
    return scorePeople
  }

  def processData(data:DataFrame,isUsedForTraining: Boolean):DataFrame={

    //利用正态分布的方差剔除异常值 本模块处理age
    //------------------------------------
    val maxval = 65
    val minval = 12
    val meanval= 34

    val compare = udf { (row:Int)=>
      if (row > maxval ||row < minval){
        Math.floor(meanval) // 向下取整数
      }
      else{
        row //注意这里不能写成return
      }
    }
    val juge= udf {(row:Int)=>
      if (row==0 ||row == -1){0}// 业务上未填写即为-1
      else{1}
    }
    val newData = data.withColumn("uid_age_new",compare(data("uid_age")))
      .withColumn("is_age_fill",juge(data("uid_age"))).drop("uid_age")
    //-------------------------------------------------------------------

    val newResult=Utils.outliersProcess(newData.select("uid_dealorders",
      "uid_emailvalid", "uid_mobilevalid", "uid_addressvalid",
      "uid_isindentify", "uid_authenticated_days", "uid_signupdays", "uid_signmonths",
      "uid_lastlogindays", "uid_samemobile", "ord_success_order_cmobile_count",
      "com_mobile_count", "pro_generous_stingy_tag", "pro_base_active",
      "pro_customervalue", "pro_phone_type", "pro_validpoints", "pro_htl_consuming_capacity"))

    if (isUsedForTraining == true) {
    return newData.select("uid_flag","uid_grade", "uid_dealorders_new",
      "uid_emailvalid", "uid_age_new","is_age_fill", "uid_mobilevalid", "uid_addressvalid",
      "uid_isindentify", "uid_authenticated_days_new", "uid_signupdays_new", "uid_signmonths_new",
      "uid_lastlogindays_new", "uid_samemobile", "ord_success_order_cmobile_count_new",
      "com_mobile_count_new", "pro_generous_stingy_tag", "pro_base_active",
      "pro_customervalue", "pro_phone_type", "pro_validpoints", "pro_htl_consuming_capacity")}
    else {
      return newData.select("uid_uid","uid_flag","uid_grade", "uid_dealorders_new",
        "uid_emailvalid", "uid_age_new","is_age_fill", "uid_mobilevalid", "uid_addressvalid",
        "uid_isindentify", "uid_authenticated_days_new", "uid_signupdays_new", "uid_signmonths_new",
        "uid_lastlogindays_new", "uid_samemobile", "ord_success_order_cmobile_count_new",
        "com_mobile_count_new", "pro_generous_stingy_tag", "pro_base_active",
        "pro_customervalue", "pro_phone_type", "pro_validpoints", "pro_htl_consuming_capacity")
    }
  }
}
