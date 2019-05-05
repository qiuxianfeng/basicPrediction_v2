import pandas
import psycopg2
import numpy
import os 
import pyspark
import pyspark.sql.functions

if not os.path.exists('data'):
    os.makedirs('data')

def download_data_old():
    ##############last_decrypt_new###############
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    tag = True
    while tag:
        try:
            data = pandas.read_sql("select * from public.last_decrypt_new",db)
            data.to_pickle('data/last_decrypt_new')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    ##############fitting_v1###############
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    tag = True
    while tag:
        try:
            data = pan.pkldas.read_sql("select * from public.pkl.fitting_v1",db)
            data.to_pickle('data/fitting_v1')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    #############user_b5_join##############
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    number = pandas.read_sql("select max(epo_id) from sync.user_b5_join",db).as_matrix().tolist()[0][0]
    for i in range(number // 100000 + 1):
        tag = True
        while tag:
            try:
                sdata = pandas.read_sql("select * from sync.user_b5_join where epo_id>%s and epo_id<%s" % (i * 100000, (i + 1) * 100000),db)
                sdata.to_pickle('data/%s' % i)
                tag = False
                print('epoch %s finish' % i)
            except:
                print('download %s batch fail, download again' % i)
    column_name = pandas.read_pickle('data/0').columns
    total_data = pandas.DataFrame(columns=column_name)
    for i in range(number // 100000 + 1):
        total_data = total_data.append(pandas.read_pickle('data/%s' % i))
    total_data.to_pickle('data/user_b5_join')
    for i in range(number // 100000 + 1):
        os.remove('data/%s' % i)

def download_data_new():
    ###############足型数据################
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    tag = True
    while tag:
        try:
            data = pandas.read_sql("select * from fitting.fitting_v1_fp1_v10", db)
            data.to_pickle('data/foot')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    ###############楦型数据###############
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    tag = True
    while tag:
        try:
            data = pandas.read_sql("select * from fitting.fitting_v1_lp1", db)
            data.to_pickle('data/last')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    ###############对应关系#############
    db = psycopg2.connect(database="prod", user="zhaojinxi", password="Zjx1989211", host="epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn", port="5439")
    tag = True
    while tag:
        try:
            data = pandas.read_sql("select * from fitting.fitting_v1_fp1_lp1_fit", db)
            data.to_pickle('data/choose')
            tag = False
            print('finish')
        except:
            print('download fail, try again')

def make_data_size():
    foot = pandas.read_pickle('data/foot')
    last = pandas.read_pickle('data/last')
    choose = pandas.read_pickle('data/choose')

    all_foot = pandas.merge(choose, foot, how='inner', on='phone')
    all_last = pandas.merge(choose, last, how='inner', on=['styleno'])
    fit_last = all_last[all_last['basicsize_x'] == all_last['basicsize_y']]
    unfit_last = all_last[all_last['basicsize_x'] != all_last['basicsize_y']]
    fit_last = fit_last.drop(columns='basicsize_y')
    fit_last = fit_last.rename(columns={'basicsize_x':'basicsize'})
    unfit_last = unfit_last.drop(columns='basicsize_x')
    unfit_last = unfit_last.rename(columns={'basicsize_y':'basicsize'})
    positive = pandas.merge(all_foot, fit_last, how='inner', on=['phone','styleno'])
    negative = pandas.merge(all_foot, unfit_last, how='inner', on=['phone','styleno'])
    positive = positive.drop(columns=['item_id_y', 'sku_y', 'comments_y', 'answer_y', 'basicsize_y'])
    positive = positive.rename(columns={'item_id_x':'item_id', 'sku_x':'sku', 'comments_x':'comments', 'answer_x':'answer', 'basicsize_x':'basicsize'})
    positive['pick'] = 1
    negative = negative.drop(columns=['item_id_y', 'sku_y', 'comments_y', 'answer_y','basicsize_x'])
    negative = negative.rename(columns={'item_id_x':'item_id', 'sku_x':'sku', 'comments_x':'comments', 'answer_x':'answer', 'basicsize_y':'basicsize'})
    negative['pick'] = 0

    data = positive.append(negative)
    data = data.drop(columns=['item_id', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id', 'comments', 'answer', 'shoetype2', 'footstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicknessheel', 'basicsize'])
    data.to_pickle('data/size_data')
    
def make_data_suit():
    foot = pandas.read_pickle('data/foot')
    last = pandas.read_pickle('data/last')
    choose = pandas.read_pickle('data/choose')
    all_foot = pandas.merge(choose,foot,how='inner',on='phone')
    all_last = pandas.merge(choose,last,how='inner',on=['styleno','basicsize'])
    data = pandas.merge(all_foot,all_last,how='inner',on=['phone','styleno','basicsize'])
    data = data.drop(columns=['item_id_x', 'sku_x', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id_y', 'sku_y', 'comments_y', 'answer_y', 'shoetype2', 'fsku_yootstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicksku_ynessheel', 'basicsize'])
    data = data.rename(columns={'comments_x':'comsku_yments', 'answer_x':'answer'})
    data.to_pickle('data/suit_data')

def make_data():
    foot = pandas.read_pickle('data/foot')
    last = pandas.read_pickle('data/last')
    choose = pandas.read_pickle('data/choose')

    all_foot = pandas.merge(choose, foot, how='inner', on='phone')
    all_last = pandas.merge(choose, last, how='inner', on=['styleno'])
    fit_last = all_last[all_last['basicsize_x'] == all_last['basicsize_y']]
    unfit_last = all_last[all_last['basicsize_x'] != all_last['basicsize_y']]
    fit_last = fit_last.drop(columns='basicsize_y')
    fit_last = fit_last.rename(columns={'basicsize_x':'basicsize'})
    unfit_last = unfit_last.drop(columns='basicsize_x')
    unfit_last = unfit_last.rename(columns={'basicsize_y':'basicsize'})
    positive = pandas.merge(all_foot, fit_last, how='inner', on=['phone','styleno'])
    negative = pandas.merge(all_foot, unfit_last, how='inner', on=['phone','styleno'])
    positive = positive.drop(columns=['item_id_y', 'sku_y', 'comments_y', 'answer_y', 'basicsize_y'])
    positive = positive.rename(columns={'item_id_x':'item_id', 'sku_x':'sku', 'comments_x':'comments', 'answer_x':'answer', 'basicsize_x':'basicsize'})
    positive['pick'] = 1
    negative = negative.drop(columns=['item_id_y', 'sku_y', 'comments_y', 'answer_y','basicsize_x'])
    negative = negative.rename(columns={'item_id_x':'item_id', 'sku_x':'sku', 'comments_x':'comments', 'answer_x':'answer', 'basicsize_y':'basicsize'})
    negative['pick'] = 0

    data = positive.append(negative)
    data = data.drop(columns=['item_id', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id', 'shoetype2', 'footstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicknessheel'])
    data.to_pickle('data/data')    

def make_data_spark():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()

    foot = spark.read.load('hdfs:///data/foot')
    last = spark.read.load('hdfs:///data/last')
    choose = spark.read.load('hdfs:///data/choose')
    choose=choose.withColumnRenamed('basicsize','fitsize')
    all_foot=choose.join(foot, on='phone', how='inner')
    all_last=choose.join(last, on='styleno', how='inner')
    fit_last=all_last.filter(all_last['fitsize']==all_last['basicsize'])
    unfit_last=all_last.filter(all_last['fitsize']!=all_last['basicsize'])
    fit_last=fit_last.drop('basicsize')
    fit_last=fit_last.withColumnRenamed('fitsize', 'basicsize')
    unfit_last=unfit_last.drop('fitsize')
    all_foot=all_foot.withColumnRenamed('answer','answer_x')
    all_foot=all_foot.withColumnRenamed('item_id','item_id_x')
    all_foot=all_foot.withColumnRenamed('sku','sku_x')
    all_foot=all_foot.withColumnRenamed('comments','comments_x')
    positive=all_foot.join(fit_last, on=['phone', 'styleno'], how='inner')
    negative=all_foot.join(unfit_last, on=['phone', 'styleno'], how='inner')
    positive=positive.drop(*['item_id_x', 'sku_x', 'comments_x', 'answer_x', 'basicsize'])
    positive=positive.withColumnRenamed('fitsize','basicsize')
    positive=positive.withColumn('pick',pyspark.sql.functions.lit(1))
    negative=negative.drop(*['item_id_x', 'sku_x', 'comments_x', 'answer_x', 'fitsize'])
    negative=negative.withColumn('pick',pyspark.sql.functions.lit(0))
    data=positive.union(negative)
    data = data.drop(*['item_id', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id', 'shoetype2', 'footstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicknessheel'])
    data.write.save('hdfs:///data/size_data')    
    
def download_data_spark():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()
    
    ###############足型数据################
    db=spark.read.jdbc(url='jdbc:redshift://epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn/prod?user=zhaojinxi&password=Zjx1989211', table='fitting.fitting_v1_fp1_v10')
    tag = True
    while tag:
        try:
            db.write.save('hdfs:///data/foot')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    ###############楦型数据###############
    db=spark.read.jdbc(url='jdbc:redshift://epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn/prod?user=zhaojinxi&password=Zjx1989211', table='fitting.fitting_v1_lp1')
    tag = True
    while tag:
        try:
            db.write.save('hdfs:///data/last')
            tag = False
            print('finish')
        except:
            print('download fail, try again')
    ###############对应关系#############
    db=spark.read.jdbc(url='jdbc:redshift://epoque-cn.cw6lztxcsj0t.cn-north-1.redshift.amazonaws.com.cn/prod?user=zhaojinxi&password=Zjx1989211', table='fitting.fitting_v1_fp1_lp1_fit')
    tag = True
    while tag:
        try:
            db.write.save('hdfs:///data/choose')
            tag = False
            print('finish')
        except:
            print('download fail, try again')

def make_data_size_spark():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()

    foot = spark.read.load('hdfs:///data/foot')
    last = spark.read.load('hdfs:///data/last')
    choose = spark.read.load('hdfs:///data/choose')
    choose=choose.withColumnRenamed('basicsize','fitsize')
    all_foot=choose.join(foot, on='phone', how='inner')
    all_last=choose.join(last, on='styleno', how='inner')
    fit_last=all_last.filter(all_last['fitsize']==all_last['basicsize'])
    unfit_last=all_last.filter(all_last['fitsize']!=all_last['basicsize'])
    fit_last=fit_last.drop('basicsize')
    fit_last=fit_last.withColumnRenamed('fitsize', 'basicsize')
    unfit_last=unfit_last.drop('fitsize')
    all_foot=all_foot.withColumnRenamed('answer','answer_x')
    all_foot=all_foot.withColumnRenamed('item_id','item_id_x')
    all_foot=all_foot.withColumnRenamed('sku','sku_x')
    all_foot=all_foot.withColumnRenamed('comments','comments_x')
    positive=all_foot.join(fit_last, on=['phone', 'styleno'], how='inner')
    negative=all_foot.join(unfit_last, on=['phone', 'styleno'], how='inner')
    positive=positive.drop(*['item_id_x', 'sku_x', 'comments_x', 'answer_x', 'basicsize'])
    positive=positive.withColumnRenamed('fitsize','basicsize')
    positive=positive.withColumn('pick',pyspark.sql.functions.lit(1))
    negative=negative.drop(*['item_id_x', 'sku_x', 'comments_x', 'answer_x', 'fitsize'])
    negative=negative.withColumn('pick',pyspark.sql.functions.lit(0))
    data=positive.union(negative)
    data = data.drop(*['item_id', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id', 'comments', 'answer', 'shoetype2', 'footstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicknessheel', 'basicsize'])
    data.write.save('hdfs:///data/size_data')

def make_data_suit_spark():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()

    foot=spark.read.load('hdfs:///data/foot')
    last=spark.read.load('hdfs:///data/last')
    choose=spark.read.load('hdfs:///data/choose')
    all_foot=choose.join(foot, on='phone', how='inner')
    all_last=choose.join(last, on=['styleno','basicsize'], how='inner')
    all_foot=all_foot.withColumnRenamed('item_id','item_id_x')
    all_foot=all_foot.withColumnRenamed('sku','sku_x')
    all_last=all_last.withColumnRenamed('item_id','item_id_y')
    all_last=all_last.withColumnRenamed('comments','comments_y')
    all_last=all_last.withColumnRenamed('sku','sku_y')
    all_last=all_last.withColumnRenamed('answer','answer_y')   
    data=all_foot.join(all_last, on=['phone', 'styleno', 'basicsize'], how='inner')
    data=data.drop(*['item_id_x', 'sku_x', 'calf_height_left', 'calf_height_right', 'below_knee_height_left', 'below_knee_height_right', 'calf_girth_left', 'calf_girth_right', 'below_knee_girth_left', 'below_knee_girth_right', 'arch_top_height_left', 'arch_top_height_right', 'arch_top_width_left', 'arch_top_width_right', 'heel_inside_convex_length_left', 'heel_inside_convex_length_right', 'heel_outside_convex_length_left', 'heel_outside_convex_length_right', 'heel_convex_width_left', 'heel_convex_width_right', 'heel_inside_convex_width_left', 'heel_inside_convex_width_right', 'heel_outside_convex_width_left', 'heel_outside_convex_width_right', 'heel_inside_convex_edge_width_left', 'heel_inside_convex_edge_width_right', 'heel_outside_convex_edge_width_left', 'heel_outside_convex_edge_width_right', 'foot_inoutside_turn_left', 'foot_inoutside_turn_right', 'foot_inoutside_rotate_left', 'foot_inoutside_rotate_right', 'first_metatarsophalangeal_length_left', 'first_metatarsophalangeal_length_right', 'second_metatarsophalangeal_length_left', 'second_metatarsophalangeal_length_right', 'third_metatarsophalangeal_length_left', 'third_metatarsophalangeal_length_right', 'shoe_type_left', 'shoe_type_right', 'item_id_y', 'sku_y', 'comments_y', 'answer_y', 'shoetype2', 'footstruct2', 'gender', 'headform2', 'shoepadthicknessforefoot', 'shoepadthicknessheel', 'basicsize'])
    data.write.save('hdfs:///data/suit_data')

def download_sales_record():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()

    db=spark.read.jdbc(url='jdbc:mysql://rm-wz92a4ng520oq2f2no.mysql.rds.aliyuncs.com', table='bdp_products.bdp_order_8g', predicates=["order_date <= '2015-01-01'","order_date > '2015-01-01' and order_date <= '2015-06-01'","order_date > '2015-06-01' and order_date <= '2016-01-01'","order_date > '2016-01-01' and order_date <= '2016-06-01'","order_date > '2016-06-01' and order_date <= '2017-01-01'","order_date > '2017-01-01' and order_date <= '2017-06-01'","order_date > '2017-06-01' and order_date <= '2018-01-01'"], properties={'user':'root', 'password':'HfywzSRZg0a2b927'}).select('phone','sku_no','order_date')
    db.write.save('hdfs:///sales_record')

def convert_sales_record():
    spark=pyspark.sql.SparkSession.builder.getOrCreate()

    sales_record=spark.read.load('hdfs:///sales_record')
    sales_record.filter("order_date >= '2017-11-01' and order_date < '2017-12-01'").toPandas().to_pickle('sales_record_2017_11')
    sales_record.filter("order_date >= '2017-12-01' and order_date < '2018-01-01'").toPandas().to_pickle('sales_record_2017_12')

if __name__ == '__main__':
    make_data()