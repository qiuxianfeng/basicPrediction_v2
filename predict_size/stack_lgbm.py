import pandas
import json
import numpy

def predict(data, last_or_shoe):
    #加载数据预处理函数
    StandardScaler0 = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_0'%last_or_shoe)
    StandardScaler1 = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_1'%last_or_shoe)
    StandardScaler2 = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_2'%last_or_shoe)
    StandardScaler3 = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_3'%last_or_shoe)
    #加载需要的数据列名
    name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
    #加载模型
    model0=pandas.read_pickle('../train_size/model/lgbm_%s_0'%last_or_shoe)
    model1=pandas.read_pickle('../train_size/model/lgbm_%s_1'%last_or_shoe)
    model2=pandas.read_pickle('../train_size/model/lgbm_%s_2'%last_or_shoe)
    model3=pandas.read_pickle('../train_size/model/lgbm_%s_3'%last_or_shoe)
    model=pandas.read_pickle('../train_size/model/stack_lgbm_%s'%last_or_shoe)
    #从原始数据提取出需要的列名
    predict_data=data[name]
    #对数据进行预处理
    predict_data0 = StandardScaler0.transform(predict_data.values.reshape(1,-1))
    predict_data1 = StandardScaler1.transform(predict_data.values.reshape(1,-1))
    predict_data2 = StandardScaler2.transform(predict_data.values.reshape(1,-1))
    predict_data3 = StandardScaler3.transform(predict_data.values.reshape(1,-1))
    #计算中间模型结果
    predict0=model0.predict_proba(predict_data0)[0][1]
    predict1=model1.predict_proba(predict_data1)[0][1]
    predict2=model2.predict_proba(predict_data2)[0][1]
    predict3=model3.predict_proba(predict_data3)[0][1]
    #拼凑最终模型输入
    predict_data=numpy.array([predict0,predict1,predict2,predict3]).reshape(1,-1)
    #预测结果的概率
    predict=model.predict_proba(predict_data)
    #预测结果转换为json格式
    result=json.dumps({'size':predict[0][1]})

    return result

if __name__=='__main__':
    # data=pandas.read_pickle('../data/size_data')
    data=pandas.read_csv('../data/data2.csv')
    for x in range(data.shape[0]):
        temp=data.iloc[x]
        result=predict(temp, 'shoe')
        print(result)