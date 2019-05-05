import pandas
import json

def predict(data, man_or_woman, shoe):
    #加载数据预处理函数
    StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_%s'%(man_or_woman,shoe))
    #加载需要的数据列名
    name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
    #加载模型
    model=pandas.read_pickle('../train_size/model/lgbm_%s_%s'%(man_or_woman,shoe))
    #从原始数据提取出需要的列名
    predict_data=data[name]
    #对数据进行预处理
    predict_data = StandardScaler.transform(predict_data.values.reshape(1,-1))
    #预测结果的概率
    predict=model.predict_proba(predict_data)
    #预测结果转换为json格式
    result=json.dumps({'size':predict[0][1]})

    return result

if __name__=='__main__':
    size_data=pandas.read_csv('k.csv')
    for x in size_data.iterrows():
        print(predict(x[1],'woman','M'))