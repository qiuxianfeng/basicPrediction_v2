import pandas
import json

def predict(data, last_or_shoe='shoe'):
    #加载数据预处理函数
    StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_askl_%s'%last_or_shoe)
    #加载需要的数据列名
    name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
    #加载模型
    model=pandas.read_pickle('../train_size/model/askl')
    #从原始数据提取出需要的列名
    predict_data=data[name]
    #对数据进行预处理
    predict_data = StandardScaler.transform(predict_data.values.reshape(1,-1))
    #预测结果
    predict=model.predict(predict_data)
    #预测结果转换为json格式
    result=json.dumps({'size':int(predict[0])})

if __name__=='__main__':
    size_data=pandas.read_pickle('../data/size_data')
    data=size_data.iloc[0]
    predict(data, 'shoe')