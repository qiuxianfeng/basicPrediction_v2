import pandas
import json
import sklearn.metrics

def predict(data, question_number=12):
    #加载数据预处理函数
    StandardScaler=pandas.read_pickle('../train_suit/intermediate/StandardScaler_askl_%s'%question_number)
    #加载需要的左右脚数据列名
    left=pandas.read_pickle('../train_suit/intermediate/suit_left')
    right=pandas.read_pickle('../train_suit/intermediate/suit_right')
    #加载模型
    model=pandas.read_pickle('../train_suit/model/askl')
    #从原始数据提取出需要的左右脚列名
    left_data=data[left]
    right_data=data[right]
    #对左右脚数据进行预处理
    left_data = StandardScaler.transform(left_data.values.reshape(1,-1))
    right_data = StandardScaler.transform(right_data.values.reshape(1,-1))
    #预测结果
    predict_left=model.predict(left_data)
    predict_right=model.predict(right_data)
    #预测结果转换为json格式
    result=json.dumps({'left':int(predict_left[0]), 'right':int(predict_right[0])})

    return result

if __name__=='__main__':
    data=pandas.read_pickle('../data/suit_data')
    data=data.iloc[0]
    predict(data, 12)