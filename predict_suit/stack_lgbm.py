import pandas
import json
import numpy

def predict(data, question_number):
    #加载数据预处理函数
    StandardScaler0 = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_0' % question_number)
    StandardScaler1 = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_1' % question_number)
    StandardScaler2 = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_2' % question_number)
    StandardScaler3 = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_3' % question_number)
    #加载需要的左右脚数据列名
    left = pandas.read_pickle('../train_suit/intermediate/suit_left')
    right = pandas.read_pickle('../train_suit/intermediate/suit_right')
    #加载模型
    model0 = pandas.read_pickle('../train_suit/model/lgbm_%s_0' % question_number)
    model1 = pandas.read_pickle('../train_suit/model/lgbm_%s_1' % question_number)
    model2 = pandas.read_pickle('../train_suit/model/lgbm_%s_2' % question_number)
    model3 = pandas.read_pickle('../train_suit/model/lgbm_%s_3' % question_number)
    model = pandas.read_pickle('../train_suit/model/stack_lgbm_%s' % question_number)
    #从原始数据提取出需要的左右脚列名
    left_data = data[left]
    right_data = data[right]
    #对左右脚数据进行预处理
    left_data0 = StandardScaler0.transform(left_data.values.reshape(1,-1))
    left_data1 = StandardScaler1.transform(left_data.values.reshape(1,-1))
    left_data2 = StandardScaler2.transform(left_data.values.reshape(1,-1))
    left_data3 = StandardScaler3.transform(left_data.values.reshape(1,-1))
    right_data0 = StandardScaler0.transform(right_data.values.reshape(1,-1))
    right_data1 = StandardScaler1.transform(right_data.values.reshape(1,-1))
    right_data2 = StandardScaler2.transform(right_data.values.reshape(1,-1))
    right_data3 = StandardScaler3.transform(right_data.values.reshape(1,-1))
    #计算中间模型结果
    left0 = model0.predict_proba(left_data0)[0][1]
    left1 = model1.predict_proba(left_data1)[0][1]
    left2 = model2.predict_proba(left_data2)[0][1]
    left3 = model3.predict_proba(left_data3)[0][1]
    right0 = model0.predict_proba(right_data0)[0][1]
    right1 = model1.predict_proba(right_data1)[0][1]
    right2 = model2.predict_proba(right_data2)[0][1]
    right3 = model3.predict_proba(right_data3)[0][1]
    #拼凑最终模型输入
    left_data=numpy.array([left0,left1,left2,left3]).reshape(1,-1)
    right_data=numpy.array([right0,right1,right2,right3]).reshape(1,-1)
    #预测结果的概率
    predict_left = model.predict_proba(left_data)
    predict_right = model.predict_proba(right_data)
    #预测结果转换为json格式
    result = json.dumps({'left':predict_left[0].tolist(), 'right':predict_right[0].tolist()})

    return result

if __name__ == '__main__':
    size_data = pandas.read_pickle('../data/suit_data')
    data = size_data.iloc[0]
    predict(data, 2)