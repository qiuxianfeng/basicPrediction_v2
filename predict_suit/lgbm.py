import pandas
import json

def predict(data, question_number, man_or_woman):
    #加载数据预处理函数
    StandardScaler = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_%s_Q' % (question_number,man_or_woman))
    #加载需要的左右脚数据列名
    left = pandas.read_pickle('../train_suit/intermediate/suit_left')
    right = pandas.read_pickle('../train_suit/intermediate/suit_right')
    #加载模型
    model = pandas.read_pickle('../train_suit/model/lgbm_%s_%s_Q' % (question_number,man_or_woman))
    #从原始数据提取出需要的左右脚列名
    left_data = data[left]
    right_data = data[right]
    #对左右脚数据进行预处理
    left_data = StandardScaler.transform(left_data.values.reshape(1,-1))
    right_data = StandardScaler.transform(right_data.values.reshape(1,-1))
    #预测结果
    predict_left = model.predict_proba(left_data)
    predict_right = model.predict_proba(right_data)
    #预测结果转换为json格式
    result = json.dumps({'left':predict_left[0].tolist(), 'right':predict_right[0].tolist()})

    return result

if __name__ == '__main__':
    question_number=2
    data = pandas.read_csv('../data/sdf.csv')
    for temp in data.iterrows():
        print(predict(temp[1], question_number, 'woman'))
        with open('问题2.txt','a') as f:
            f.write(predict(temp[1], question_number, 'woman')+'\n')