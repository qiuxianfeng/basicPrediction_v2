import numpy
import tensorflow
import pandas
import json

tensorflow.enable_eager_execution()

def predict(data, last_or_shoe, man_or_woman):
    if man_or_woman=='man':
        input_dimension=6
        n=3
    elif man_or_woman=='woman':
        input_dimension=8
        n=4
    k = []
    for times in range(n):
        name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
        predict_data=data[name]
        StandardScaler = pandas.read_pickle('../train_size/intermediate/StandardScaler_mlp_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
        temp_data = StandardScaler.transform(predict_data.values.reshape(1,-1)).astype(numpy.float32)
        w1=tensorflow.get_variable('w1_%s'%times, [147, 147], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1=tensorflow.get_variable('b1_%s'%times, [147], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        w6=tensorflow.get_variable('w6_%s'%times, [147, 2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b6=tensorflow.get_variable('b6_%s'%times, [2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))  
        Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
        Saver.restore('../train_size/model/mlp_%s_%s_%s'%(last_or_shoe, times, man_or_woman))

        h1 = tensorflow.nn.relu(tensorflow.matmul(temp_data, w1) + b1)
        h6 = tensorflow.matmul(h1, w6) + b6
        y=tensorflow.nn.softmax(h6).numpy()
        k.append(y[:,1].reshape(-1,1)[0])

    for times in range(n):
        lgbm = pandas.read_pickle('../train_size/model/lgbm_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
        StandardScaler = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
        temp = StandardScaler.transform(predict_data.values.reshape(1,-1))
        y = lgbm.predict_proba(temp)
        k.append(y[:,1].reshape(-1,1)[0])

    temp_data=numpy.array(k).reshape(-1,2*n).astype(numpy.float32)

    w1_stack=tensorflow.get_variable('w1_stack', [input_dimension, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    b1_stack=tensorflow.get_variable('b1_stack', [32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    w6_stack=tensorflow.get_variable('w6_stack', [32, 2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    b6_stack=tensorflow.get_variable('b6_stack', [2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    Saver = tensorflow.contrib.eager.Saver([w1_stack, w6_stack, b1_stack, b6_stack])
    Saver.restore('../train_size/model/stack_mlp_%s_%s'%(last_or_shoe, man_or_woman))

    h1_stack = tensorflow.nn.relu(tensorflow.matmul(temp_data, w1_stack) + b1_stack)
    h6_stack = tensorflow.matmul(h1_stack, w6_stack) + b6_stack
    y_stack=tensorflow.nn.softmax(h6_stack).numpy()
    result=json.dumps({'size':float(y_stack[0][1])})

    return result

if __name__=='__main__':
    size_data=pandas.read_pickle('../data/size_data')
    data=size_data[(size_data['sex']==1) & (size_data['pick']==1)]
    for index, row in data.iterrows():
        print(predict(data.loc[index], 'shoe', 'man'))