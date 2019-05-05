import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import sklearn.preprocessing
import json

tensorflow.enable_eager_execution()

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/cpu:0')
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--input_dimension', default=147)
parser.add_argument('--fc1_dimension', default=147)
parser.add_argument('--output_dimension', default=2)
flag, unparsed = parser.parse_known_args()

with tensorflow.device(flag.device):
    w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
    b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
    w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
    b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

def model(x):
    h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
    h6 = tensorflow.matmul(h1, w6) + b6
    return h6

def predict(data, last_or_shoe):
    StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_mlp_%s'%last_or_shoe)
    name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
    Saver.restore('../train_size/model/mlp_%s'%last_or_shoe)
    predict_data=data[name]
    predict_data = StandardScaler.transform(predict_data.values.reshape(1,-1)).astype(numpy.float32)
    predict=model(predict_data)
    probability=tensorflow.nn.softmax(predict).numpy()
    result=json.dumps({'size':float(probability[0][1])})
    return result

if __name__ == '__main__':
    size_data=pandas.read_pickle('../data/size_data')
    data=size_data.iloc[0]
    predict(data, 'shoe')