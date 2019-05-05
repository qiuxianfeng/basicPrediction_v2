import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import sklearn.preprocessing
import json

tensorflow.contrib.eager.enable_eager_execution()

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/cpu:0')
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--input_dimension', default=92)
parser.add_argument('--fc1_dimension', default=92)
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

def predict(data, question_number):
    StandardScaler=pandas.read_pickle('../train_suit/intermediate/StandardScaler_mlp_%s'%question_number)
    left=pandas.read_pickle('../train_suit/intermediate/suit_left')
    right=pandas.read_pickle('../train_suit/intermediate/suit_right')
    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
    Saver.restore('../train_suit/model/mlp_%s'%question_number)
    left_data=data[left]
    right_data=data[right]
    left_data = StandardScaler.transform(left_data.values.reshape(1,-1)).astype(numpy.float32)
    right_data = StandardScaler.transform(right_data.values.reshape(1,-1)).astype(numpy.float32)
    predict_left=model(left_data)
    predict_right=model(right_data)
    probability_left=tensorflow.nn.softmax(predict_left).numpy()
    probability_right=tensorflow.nn.softmax(predict_right).numpy()
    result=json.dumps({'left':float(probability_left[0][0]), 'right':float(probability_right[0][0])})
    return result

if __name__ == '__main__':
    data=pandas.read_pickle('../data/suit_data')
    data=data.iloc[0]
    predict(data, 12)