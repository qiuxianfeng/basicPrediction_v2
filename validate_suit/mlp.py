import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import sklearn.preprocessing

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

def validate(question_number):
    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
    Saver.restore('../train_suit/model/mlp_%s'%question_number)
    data, label = preprocess_data.suit_all(question_number)
    StandardScaler=pandas.read_pickle('../train_suit/intermediate/StandardScaler_mlp_%s'%question_number)
    data = StandardScaler.transform(data).astype(numpy.float32)
    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(label.as_matrix().reshape(-1,1))
    label = OneHotEncoder.transform(label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    y=model(data)
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(label, 1)), tensorflow.float32))
    print(accuracy.numpy())

if __name__ == '__main__':
    validate(12)