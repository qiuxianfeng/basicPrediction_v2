import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data

tensorflow.contrib.eager.enable_eager_execution()

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/cpu:0')
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--batch_size', default=900)
parser.add_argument('--max_iteration', default=9000)
parser.add_argument('--max_learning_rate', default=0.01)
parser.add_argument('--min_learning_rate', default=0.0001)
parser.add_argument('--input_dimension', default=[None,147,1])
parser.add_argument('--cnn1_dimension', default=[9,1,16])
parser.add_argument('--cnn2_dimension', default=[7,16,32])
parser.add_argument('--cnn3_dimension', default=[5,32,64])
parser.add_argument('--cnn4_dimension', default=[3,64,32])
parser.add_argument('--fc5_dimension', default=16)
parser.add_argument('--output_dimension', default=2)
parser.add_argument('--keep_prob', default=0.8)
flag, unparsed = parser.parse_known_args()

train_x, train_y, test_x, test_y = preprocess_data.size_1d('shoe')

with tensorflow.device(flag.device):
    w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn1_dimension, stddev=0.1), name='w1')
    b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn1_dimension[2]), name='b1')
    w2 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn2_dimension, stddev=0.1), name='w2')
    b2 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn2_dimension[2]), name='b2')
    w3 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn3_dimension, stddev=0.1), name='w3')
    b3 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn3_dimension[2]), name='b3')
    w4 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal(flag.cnn4_dimension, stddev=0.1), name='w4')
    b4 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.cnn4_dimension[2]), name='b4')
    w5 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal((192, flag.fc5_dimension), stddev=0.1), name='w5')
    b5 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc5_dimension), name='b5')
    w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal((flag.fc5_dimension, flag.output_dimension), stddev=0.1), name='w6')
    b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

def model(x, is_train):
    h1 = tensorflow.nn.leaky_relu(tensorflow.nn.conv1d(x, w1, stride=2, padding='VALID') + b1)
    #h1 = tensorflow.reshape(h1,(-1,1,139,16))
    #h1 = tensorflow.nn.max_pool(h1, [1,1,2,1], [1,1,2,1], 'VALID')
    #h1 = tensorflow.reshape(h1,(-1,74,16))

    h2 = tensorflow.nn.leaky_relu(tensorflow.nn.conv1d(h1, w2, stride=2, padding='VALID') + b2)
    #h2 = tensorflow.reshape(h2,(-1,1,74,32))
    #h2 = tensorflow.nn.max_pool(h2, [1,1,2,1], [1,1,2,1], 'VALID')
    #h2 = tensorflow.reshape(h2,(-1,37,32))

    h3 = tensorflow.nn.leaky_relu(tensorflow.nn.conv1d(h2, w3, stride=2, padding='VALID') + b3)
    #h3 = tensorflow.reshape(h3,(-1,1,37,64))
    #h3 = tensorflow.nn.max_pool(h3, [1,1,2,1], [1,1,2,1], 'VALID')
    #h3 = tensorflow.reshape(h3,(-1,19,64))

    h4 = tensorflow.nn.leaky_relu(tensorflow.nn.conv1d(h3, w4, stride=2, padding='VALID') + b4)
    #h4 = tensorflow.reshape(h4,(-1,1,19,32))
    #h4 = tensorflow.nn.max_pool(h4, [1,1,2,1], [1,1,2,1], 'VALID')
    #h4 = tensorflow.reshape(h4,(-1,10,32))
    h4 = tensorflow.reshape(h4, (x.shape[0], -1))

    h5 = tensorflow.nn.leaky_relu(tensorflow.matmul(h4, w5) + b5)
    h5 = tensorflow.contrib.layers.dropout(h5, flag.keep_prob, is_training=is_train)

    out = tensorflow.matmul(h5, w6) + b6

    return out

@tensorflow.contrib.eager.implicit_value_and_gradients
def loss_fun(x, y_, is_train):
    y = model(x, is_train)
    loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    return loss

def score(x, y_, is_train):
    y = model(x, is_train)
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1)), tensorflow.float32))
    return accuracy

def main(_):
    optimizer = tensorflow.train.AdamOptimizer()
    best_score=0

    with tensorflow.device(flag.device):
        num = train_x.shape[0] // flag.batch_size
        for i in range(flag.max_iteration):
            x = train_x[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            y_ = train_y[i % num * flag.batch_size:i % num * flag.batch_size+flag.batch_size,:]
            learning_rate = flag.max_learning_rate - (flag.max_learning_rate - flag.min_learning_rate)*(i / flag.max_iteration)
            optimizer._lr = learning_rate
            loss, gradient = loss_fun(x, y_, True)
            optimizer.apply_gradients(gradient)
            print("step: {}  loss: {}".format(i, loss.numpy()))
            if i % 10 == 0:
                accuracy = score(x, y_, False)
                print("step: {}  train accuracy: {}".format(i, accuracy.numpy()))
            if i % 100 == 0:
                accuracy = score(test_x, test_y, False)
                print("step: {}  test accuracy: {}".format(i, accuracy.numpy()))
                if accuracy.numpy()>best_score:
                    best_score=accuracy.numpy()

    print("best accuracy: {}".format(best_score))

    Saver = tensorflow.contrib.eager.Saver([w1, w2, w3, w4, w5, w6, b1, b2, b3, b4, b5, b6])
    Saver.save(flag.model_dir + 'cnn_1d')

if __name__ == '__main__':
    tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)