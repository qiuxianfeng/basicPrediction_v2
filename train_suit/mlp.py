import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data

tensorflow.contrib.eager.enable_eager_execution()

question_number=6

if question_number==12:
    output_dimension=2
else:
    output_dimension=3

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/cpu:0')
parser.add_argument('--max_iteration', default=10000)
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--batch_size', default=900)
parser.add_argument('--max_learning_rate', default=0.01)
parser.add_argument('--min_learning_rate', default=0.0001)
parser.add_argument('--input_dimension', default=92)
parser.add_argument('--fc1_dimension', default=92)
parser.add_argument('--L2_ratio', default=0.0001)
parser.add_argument('--keep_prob', default=0.8)
flag, unparsed = parser.parse_known_args()

train_x, train_y, test_x, test_y = preprocess_data.suit_mlp(question_number)

with tensorflow.device(flag.device):
    w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
    b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
    w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, output_dimension], stddev=0.1), name='w6')
    b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = output_dimension), name='b6')

def model(x, is_train):
    h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
    h6 = tensorflow.matmul(h1, w6) + b6
    return h6

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
    best_score = 0

    with tensorflow.device(flag.device):
        num = train_x.shape[0] // flag.batch_size
        for i in range(flag.max_iteration):
            x = train_x[i % num * flag.batch_size:i % num * flag.batch_size + flag.batch_size,:]
            y_ = train_y[i % num * flag.batch_size:i % num * flag.batch_size + flag.batch_size,:]
            learning_rate = flag.max_learning_rate - (flag.max_learning_rate - flag.min_learning_rate) * (i / flag.max_iteration)
            optimizer._lr = learning_rate
            loss, gradient = loss_fun(x, y_, True)
            optimizer.apply_gradients(gradient)
            if i % 100 == 0:
                print("step: {}  loss: {}".format(i, loss.numpy()))
                accuracy = score(x, y_, False)
                print("step: {}  train accuracy: {}".format(i, accuracy.numpy()))
            if i % 1000 == 0:
                accuracy = score(test_x, test_y, False)
                print("step: {}  test accuracy: {}".format(i, accuracy.numpy()))
                if accuracy.numpy() > best_score:
                    best_score = accuracy.numpy()
                    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
                    Saver.save(flag.model_dir + 'mlp_%s'%question_number)

    print("best test accuracy: {}".format(best_score))

if __name__ == '__main__':
    tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)