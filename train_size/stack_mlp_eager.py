import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import random
import lightgbm
import sklearn.metrics

tensorflow.enable_eager_execution()

last_or_shoe = 'shoe'
man_or_woman='woman'
max_iteration=10001
summary_dir='summary/'
model_dir='model/'
batch_size=900
max_learning_rate=0.01
min_learning_rate=0.0001
fc1_dimension=32
output_dimension=2

if man_or_woman=='man':
    input_dimension=6
elif man_or_woman=='woman':
    input_dimension=8

train_x, train_y, test_x, test_y = preprocess_data.size_stack_mlp_eager(last_or_shoe, man_or_woman)

w1=tensorflow.get_variable('w1_stack', [input_dimension, fc1_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
b1=tensorflow.get_variable('b1_stack', [fc1_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
w6=tensorflow.get_variable('w6_stack', [fc1_dimension, output_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
b6=tensorflow.get_variable('b6_stack', [output_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))   

def model(x):
    h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
    h6 = tensorflow.matmul(h1, w6) + b6
    return h6

def loss_fun(x, y_):
    y = model(x)
    loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    return loss

def score(x, y_):
    y = model(x)
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1)), tensorflow.float32))
    return accuracy 

optimizer = tensorflow.train.AdamOptimizer()
best_score = 0

bin_size = train_x.shape[0] // batch_size
for i in range(max_iteration):
    x = train_x[i % bin_size * batch_size:i % bin_size * batch_size + batch_size,:]
    y_ = train_y[i % bin_size * batch_size:i % bin_size * batch_size + batch_size,:]
    learning_rate = max_learning_rate - (max_learning_rate - min_learning_rate) * (i / max_iteration)
    optimizer._lr = learning_rate
    with tensorflow.GradientTape() as gt:
        loss = loss_fun(x, y_)
    grads = gt.gradient(loss, [w1,b1,w6,b6])
    optimizer.apply_gradients(zip(grads, [w1,b1,w6,b6])) 
    if i % 100 == 0 and i > 0:
        print("step: {}  loss: {}".format(i, loss.numpy()))
        accuracy = score(x, y_)
        print("step: {}  train accuracy: {}".format(i, accuracy.numpy()))
    if i % 1000 == 0 and i > 0:
        accuracy = score(test_x, test_y)
        print("step: {}  test accuracy: {}".format(i, accuracy.numpy()))
        if accuracy.numpy() > best_score:
            best_score = accuracy.numpy()
            Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
            Saver.save(model_dir + 'stack_mlp_%s_%s' % (last_or_shoe, man_or_woman))

print("best accuracy: {}".format(best_score))