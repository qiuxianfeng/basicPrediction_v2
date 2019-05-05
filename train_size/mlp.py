import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import tensorflow

last_or_shoe = 'shoe'
man_or_woman='woman'

if man_or_woman=='man':
    n=3
elif man_or_woman=='woman':
    n=4
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=900)
parser.add_argument('--max_iteration',type=int,default=10001)
parser.add_argument('--max_learning_rate',type=float,default=0.01)
parser.add_argument('--min_learning_rate',type=float,default=0.0001)
parser.add_argument('--input_dimension',type=int,default=147)
parser.add_argument('--fc1_dimension',type=int,default=147)
parser.add_argument('--output_dimension',type=int,default=2)
FLAGS,unparsed = parser.parse_known_args()

w1 = tensorflow.Variable(tensorflow.truncated_normal([FLAGS.input_dimension, FLAGS.fc1_dimension], stddev=0.1),name='w1')
b1 = tensorflow.Variable(tensorflow.constant(0.1,shape=[FLAGS.fc1_dimension]),name='b1')
w6 = tensorflow.Variable(tensorflow.truncated_normal([FLAGS.fc1_dimension, FLAGS.output_dimension], stddev=0.1),name='w6')
b6 = tensorflow.Variable(tensorflow.constant(0.1,shape=[FLAGS.output_dimension]),name='b6')

def model(x,keep_prob):
    z1 = tensorflow.matmul(x,w1) + b1
    fc1 = tensorflow.nn.relu(z1)
    fc1 = tensorflow.nn.dropout(fc1,keep_prob)

    y = tensorflow.add(tensorflow.matmul(fc1, w6), b6, name='y')
    return y

for times in range(n):
    train1_phone = pandas.read_pickle('intermediate/train1_phone_%s_%s_%s' % (last_or_shoe, times,man_or_woman))
    validate1_phone = pandas.read_pickle('intermediate/validate1_phone_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
    train1_unselect = pandas.read_pickle('intermediate/train1_unselect_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
    train_x, train_y, test_x, test_y = preprocess_data.size_mlp_ditermine(last_or_shoe, man_or_woman, train1_phone, validate1_phone, train1_unselect, times)

    x = tensorflow.placeholder(tensorflow.float32,[None,FLAGS.input_dimension], name='x')
    y_ = tensorflow.placeholder(tensorflow.float32,[None,FLAGS.output_dimension], name='y_')
    iteration = tensorflow.placeholder(tensorflow.int32)
    keep_prob = tensorflow.placeholder(tensorflow.float32, name='keep_prob')
    lr = tensorflow.placeholder(tensorflow.float32)

    y = model(x, keep_prob)

    loss=tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    optimize = tensorflow.train.AdamOptimizer(lr).minimize(loss)
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1)), tensorflow.float32))

    Saver = tensorflow.train.Saver(max_to_keep=1)
    max_acc = 0

    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        num = train_x.shape[0] // FLAGS.batch_size
        for i in range(FLAGS.max_iteration):
            temp_train = train_x[i % num * FLAGS.batch_size:i % num * FLAGS.batch_size + FLAGS.batch_size,:]
            temp_label = train_y[i % num * FLAGS.batch_size:i % num * FLAGS.batch_size + FLAGS.batch_size,:]
            learning_rate = FLAGS.max_learning_rate - (FLAGS.max_learning_rate - FLAGS.min_learning_rate) * (i / FLAGS.max_iteration)
            sess.run(optimize, feed_dict={x:temp_train, y_:temp_label, keep_prob:0.8, lr:learning_rate})
            if i % 100 == 0 and i > 0:
                train_accuracy = accuracy.eval(feed_dict={x:temp_train, y_:temp_label, keep_prob:1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            if i % 1000 == 0 and i > 0:
                validation_accuracy = accuracy.eval(feed_dict={x:test_x, y_:test_y, keep_prob:1.0})
                print("step %d, validation accuracy %g"%(i, validation_accuracy))
                if validation_accuracy > max_acc:
                    max_acc = validation_accuracy
                    Saver.save(sess,'model/mlp_%s_%s_%s'%(last_or_shoe, times,man_or_woman))
    print("best accuracy: {}".format(max_acc))