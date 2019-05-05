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

# all_phone = pandas.read_pickle('intermediate/all_phone_woman')
# all_select = pandas.read_pickle('intermediate/all_%s_woman' % last_or_shoe)
# test_phone = random.sample(all_phone, 224)
# pandas.to_pickle(test_phone, 'intermediate/test_phone_%s_woman'%last_or_shoe)
# all_phone = list(set(all_phone) - set(test_phone))
# train2_phone=random.sample(all_phone, 800)
# pandas.to_pickle(train2_phone, 'intermediate/train2_phone_%s_woman'%last_or_shoe)
# all_phone = list(set(all_phone) - set(train2_phone))
# all_phone = pandas.Series(all_phone)
# all_select = pandas.Series(all_select)
# for times in range(4):
#    validate1_phone = all_phone[2200 * times:(2200 * times + 2200)].as_matrix().tolist()
#    train1_phone=list(set(all_phone.as_matrix().tolist())-set(validate1_phone))
#    pandas.to_pickle(train1_phone, 'intermediate/train1_phone_%s_%s_woman'%(last_or_shoe, times))
#    pandas.to_pickle(validate1_phone, 'intermediate/validate1_phone_%s_%s_woman'%(last_or_shoe, times))
#    unselect = all_select[(len(all_select) // 4) * times:((len(all_select) // 4) * times + (len(all_select) // 4))].as_matrix().tolist()
#    pandas.to_pickle(unselect, 'intermediate/train1_unselect_%s_%s_woman'%(last_or_shoe, times))

# all_phone = pandas.read_pickle('intermediate/all_phone_man')
# all_select = pandas.read_pickle('intermediate/all_%s_man' % last_or_shoe)
# test_phone = random.sample(all_phone, 90)
# pandas.to_pickle(test_phone, 'intermediate/test_phone_%s_man'%last_or_shoe)
# all_phone = list(set(all_phone) - set(test_phone))
# train2_phone=random.sample(all_phone, 340)
# pandas.to_pickle(train2_phone, 'intermediate/train2_phone_%s_man'%last_or_shoe)
# all_phone = list(set(all_phone) - set(train2_phone))
# all_phone = pandas.Series(all_phone)
# all_select = pandas.Series(all_select)
# for times in range(3):
#    validate1_phone = all_phone[1210 * times:(1210 * times + 1210)].as_matrix().tolist()
#    train1_phone=list(set(all_phone.as_matrix().tolist())-set(validate1_phone))
#    pandas.to_pickle(train1_phone, 'intermediate/train1_phone_%s_%s_man'%(last_or_shoe, times))
#    pandas.to_pickle(validate1_phone, 'intermediate/validate1_phone_%s_%s_man'%(last_or_shoe, times))
#    unselect = all_select[(len(all_select) // 3) * times:((len(all_select) // 3) * times + (len(all_select) // 3))].as_matrix().tolist()
#    pandas.to_pickle(unselect, 'intermediate/train1_unselect_%s_%s_man'%(last_or_shoe, times))

train_x, train_y, test_x, test_y = preprocess_data.size_stack_mlp(last_or_shoe)

flag = None
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/cpu:0')
parser.add_argument('--max_iteration', default=5100)
parser.add_argument('--summary_dir', default='summary/')
parser.add_argument('--model_dir', default='model/')
parser.add_argument('--batch_size', default=900)
parser.add_argument('--repeat', default=20)
parser.add_argument('--max_learning_rate', default=0.01)
parser.add_argument('--min_learning_rate', default=0.0001)
parser.add_argument('--input_dimension', default=8)
parser.add_argument('--fc1_dimension', default=32)
parser.add_argument('--output_dimension', default=2)
parser.add_argument('--L2_ratio', default=0.0001)
parser.add_argument('--keep_prob', default=0.8)
flag, unparsed = parser.parse_known_args()

@tensorflow.contrib.eager.implicit_value_and_gradients
def loss_fun(x, y_, is_train):
    y = model(x, is_train)
    loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    return loss

def score(x, y_, is_train):
    y = model(x, is_train)
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1)), tensorflow.float32))
    return accuracy

def model(x, is_train):
    h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
    h6 = tensorflow.matmul(h1, w6) + b6
    return h6

with tensorflow.device(flag.device):
    w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
    b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
    w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
    b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

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
        if i % 100 == 0 and i > 0:
            print("step: {}  loss: {}".format(i, loss.numpy()))
            accuracy = score(x, y_, False)
            print("step: {}  train accuracy: {}".format(i, accuracy.numpy()))
        if i % 1000 == 0 and i > 0:
            accuracy = score(test_x, test_y, False)
            print("step: {}  test accuracy: {}".format(i, accuracy.numpy()))
            if accuracy.numpy() > best_score:
                best_score = accuracy.numpy()
                Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
                Saver.save(flag.model_dir + 'stack_mlp_%s' % (last_or_shoe))

print("best accuracy: {}".format(best_score))