import tensorflow
import sys
sys.path.append("..")
import preprocess_data
import pandas
import random

all_phone = pandas.read_pickle('intermediate/all_phone')
test_phone = random.sample(all_phone, 824)
train_phone = list(set(all_phone) - set(test_phone))
pandas.to_pickle(train_phone, 'intermediate/train_phone')
pandas.to_pickle(test_phone, 'intermediate/test_phone')
train_x, train_y, test_x, test_y=preprocess_data.regress_mlp(train_phone, test_phone)

batch_size=500
max_iter=30000
max_learn_rate=0.01
min_learn_rate=0.0001
in_dim=147
fc1_dim=200
fc2_dim=100
out_dim=5

def model(x):
    w1 = tensorflow.Variable(tensorflow.truncated_normal([in_dim,fc1_dim], stddev=0.1),name='w1')
    b1 = tensorflow.Variable(tensorflow.constant(0.1,shape=[fc1_dim]),name='b1')
    o1 = tensorflow.nn.leaky_relu(tensorflow.matmul(x,w1) + b1)

    w2 = tensorflow.Variable(tensorflow.truncated_normal([fc1_dim,fc2_dim], stddev=0.1),name='w2')
    b2 = tensorflow.Variable(tensorflow.constant(0.1,shape=[fc2_dim]),name='b2')
    o2 = tensorflow.nn.leaky_relu(tensorflow.matmul(o1,w2) + b2)

    w3 = tensorflow.Variable(tensorflow.truncated_normal([fc2_dim,out_dim], stddev=0.1),name='w2')
    b3 = tensorflow.Variable(tensorflow.constant(0.1,shape=[out_dim]),name='b2')
    o3 =tensorflow.add(tensorflow.matmul(o2,w3),b3 ,name='y')

    return o3

x = tensorflow.placeholder(tensorflow.float32, [None, in_dim], name='x')
y_ = tensorflow.placeholder(tensorflow.float32, [None, out_dim], name='y_')
iterate = tensorflow.placeholder(tensorflow.int32, name='iterate')
learn_rate = tensorflow.placeholder(tensorflow.float32, name='learn_rate')

y = model(x)

loss=tensorflow.reduce_mean([tensorflow.reduce_mean(tensorflow.reduce_sum(tensorflow.abs(y_-y), 1)), tensorflow.reduce_mean(tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(y_-y), 1)))], name='loss')

optimize = tensorflow.train.AdamOptimizer(learn_rate).minimize(loss)

Saver = tensorflow.train.Saver(max_to_keep=1)
max_loss = 10000

with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())

    tensorflow.summary.scalar('test loss', loss)
    merged = tensorflow.summary.merge_all()
    writer = tensorflow.summary.FileWriter('summary', sess.graph)

    num = train_x.shape[0] // batch_size
    for i in range(max_iter):
        temp_x = train_x[i % num * batch_size:i % num * batch_size + batch_size,:]
        temp_y = train_y[i % num * batch_size:i % num * batch_size + batch_size,:]
        lr = max_learn_rate - (max_learn_rate - min_learn_rate) * (i / max_iter)
        sess.run(optimize, feed_dict={x:temp_x, y_:temp_y, learn_rate:lr})
        if i % 100 == 0 and i > 0:
            train_loss = loss.eval(feed_dict={x:temp_x, y_:temp_y})
            print('step %d, training loss %g' % (i, train_loss))
        if i % 1000 == 0 and i > 0:
            test_loss = loss.eval(feed_dict={x:test_x, y_:test_y})
            print("step %d, test loss %g"%(i, test_loss))
            summary = sess.run(merged, feed_dict={x:test_x, y_:test_y})
            writer.add_summary(summary, i)
            if test_loss < max_loss:
                max_loss = test_loss
                Saver.save(sess,'model/mlp')
                
print("best loss: {}".format(max_loss))