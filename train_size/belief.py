import numpy
import tensorflow
import pandas
import sys
sys.path.append("..")
import preprocess_data
import sklearn.metrics
import json
import lightgbm

tensorflow.contrib.eager.enable_eager_execution()

last_or_shoe = 'shoe'

#name=pandas.read_pickle('intermediate/size_dimension_name')
#data=pandas.read_pickle('../data/size_data_with_size')
#data=data[data['sex']==2]
#test_phone=pandas.read_pickle('intermediate/test_phone_%s'%last_or_shoe)
#data=data[data['phone'].isin(test_phone)]
#model3_train=[]
#model3_label=[]
#gb=data.groupby(by=['styleno', 'phone'])
#for x in gb:
#    one=x[1].sort_values(by='basicsize')
#    suitsize=one['pick'].values
#    #assert one.shape[0]==8
#    one=one[name]
#    k=numpy.zeros([one.shape[0],1])
#    for times in range(4):
#        StandardScaler = pandas.read_pickle('intermediate/StandardScaler_mlp_%s_%s' % (last_or_shoe, times))
#        temp_data = StandardScaler.transform(one).astype(numpy.float32)

#        with tensorflow.device('/cpu:0'):
#            w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([147, 147], stddev=0.1), name='w1')
#            b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = 147), name='b1')
#            w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([147, 2], stddev=0.1), name='w6')
#            b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = 2), name='b6')

#        Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
#        Saver.restore('../train_size/model/mlp_%s_%s'%(last_or_shoe, times))
#        h1 = tensorflow.nn.relu(tensorflow.matmul(temp_data, w1) + b1)
#        h6 = tensorflow.matmul(h1, w6) + b6
#        y=tensorflow.nn.softmax(h6).numpy()
#        k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

#    for times in range(4):
#        lgbm = pandas.read_pickle('../train_size/model/lgbm_%s_%s' % (last_or_shoe, times))
#        StandardScaler = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_%s' % (last_or_shoe, times))
#        temp = StandardScaler.transform(one)
#        y = lgbm.predict_proba(temp)
#        k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

#    temp_data=numpy.delete(k,0,1).astype(numpy.float32)

#    with tensorflow.device('/cpu:0'):
#        w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([8, 32], stddev=0.1), name='w1')
#        b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = 32), name='b1')
#        w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([32, 2], stddev=0.1), name='w6')
#        b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = 2), name='b6')

#    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
#    Saver.restore('../train_size/model/stack_mlp_%s'%(last_or_shoe))
#    h1 = tensorflow.nn.relu(tensorflow.matmul(temp_data, w1) + b1)
#    h6 = tensorflow.matmul(h1, w6) + b6
#    y=tensorflow.nn.softmax(h6).numpy()
#    outcome = tensorflow.cast(tensorflow.equal(tensorflow.argmax(y[:,1]), tensorflow.argmax(suitsize)), tensorflow.float32)
#    model3_train.append(y[:,1].tolist())
#    model3_label.append(outcome.numpy())

#model3_train=numpy.array(model3_train)
#model3_label=numpy.array(model3_label)

model3_train=pandas.read_pickle('model3_train')
model3_label=pandas.read_pickle('model3_label')

model3_train=pandas.DataFrame(model3_train)
model3_train['label']=model3_label
model3_train['label']=model3_train['label'].astype(int)
positive=model3_train[model3_train['label']==1]
negative=model3_train[model3_train['label']==0]
positive=positive.sample(n=positive.shape[0])
negative=negative.sample(n=negative.shape[0])
test1=positive.iloc[:287,:]
test2=negative.iloc[:228,:]
train1=positive.iloc[287:,:]
train2=negative.iloc[228:,:]
train=train1.append(train2)
train=train.sample(n=train.shape[0])
test=test1.append(test2)
test=test.sample(n=test.shape[0])
train_label=train['label']
test_label=test['label']
train=train.drop(columns='label')
test=test.drop(columns='label')

params = {
    'boosting_type':'dart',
    #'learning_rate':0.1, 
    #'n_estimators':100,
    #'subsample':0.9, 
    #'colsample_bytree':0.9,
    #'reg_alpha':0.1,
    #'reg_lambda':0.1,
    #'min_child_samples':200,
    #'min_child_weight':0.01, 
    #'num_leaves':500,
    'class_weight':'balanced'
    }

model = lightgbm.sklearn.LGBMClassifier()
model = model.fit(train, train_label)
y = model.predict(test)
score = sklearn.metrics.accuracy_score(test_label, y)
print(score)