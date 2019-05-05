import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import sklearn.metrics

tensorflow.contrib.eager.enable_eager_execution()

last_or_shoe = 'shoe'
all_number=[0,1,2,3]
all_test=False
if all_test==True:
    all_sku=range(0,1)
else:
    all_sku=['A','B','C','D']
all_positive=False
test_data=pandas.read_pickle('test_data_%s'%last_or_shoe)
for x in ['A','B','C','D']:
    print('%s:'%x,test_data[test_data['sku']=='%s'%x].shape[0])
print('total:',test_data.shape[0])

#all_data = pandas.read_pickle('../data/size_data')
#all_data['sku'] = all_data['sku'].apply(lambda x:x[-3])
#gb_sex = all_data.groupby(by='sex')
#for one_sex in gb_sex:
#    if one_sex[0] == 1:
#        continue
#    test_phone=pandas.read_pickle('../train_size/intermediate/test_phone_%s'%last_or_shoe)
#    test_data=one_sex[1][one_sex[1]['phone'].isin(test_phone)]
#    pandas.to_pickle(test_data,'test_data_%s'%last_or_shoe)

def validate(last_or_shoe):
    for sku in all_sku:
        if all_test==True:
            data=test_data
        else:
            data=test_data[test_data['sku']==sku]
        positive = data[data['pick'] == 1]
        negative = data[data['pick'] == 0]
        negative = negative.sample(n=positive.shape[0]*7)
        data = positive.append(negative)
        if all_positive==True:
            data=data[data['pick']==1]
        label=data['pick']
        data = data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])

        k=numpy.zeros((data.shape[0],1))
        #for times in range(4):
        #    StandardScaler = pandas.read_pickle('../train_size/intermediate/StandardScaler_mlp_%s_%s' % (last_or_shoe, times))
        #    temp_data = StandardScaler.transform(data).astype(numpy.float32)

        #    flag = None
        #    parser = argparse.ArgumentParser()
        #    parser.add_argument('--device', default='/cpu:0')
        #    parser.add_argument('--summary_dir', default='summary/')
        #    parser.add_argument('--model_dir', default='model/')
        #    parser.add_argument('--input_dimension', default=147)
        #    parser.add_argument('--fc1_dimension', default=147)
        #    parser.add_argument('--output_dimension', default=2)
        #    flag, unparsed = parser.parse_known_args()

        #    with tensorflow.device(flag.device):
        #        w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
        #        b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
        #        w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
        #        b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

        #    def model(x, is_train):
        #        h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
        #        h6 = tensorflow.matmul(h1, w6) + b6
        #        return h6

        #    Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
        #    Saver.restore('../train_size/model/mlp_%s_%s'%(last_or_shoe, times))
        #    y = model(temp_data, False)
        #    y=tensorflow.nn.softmax(y).numpy()
        #    k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        for times in range(4):
            model = pandas.read_pickle('../train_size/model/lgbm_%s_%s' % (last_or_shoe, times))
            StandardScaler = pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_%s' % (last_or_shoe, times))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        temp_data=numpy.delete(k,0,1)
        model = pandas.read_pickle('../train_size/model/stack_lgbm_%s' % (last_or_shoe))
        y = model.predict(temp_data)
        accuracy=sklearn.metrics.accuracy_score(y, label)
        print(print('stack accuracy:',accuracy))

if __name__=='__main__':
    validate('shoe')