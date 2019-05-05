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
all_test=True
if all_test==True:
    all_sku=range(0,1)
else:
    all_sku=['A','B','C','D']
all_positive=False
test_data=pandas.read_pickle('test_data_%s'%last_or_shoe)
for x in ['A','B','C','D']:
    print('%s:'%x,test_data[test_data['sku']=='%s'%x].shape[0])
print('total:',test_data.shape[0])

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
        for number in all_number:
            StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_mlp_%s_%s'%(last_or_shoe, number))
            temp_data = StandardScaler.transform(data).astype(numpy.float32)
            OneHotEncoder=pandas.read_pickle('../train_size/intermediate/OneHotEncoder_mlp_%s_%s'%(last_or_shoe, number))
            temp_label=OneHotEncoder.transform(label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)

            flag = None
            parser = argparse.ArgumentParser()
            parser.add_argument('--device', default='/cpu:0')
            parser.add_argument('--summary_dir', default='summary/')
            parser.add_argument('--model_dir', default='model/')
            parser.add_argument('--input_dimension', default=147)
            parser.add_argument('--fc1_dimension', default=147)
            parser.add_argument('--output_dimension', default=2)
            flag, unparsed = parser.parse_known_args()

            with tensorflow.device(flag.device):
                w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
                b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
                w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
                b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

            def model(x, is_train):
                h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
                h6 = tensorflow.matmul(h1, w6) + b6
                return h6

            def score(x, y_, is_train):
                y = model(x, is_train)
                accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1)), tensorflow.float32))
                return accuracy

            Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
            Saver.restore('../train_size/model/mlp_%s_%s'%(last_or_shoe, number))
            with tensorflow.device(flag.device):
                accuracy = score(temp_data, temp_label, False)
                print('sku:',sku,' mlp model:',number,' accuracy:',accuracy.numpy())

if __name__=='__main__':
    validate('shoe')