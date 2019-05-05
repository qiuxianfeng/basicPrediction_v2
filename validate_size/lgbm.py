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
            StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_lgbm_%s_%s'%(last_or_shoe, number))
            temp_data = StandardScaler.transform(data)
            model=pandas.read_pickle('../train_size/model/lgbm_%s_%s'%(last_or_shoe, number))
            predict=model.predict(temp_data)
            accuracy=sklearn.metrics.accuracy_score(predict, label)
            print('sku:',sku,' lgbm model:',number,' accuracy:',accuracy)

if __name__=='__main__':
    validate('shoe')