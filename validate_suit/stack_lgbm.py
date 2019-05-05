import pandas
import sklearn.metrics
import sys
sys.path.append("..")
import preprocess_data
import numpy

def validate(question_number):
    data, label = preprocess_data.suit_all(question_number)
    k=numpy.zeros((data.shape[0],1))
    for times in range(4):
        model=pandas.read_pickle('../train_suit/model/lgbm_%s_%s'%(question_number, times))
        StandardScaler = pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_%s'%(question_number, times))
        temp=StandardScaler.transform(data)
        y=model.predict_proba(temp)
        k=numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)
    data=numpy.delete(k,0,1)
    model=pandas.read_pickle('../train_suit/model/stack_lgbm_%s'%question_number)
    predict=model.predict(data)
    print(sklearn.metrics.accuracy_score(predict, label))

if __name__=='__main__':
    validate(2)