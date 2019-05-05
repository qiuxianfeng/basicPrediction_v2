import pandas
import sklearn.metrics
import sys
sys.path.append("..")
import preprocess_data
import numpy

def validate(question_number):
    data, label = preprocess_data.suit_all(question_number)
    label=label.astype(int)
    StandardScaler=pandas.read_pickle('../train_suit/intermediate/StandardScaler_askl_%s'%question_number)
    data = StandardScaler.transform(data)
    model=pandas.read_pickle('../train_suit/model/askl')
    predict=model.predict(data)
    print(sklearn.metrics.accuracy_score(predict, label))

if __name__=='__main__':
    validate(12)