import pandas
import sklearn.metrics
import sys
sys.path.append("..")
import preprocess_data
import numpy

def validate(question_number, man_or_woman):
    data, label = preprocess_data.suit_all(question_number)
    StandardScaler=pandas.read_pickle('../train_suit/intermediate/StandardScaler_lgbm_%s_%s'%(question_number, man_or_woman))
    data = StandardScaler.transform(data)
    model=pandas.read_pickle('../train_suit/model/lgbm_%s_%s'%(question_number, man_or_woman))
    predict=model.predict(data)
    print(model.predict_proba(data))
    print(sklearn.metrics.accuracy_score(predict, label))

if __name__=='__main__':
    validate(2, 'woman')