import numpy
import pandas
import sys
sys.path.append("..")
import preprocess_data
import autosklearn.metrics
import autosklearn.classification

train_x, train_y, test_x, test_y = preprocess_data.size_askl('shoe')

time=36000

AutoSklearnClassifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time)
AutoSklearnClassifier.fit(train_x, train_y)
pandas.to_pickle(AutoSklearnClassifier, 'model/askl')
y=AutoSklearnClassifier.predict(test_x)
score=autosklearn.metrics.accuracy(y_true=test_y, y_pred=y)
print(score)