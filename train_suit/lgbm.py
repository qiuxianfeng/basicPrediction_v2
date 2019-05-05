import sklearn.model_selection
import sklearn.metrics
import lightgbm
import pandas
import numpy
import os
import sys
sys.path.append("..")
import preprocess_data

question_number=2
man_or_woman='woman'

train_x, train_y, test_x, test_y = preprocess_data.suit_lgbm(question_number, man_or_woman)

params = {
    'boosting_type':'dart',
    'learning_rate':0.1, 
    'n_estimators':100,
    'subsample':0.9, 
    'colsample_bytree':0.9,
    'reg_alpha':0.1,
    'reg_lambda':0.1,
    'min_child_samples':200,
    'min_child_weight':0.01, 
    'num_leaves':500,
    'class_weight':'balanced'}

model = lightgbm.sklearn.LGBMClassifier()
model = model.fit(train_x, train_y)
y=model.predict(test_x)
score=sklearn.metrics.accuracy_score(test_y, y)
print(score)
pandas.to_pickle(model,'model/lgbm_%s_%s'%(question_number,man_or_woman))