import sklearn.model_selection
import sklearn.metrics
import lightgbm
import pandas
import numpy
import os
import sys
sys.path.append("..")
import preprocess_data
import random

last_or_shoe = 'shoe'
man_or_woman='woman'

if man_or_woman=='man':
    n=3
elif man_or_woman=='woman':
    n=4
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
    #'class_weight':'balanced'
    }

for times in range(n):
    train1_phone = pandas.read_pickle('intermediate/train1_phone_%s_%s_%s' % (last_or_shoe, times,man_or_woman))
    validate1_phone = pandas.read_pickle('intermediate/validate1_phone_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
    train1_unselect = pandas.read_pickle('intermediate/train1_unselect_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
    train_x, train_y, test_x, test_y = preprocess_data.size_lgbm_ditermine(last_or_shoe, man_or_woman, train1_phone, validate1_phone, train1_unselect, times)

    model = lightgbm.sklearn.LGBMClassifier(**params)
    model = model.fit(train_x, train_y)
    y = model.predict(test_x)
    score = sklearn.metrics.accuracy_score(test_y, y)
    print(score)

    pandas.to_pickle(model,'model/lgbm_%s_%s_%s' % (last_or_shoe, times,man_or_woman))