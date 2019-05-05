import numpy
import tensorflow
import pandas
import argparse
import sys
sys.path.append("..")
import preprocess_data
import random
import lightgbm
import sklearn.metrics

tensorflow.enable_eager_execution()

last_or_shoe = 'shoe'
man_or_woman='woman'

if man_or_woman=='man':
    n=3
elif man_or_woman=='woman':
    n=4
    
train_x, train_y, test_x, test_y = preprocess_data.size_stack_lgbm(last_or_shoe)

params = {
    'boosting_type':'dart',
    'learning_rate':0.1, 
    'n_estimators':100,
    'subsample':0.9, 
    'colsample_bytree':0.9,
    'reg_alpha':0.1,
    'reg_lambda':0.1,
    'min_child_samples':100,
    'min_child_weight':0.01, 
    'num_leaves':100,
    'class_weight':'balanced'
    }
model = lightgbm.sklearn.LGBMClassifier(**params)
model = model.fit(train_x, train_y)
y = model.predict(test_x)
score = sklearn.metrics.accuracy_score(test_y, y)
print(score)
pandas.to_pickle(model,'model/stack_lgbm_%s' % last_or_shoe)