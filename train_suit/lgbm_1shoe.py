import sklearn.model_selection
import sklearn.metrics
import lightgbm
import pandas
import numpy
import os
import sys
sys.path.append("..")
import preprocess_data

question_number=6

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

for man_or_woman in ['man','woman']:
    if man_or_woman=='man':
        all_shoe=['M','L','D']
    elif man_or_woman=='woman':
        all_shoe=['Z','D','M','L','Q']
    for shoe in all_shoe:
        train_x, train_y, test_x, test_y = preprocess_data.suit_1shoe(question_number, man_or_woman, shoe)

        model = lightgbm.sklearn.LGBMClassifier()
        model = model.fit(train_x, train_y)
        y=model.predict(test_x)
        report=sklearn.metrics.classification_report(test_y, y)

        with open("summary/lgbm_%s_%s_%s"%(question_number,man_or_woman,shoe),'w') as f:
            f.write(report)
        print(report)

        pandas.to_pickle(model,'model/lgbm_%s_%s_%s'%(question_number,man_or_woman,shoe))