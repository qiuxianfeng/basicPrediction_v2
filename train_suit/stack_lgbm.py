import sklearn.model_selection
import sklearn.metrics
import lightgbm
import pandas
import numpy
import os
import sys
sys.path.append("..")
import preprocess_data

n = 100000
n_iter = 50
question_number = 6
all_phone = pandas.read_pickle('../train_size/intermediate/all_phone')
all_select = pandas.read_pickle('../train_size/intermediate/all_last')
all_phone = pandas.Series(all_phone)
all_select = pandas.Series(all_select)

params = {
    'boosting_type':['dart'],
    'learning_rate':[0.05,0.1,0.2], 
    'n_estimators':[50,100,200],
    'subsample':[0.5,0.7,0.9], 
    'colsample_bytree':[0.5,0.7,0.9],
    'reg_alpha':[0.01, 0.1, 1],
    'reg_lambda':[0.01, 0.1, 1],
    'min_child_samples':[100,200,300],
    'min_child_weight':[0.001,0.01,0.1], 
    'num_leaves':[1000,1500,2000],
    'class_weight':['balanced']
    }

for times in range(4):
    phone = all_phone[2456 * times:(2456 * times + 2456)].as_matrix().tolist()
    select = all_select[(len(all_select) // 4) * times:((len(all_select) // 4) * times + (len(all_select) // 4))].as_matrix().tolist()
    train_x, train_y, test_x, test_y = preprocess_data.suit_lgbm_ditermine(question_number, phone, select, times)

    model = lightgbm.sklearn.LGBMClassifier()

    x = pandas.DataFrame(train_x)
    x['y'] = train_y.as_matrix()
    x = x.sample(n=n)
    y = x['y']
    x = x.drop(columns='y')
    RandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=3, scoring='accuracy')
    RandomizedSearchCV.fit(x, y)
    with open('summary/lgbm_log_%s.txt' % times,'w') as f:
        f.write('%s' % RandomizedSearchCV.cv_results_)

    model = lightgbm.sklearn.LGBMClassifier(**RandomizedSearchCV.best_estimator_.get_params())
    print('final model:',model)
    model = model.fit(train_x, train_y)
    y = model.predict(test_x)
    score = sklearn.metrics.accuracy_score(test_y, y)
    print(score)

    pandas.to_pickle(model,'model/lgbm_%s_%s' % (question_number, times))

train_x, train_y, test_x, test_y = preprocess_data.suit_stack_randomsearch_lgbm(question_number)

model = lightgbm.sklearn.LGBMClassifier()
x = pandas.DataFrame(train_x)
x['y'] = train_y.as_matrix()
x = x.sample(n=n)
y = x['y']
x = x.drop(columns='y')
RandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=3, scoring='accuracy')
RandomizedSearchCV.fit(x, y)

model = lightgbm.sklearn.LGBMClassifier(**RandomizedSearchCV.best_estimator_.get_params())
print('stack model:',model)
model = model.fit(train_x, train_y)
y = model.predict(test_x)
score = sklearn.metrics.accuracy_score(test_y, y)
print(score)
pandas.to_pickle(model,'model/stack_lgbm_%s' % (question_number))