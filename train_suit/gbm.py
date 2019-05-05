import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import pandas
import sys
sys.path.append("..")
import preprocess_data

train_x, train_y, test_x, test_y = preprocess_data.suit_gbm(12)

param_ensemble = {
    'learning_rate':[0.05,0.1,0.2], 
    'n_estimators':[50,100,200],
    'subsample':[0.5,0.7,0.9],
    }
param_tree = {
    'min_samples_split':[0.0001,0.001,0.01], 
    'min_samples_leaf':[0.0001,0.001,0.01], 
    'max_depth':[5,10,15], 
    }

model = sklearn.ensemble.GradientBoostingClassifier()
with open('summary/gbm_log.txt','w') as f:
    f.write('')

for times in range(5):
    x=pandas.DataFrame(train_x)
    x['y']=train_y.as_matrix()
    x=x.sample(n=10000)
    y=x['y']
    x=x.drop(columns='y')

    RandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV(model, param_distributions=param_ensemble, n_iter=5, cv=3, scoring='accuracy')
    RandomizedSearchCV.fit(x, y)
    with open('summary/gbm_log.txt','a') as f:
        f.write('round:%s, best params:%s, best score:%s\n'%(times, RandomizedSearchCV.best_params_, RandomizedSearchCV.best_score_))
    model = sklearn.ensemble.GradientBoostingClassifier(**RandomizedSearchCV.best_estimator_.get_params())

    RandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV(model, param_distributions=param_tree, n_iter=5, cv=3, scoring='accuracy')
    RandomizedSearchCV.fit(x, y)
    with open('summary/gbm_log.txt','a') as f:
        f.write('round:%s, best params:%s, best score:%s\n'%(times, RandomizedSearchCV.best_params_, RandomizedSearchCV.best_score_))
    model = sklearn.ensemble.GradientBoostingClassifier(**RandomizedSearchCV.best_estimator_.get_params())

print('final model:',model)
model = model.fit(train_x, train_y)
y=model.predict(test_x)
score=sklearn.metrics.accuracy_score(test_y, y)
print(score)

pandas.to_pickle(model,'model/gbm')