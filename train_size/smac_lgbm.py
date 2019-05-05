import smac.configspace
import ConfigSpace.conditions
import ConfigSpace.hyperparameters
import smac.scenario.scenario
import smac.facade.smac_facade
import sklearn.model_selection
import sklearn.metrics
import lightgbm
import pandas
import numpy
import os
import sys
sys.path.append("..")
import preprocess_data

last_or_shoe='shoe'
train_x, train_y, test_x, test_y = preprocess_data.size_lgbm(last_or_shoe)
x=pandas.DataFrame(train_x)
x['y']=train_y.as_matrix()
x=x.sample(n=1000)
y=x['y']
x=x.drop(columns='y')

def evaluate(param):
    model = lightgbm.LGBMClassifier(**param)
    score = sklearn.model_selection.cross_val_score(model, x, y, cv=3)
    return 1-numpy.mean(score)

params={
    'boosting_type':['gbdt','dart'],
    'learning_rate':[0.05,0.1,0.2], 
    'n_estimators':[50,100,200],
    'subsample':[0.5,0.7,0.9], 
    'colsample_bytree':[0.5,0.7,0.9],
    'reg_alpha':[0.01, 0.1, 1],
    'reg_lambda':[0.01, 0.1, 1],
    'min_child_samples':[20,100,200],
    'min_child_weight':[0.001,0.01,0.1], 
    'num_leaves':[100,1000,5000],
    }

ConfigurationSpace = smac.configspace.ConfigurationSpace()
boosting_type = ConfigSpace.hyperparameters.CategoricalHyperparameter("boosting_type", ["gbdt", "dart"], default_value="gbdt")
learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter("learning_rate", 0.05, 0.2, default_value=0.1)
subsample=ConfigSpace.hyperparameters.UniformFloatHyperparameter("subsample", 0.5, 0.9, default_value=0.7)
colsample_bytree=ConfigSpace.hyperparameters.UniformFloatHyperparameter("colsample_bytree", 0.5, 0.9, default_value=0.7)
reg_alpha=ConfigSpace.hyperparameters.UniformFloatHyperparameter("reg_alpha", 0.01, 1, default_value=0.1)
reg_lambda=ConfigSpace.hyperparameters.UniformFloatHyperparameter("reg_lambda", 0.01, 1, default_value=0.1)
min_child_weight=ConfigSpace.hyperparameters.UniformFloatHyperparameter("min_child_weight", 0.001, 0.1, default_value=0.01)
num_leaves=ConfigSpace.hyperparameters.UniformIntegerHyperparameter("num_leaves", 100, 5000, default_value=1000)
min_child_samples=ConfigSpace.hyperparameters.UniformIntegerHyperparameter("min_child_samples", 20, 200, default_value=100)
n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("n_estimators", 20, 200, default_value=100)
ConfigurationSpace.add_hyperparameters([boosting_type, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight, num_leaves, min_child_samples, n_estimators])
scenario = smac.scenario.scenario.Scenario({
    "run_obj": "quality", 
    "runcount-limit": 3, 
    "cs": ConfigurationSpace, 
    "deterministic": "true"
    })

error = evaluate(ConfigurationSpace.get_default_configuration())
print("default error: %s" % (error))

finder = smac.facade.smac_facade.SMAC(scenario=scenario, tae_runner=evaluate)
incumbent = finder.optimize()
inc_value = evaluate(incumbent)
print("Optimized error: %s" % (inc_value))