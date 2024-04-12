# -*- coding: utf-8 -*-
#!pip install lazypredict
#!pip install tqdm
#!pip install xgboost
#!pip install catboost
#!pip install lightgbm
#!pip install pytest

# Load libraries
#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import time
from joblib import Parallel, delayed

import random
import os
import json


import warnings
warnings.filterwarnings('ignore')

SEED = 65

# SET RANDOM SEED FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import pandas as pd
print("### DATASET: ", 'DIS_lab_LoS_8.csv')
dataset = pd.read_csv('data_files/ultra_dense/DIS_lab_LoS_8.csv')

#df_var = dataset.iloc[:,0:-2]
columns_to_normalize = dataset.columns[:-2]
df_var = (dataset[columns_to_normalize] - dataset[columns_to_normalize].min()) / (dataset[columns_to_normalize].max() - dataset[columns_to_normalize].min())
df_X = dataset.iloc[:,-2]
df_Y = dataset.iloc[:,-1]
np.random.seed(SEED)
df_var = df_var.sample(frac=1).reset_index(drop=True)
np.random.seed(SEED)
df_X = df_X.sample(frac=1).reset_index(drop=True)
np.random.seed(SEED)
df_Y = df_Y.sample(frac=1).reset_index(drop=True)
# Training size
trainings_size = 0.85                     # 85% training set
validation_size = 0.1                     # 10% validation set
test_size = 0.05                         # 5% test set
# Para posicion X
X_train = df_var.iloc[:int(trainings_size*len(df_var))]
y_train_x = df_X.iloc[:int(trainings_size*len(df_X))]
y_train_y = df_Y.iloc[:int(trainings_size*len(df_Y))]
X_val = df_var.iloc[int(trainings_size*len(df_var)):int((trainings_size+validation_size)*len(df_var))]
y_val_x = df_X.iloc[int(trainings_size*len(df_X)):int((trainings_size+validation_size)*len(df_X))]
y_val_y = df_Y.iloc[int(trainings_size*len(df_Y)):int((trainings_size+validation_size)*len(df_Y))]
X_test = df_var.iloc[-int(test_size*len(df_var)):]
y_test_x = df_X.iloc[-int(test_size*len(df_X)):]
y_test_y = df_Y.iloc[-int(test_size*len(df_Y)):]
validation_x = y_val_x
test_x = y_test_x
validation_y = y_val_y
test_y = y_test_y
# Spot Check Algorithms
models = []
models.append(('Bagging', BaggingRegressor(random_state=SEED, n_estimators = 200, n_jobs=-1)))
models.append(('ET', ExtraTreesRegressor(random_state=SEED, n_estimators = 200, n_jobs=-1)))
models.append(('HGBR', HistGradientBoostingRegressor(random_state=SEED, max_iter = 500)))
models.append(('k-NN', KNeighborsRegressor(n_neighbors=11, n_jobs=-1)))
models.append(('LiR', LinearRegression()))
models.append(('MLPRegressor', MLPRegressor(random_state=SEED, max_iter=500, early_stopping=True)))
models.append(('RF', RandomForestRegressor(random_state=SEED, n_estimators = 200, n_jobs=-1)))
models.append(('RidgeCV', RidgeCV()))
models.append(('XGBR', xgb.XGBRegressor(objective="reg:linear", n_estimators=200, random_state=SEED)))
models.append(('LGBMRegressor', lgb.LGBMRegressor(random_state=SEED, n_estimators=200)))
n_jobs = len(models)
def true_dist(y_pred, y_true):
    return np.mean(np.sqrt(
        np.square(np.abs(y_pred[:,0] - y_true[:,0]))
        + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
        ))
def obtain_results_x(name, model):
    #kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    unique_name = f"{name}_{model.n_neighbors}"
    print(f"model {name}")
    results = {}
    results_dict = {}
    t0 = time.time()
    model.fit(X_train, y_train_x)
    print("----------------------------", model, "---------------------")
    predX_val = model.predict(X_val)
    results['Validation R2 X'] = r2_score(y_val_x, predX_val)
    results['Validation RMSE X'] = mean_squared_error(y_val_x, predX_val, squared=False)
    results['Validation MAE X'] = mean_absolute_error(y_val_x, predX_val)
    results['Validation MSE X'] = mean_squared_error(y_val_x, predX_val)
    results['Validation Time X'] = time.time() - t0
    
    t0 = time.time()
    predX_test = model.predict(X_test)
    results['Test R2 X'] = r2_score(y_test_x, predX_test)
    results['Test RMSE X'] = mean_squared_error(y_test_x, predX_test, squared=False)
    results['Test MAE X'] = mean_absolute_error(y_test_x, predX_test)
    results['Test MSE X'] = mean_squared_error(y_test_x, predX_test)
    results['Test Time X'] = time.time() - t0
    t0 = time.time()
    model.fit(X_train, y_train_y)
    print("----------------------------", model, "---------------------")
    predY_val = model.predict(X_val)
    results['Validation R2 Y'] = r2_score(y_val_y, predY_val)
    results['Validation RMSE Y'] = mean_squared_error(y_val_y, predY_val, squared=False)
    results['Validation MAE Y'] = mean_absolute_error(y_val_y, predY_val)
    results['Validation MSE Y'] = mean_squared_error(y_val_y, predY_val)
    results['Validation Time Y'] = time.time() - t0
    
    t0 = time.time()
    predY_test = model.predict(X_test)
    results['Test R2 Y'] = r2_score(y_test_y, predY_test)
    results['Test RMSE Y'] = mean_squared_error(y_test_y, predY_test, squared=False)
    results['Test MAE Y'] = mean_absolute_error(y_test_y, predY_test)
    results['Test MSE Y'] = mean_squared_error(y_test_y, predY_test)
    results['Test Time Y'] = time.time() - t0
    
    predictions_valid = pd.DataFrame()
    predictions_valid["realX"] = validation_x
    predictions_valid["realY"] = validation_y
    
    predictions_valid["predX"] = predX_val
    predictions_valid["predY"] = predY_val
    
    error_valid = true_dist(predictions_valid[["predX", "predY"]].to_numpy(), predictions_valid[["realX", "realY"]].to_numpy())

    results['Validation Error'] = error_valid
    
    predictions_test = pd.DataFrame()
    predictions_test["realX"] = test_x
    predictions_test["realY"] = test_y
    
    predictions_test["predX"] = predX_test
    predictions_test["predY"] = predY_test
    
    error_test = true_dist(predictions_test[["predX", "predY"]].to_numpy(), predictions_test[["realX", "realY"]].to_numpy())
    results['Test Error'] = error_test
    
    results['n_neigbors'] = model.n_neighbors

    return results
#results  = {name: obtain_results(name, model) for name, model in models}
results_list = {f"{name}_{model.n_neighbors}": obtain_results_x(name, model) for name, model in models}

# Save results as json
with open('LazyRegressor.json', 'w') as f:
    json.dump(results_list, f)