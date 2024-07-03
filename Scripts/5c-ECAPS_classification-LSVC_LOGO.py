"""
This code should fully optimize ONE MODEL (LinearSVC)
using ONE PIPELINE (RNV+MinMaxScaler)
"""

# os imports
import datetime
import os

# data handling
import pandas as pd
import numpy as np

# machine learning
import sklearn
import xarray as xr
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             balanced_accuracy_score)

from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import (KFold, StratifiedShuffleSplit,
                                     StratifiedKFold, LeavePGroupsOut)

# Trying optuna
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

#open the data
date = '20240628153159'
eval_class = 'Class2'
pipe = 'AllColors-AllStates_RNV_None2_None3_None4_MinMaxScaler_None6'
X = xr.open_dataset(f'NetCDFs/{date}_preprocessed_X_example.nc')[pipe]
Y = xr.open_dataset(f'NetCDFs/{date}_preprocessed_Y_example.nc')

#check for location to save data
newpath = r'HP_Param_Scores/Optuna/Example/IndvModel/LSVC/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# set up OHE
ohe_c1 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_c1.fit(Y.Class1.values.reshape(-1,1))
ohe_c2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_c2.fit(Y.Class2.values.reshape(-1,1))
ohe_dict={'Class1': ohe_c1, 'Class2':ohe_c2}

polymers = Y['polymer']
Y = Y[eval_class]

# create objects for outer CV loop
# split data for outer_CV, 10 fold split

o_cv = CrossValidatorWrapper(StratifiedShuffleSplit(n_splits=7, test_size=2/7),
                             dim='sample')
o_results = pd.DataFrame()
clsfr = 'LinearSVC'
# start outer loop
# this is our leave out group for final testing 
date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
i = 0
for train_ix, test_ix in o_cv.split(X, Y):
    X_train, X_test = X.isel(sample=train_ix), X.isel(sample=test_ix)
    Y_train, Y_test = Y.isel(sample=train_ix), Y.isel(sample=test_ix)
    P_train, P_test = polymers.isel(sample=train_ix), polymers.isel(sample=test_ix)

    #create the objective function for the LinearSVC
    def objective(trial):
        # generate hyper-parameter suggestions
        C = trial.suggest_float("C", 10, 150, log=False)
        class_weight = trial.suggest_categorical("class_weight",[None, 'balanced'])
        max_iter = trial.suggest_categorical("max_iter",[1000])
        multi_class = trial.suggest_categorical("multi_class",['ovr', 'crammer_singer'])
        penalty = trial.suggest_categorical("penalty",['l1'])
        dual = trial.suggest_categorical("dual", ["auto"])
        tol = trial.suggest_float("tol", 0.0001, 0.1, log=False)
        
        # create model
        model = svm.LinearSVC(C=C, class_weight=class_weight,
                              max_iter=max_iter, multi_class=multi_class,
                              penalty=penalty, dual=dual, tol=tol)
        #create Leave-one-polymer-out cross validator
        lopo = LeavePGroupsOut(n_groups=1)
        #score the model with the one set of HP-parameters
        cv_i_score = cross_val_score(model, X_train, Y_train, cv=lopo, groups=P_train)
        accuracy = cv_i_score.mean()
        return accuracy
    
    
    # create and run the optuna study
    # NOTE: n_trials is reduced to enable rapid testing of the code
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,
                   n_trials=25, n_jobs=32)
    # save the results in a csv incase we want it later
    study.trials_dataframe().to_csv(f'{newpath}/{date}_OStudy_{clsfr}_{i}.csv')
    # grab the best trial and get the inner cv score and hyper parameters out of it
    best_trial = study.best_trial
    i_cv_acc = best_trial.value
    pdict = best_trial.params
    model = svm.LinearSVC(**pdict)
    # train the model
    model.fit(X_train, Y_train[eval_class])
    # test the model
    Y_pred = model.predict(X_test)
    ol_acc = accuracy_score(Y_test[eval_class], Y_pred)
    ol_acc_bal = balanced_accuracy_score(Y_test[eval_class], Y_pred)
    ol_f1 = f1_score(Y_test[eval_class], Y_pred, average='weighted')
    ol_prec = precision_score(Y_test[eval_class], Y_pred, average='weighted')
    ol_rec = recall_score(Y_test[eval_class], Y_pred, average='weighted')
    # save our parameters and metrics
    out_dict = pdict.copy()
    out_dict.update({'Inner_acc':i_cv_acc,
                     'Outer_acc':ol_acc,
                     'Outer_acc_bal':ol_acc_bal,
                     'Outer_f1':ol_f1,
                     'Outer_prec':ol_prec,
                     'Outer_rec':ol_rec})
    df_out = pd.DataFrame(out_dict, index=[i])
    df_out.to_csv(f'{newpath}/{date}_OuterLoop_{clsfr}_{i}.csv')
    o_results = pd.concat([o_results, df_out])
    i+=1
# output a dataframe with all of our outer loop results    
o_results.to_csv(f'{newpath}/{date}_FullLoopSet.csv')

print('Finished!')