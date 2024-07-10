"""
This file is for using Optuna to tune a bunch of random forest classifiers to
identify the best preprocessing parameters FROM AN ALREADY REDUCED SET of
prepocessing parameters. THIS USES NESTED CV
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   OneHotEncoder)
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# make the optuna objective function
# Trying optuna
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

#open the data
with open('netCDF_date.txt', 'r') as file:
    date = file.read()
X = xr.open_dataset(f'NetCDFs/{date}_preprocessed_X_example.nc')
Y = xr.open_dataset(f'NetCDFs/{date}_preprocessed_Y_example.nc')
eval_class = 'Class2'

#check for location to save data
newpath = r'HP_Param_Scores/Optuna/Example/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# create objects for outer CV loop
# split data for outer_CV, 10 fold split
# o_cv = CrossValidatorWrapper(StratifiedKFold(n_splits=7, shuffle=True),
#                              dim='sample')
o_cv = CrossValidatorWrapper(StratifiedShuffleSplit(n_splits=10),
                             dim='sample')
o_results = pd.DataFrame()

# start outer loop
date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
i = 0
for train_ix, test_ix in o_cv.split(X, Y[eval_class]):
    X_train, X_test = X.isel(sample=train_ix), X.isel(sample=test_ix)
    Y_train, Y_test = Y.isel(sample=train_ix)[eval_class], Y.isel(sample=test_ix)[eval_class]
    # define optuna objective function to get ideal params
    # this should include both preprocessing steps and
    # actual hyper parameters
    def objective(trial): 
        # use optuna to set preprocessing steps
        # pp1 = trial.suggest_categorical("pp1", ['None1', 'MeanCentering', 'SNV', 'RNV'])
        pp1 = trial.suggest_categorical("pp1", ['RNV'])
        # pp2 = trial.suggest_categorical("pp2", ['None2','Detrending'])
        pp2 = trial.suggest_categorical("pp2", ['None2'])
        pp3 = trial.suggest_categorical("pp3", ['None3'])
        pp4 = trial.suggest_categorical("pp4", ['None4'])
        # pp5 = trial.suggest_categorical("pp5", ['None5', 'MinMaxScaler', 'StandardScaler']) 
        pp5 = trial.suggest_categorical("pp5", ['None5', 'MinMaxScaler'])

        # and data reduction steps
        dr = trial.suggest_categorical("dr", ['None6'])
        # now select the data for training/HP-tuning
        il_pipe = f'AllColors-AllStates_{pp1}_{pp2}_{pp3}_{pp4}_{pp5}_{dr}'
        # select data based on pipe
        X_cv = X_train[il_pipe].dropna(dim='feature').values
        Y_cv = Y_train.values
        #setup hyper parameter variables 
        criterion = trial.suggest_categorical("criterion", ['gini', 'entropy'])
        max_depth = trial.suggest_int("max_depth", 1, 50, log=True)
        n_estimators = trial.suggest_int("n_estimators",100,1000,log=False)
        max_features = trial.suggest_categorical("max_features",['log2', 'sqrt'])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=False)
        bootstrap = trial.suggest_categorical("bootstrap", [False])
        # create model using our optuna variables
        rf = RandomForestClassifier(criterion=criterion, max_depth=max_depth,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    n_estimators=n_estimators, bootstrap=bootstrap)
        #score the model with the one set of HP-parameters
        cv_i_score = cross_val_score(rf, X_cv, Y_cv, cv=5)
        accuracy = cv_i_score.mean()
        return accuracy

    # create and run the optuna study
    # NOTE: n_trials is reduced to enable rapid testing of the code
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=32)
    
    # save the results in a csv incase we want it later
    study.trials_dataframe().to_csv(f'{newpath}/{date}_OStudy_{i}.csv')
    
    # grab the best trial and get the inner cv score and hyper parameters out of it
    best_trial = study.best_trial
    i_cv_acc = best_trial.value
    pdict = best_trial.params
    # make a copy of the HP dictionary so we can pop out the preprocessing
    # steps and be left with only the classifier parameters
    popd = pdict.copy()
    ol_pipe=(f"AllColors-AllStates_"
             + f"{popd.pop('pp1')}_{popd.pop('pp2')}_{popd.pop('pp3')}_"
             + f"{popd.pop('pp4')}_{popd.pop('pp5')}_{popd.pop('dr')}")
    # make our model with the remaining params
    model = RandomForestClassifier(**popd)
    # select our data    
    sub_X_train = X_train[ol_pipe]
    sub_Y_train = Y_train # we don't REALLY need this for Y, but for consistency we'll have it.
    sub_X_test = X_test[ol_pipe]
    sub_Y_test = Y_test
    # train the model
    model.fit(sub_X_train, sub_Y_train)
    # test the model
    Y_pred = model.predict(sub_X_test)
    ol_acc = accuracy_score(sub_Y_test, Y_pred)
    ol_acc_bal = balanced_accuracy_score(sub_Y_test, Y_pred)
    ol_f1 = f1_score(sub_Y_test, Y_pred, average='weighted')
    ol_prec = precision_score(sub_Y_test, Y_pred, average='weighted')
    ol_rec = recall_score(sub_Y_test, Y_pred, average='weighted')
    # save our parameters and metrics
    out_dict = pdict.copy()
    out_dict.update({f'Inner_acc_{i}':i_cv_acc,
                     f'Outer_acc_{i}':ol_acc,
                     f'Outer_acc_ball_{i}':ol_acc_bal,
                     f'Outer_f1_{i}':ol_f1,
                     f'Outer_prec_{i}':ol_prec,
                     f'Outer_rec_{i}':ol_rec})
    df_out = pd.DataFrame(out_dict, index=[i])
    df_out.to_csv(f'{newpath}/{date}_OuterLoop_{i}.csv')
    o_results = pd.concat([o_results, df_out])
    i+=1
# output a dataframe with all of our outer loop results    
o_results.to_csv(f'{newpath}/{date}_FullLoopSet.csv')
    
print('Finished!')