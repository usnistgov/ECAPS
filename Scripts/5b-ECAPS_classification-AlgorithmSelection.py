"""
This code trains multiple models on ONE PIPELINE,
NESTED CV.
This took about 20 hours to run on my 2021 Macbook Pro (M1 chip)
"""

# os imports
import datetime
import os

# misc function imports
import classifier_dictionaries as cd

# data handling
import pandas as pd
import numpy as np

# machine learning
import sklearn
import xarray as xr
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             balanced_accuracy_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

# make the optuna objective function
# Trying optuna
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

#open the data
with open('netCDF_date.txt', 'r') as file:
    date = file.read()
eval_class = 'Class2'
pipe = 'AllColors-AllStates_RNV_None2_None3_None4_MinMaxScaler_None6'
X = xr.open_dataset(f'NetCDFs/{date}_preprocessed_X_example.nc')[pipe]
Y = xr.open_dataset(f'NetCDFs/{date}_preprocessed_Y_example.nc')

#check for location to save data
newpath = r'HP_Param_Scores/Optuna/Example/IndvModel/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

ohe_c1 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_c1.fit(Y.Class1.values.reshape(-1,1))
ohe_c2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_c2.fit(Y.Class2.values.reshape(-1,1))
ohe_dict={'Class1': ohe_c1, 'Class2':ohe_c2}

Y = Y[eval_class]

# create objects for outer CV loop
# split data for outer_CV, 10 fold split
o_cv = CrossValidatorWrapper(StratifiedShuffleSplit(n_splits=10),
                             dim='sample')
o_results = pd.DataFrame()

# start outer loop
# this is our leave out group for final testing 
date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
i = 0
for train_ix, test_ix in o_cv.split(X, Y):
    X_train, X_test = X.isel(sample=train_ix), X.isel(sample=test_ix)
    Y_train, Y_test = Y.isel(sample=train_ix), Y.isel(sample=test_ix)
    #select the classifier we want to use:
    for clsfr in [#'SIMCA','PLS-DA',
                  'AdaBoost',
                  'LinearSVC','RBF_SVC','GaussianNB','KNN',
                  'LDA','QDA', 'RandomForest','MLPC'
    ]:
        # create and run the optuna study
        study = optuna.create_study(direction="maximize")
        # NOTE: n_trials is reduced to enable rapid testing of the code
        study.optimize(lambda trial: cd.objective(trial, clsfr,
                                                  X_train, Y_train,
                                                  eval_class),
                       n_trials=25, n_jobs=-1)
        # save the results in a csv incase we want it later
        study.trials_dataframe().to_csv(f'{newpath}/{date}_OStudy_{clsfr}_{i}.csv')
        # grab the best trial and get the inner cv score and hyper parameters out of it
        best_trial = study.best_trial
        i_cv_acc = best_trial.value
        pdict = best_trial.params
        if clsfr == 'AdaBoost':
            pdict['estimator'] =  DecisionTreeClassifier(max_depth=pdict['estimator'],
                                                         min_samples_leaf=1,
                                                         max_features='sqrt')
        model = cd.clsfr_dict[clsfr](**pdict)
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
        print(f'loop {i}, {clsfr} finished evaluating')
    i+=1
# output a dataframe with all of our outer loop results    
o_results.to_csv(f'{newpath}/{date}_FullLoopSet.csv')
    
print('Finished!')    
