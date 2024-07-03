
from tqdm import tqdm
import misc_funcs as misc
import pandas as pd
import xarray as xr
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import warnings

date = '20240703130237' # NOTE: This will need to be adjusted to when you generated your preprocessed data
ds_pp_X = xr.open_dataset(f'NetCDFs/{date}_preprocessed_X_example.nc')
ds_pp_y = xr.open_dataset(f'NetCDFs/{date}_preprocessed_Y_example.nc')
print('ds_pp* made')

ohe_c2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_c2.fit(ds_pp_y.Class2.values.reshape(-1,1))


ohe_dict={'Class2':ohe_c2}
eval_class = 'Class2'
print('ohe_dict made')

# here is a dictionary to init each classifier, you can comment some out
# if you are not interested in testing them. This will greadly speed up
# the process if you're short on time.
clsfr_dict = {'SIMCA':misc.SIMCA_classifier(cat_encoder=ohe_dict[eval_class],
                                           simca_type="SIMCA"),

              'PLS-DA3':PLSRegression(n_components=3),
              'PLS-DA5':PLSRegression(n_components=5),
              'RandomForest': RandomForestClassifier(),
              'MLPC':MLPClassifier(max_iter=2000),
              'LDA':LinearDiscriminantAnalysis(),
              'QDA':QuadraticDiscriminantAnalysis(),
              'LinearSVC':svm.LinearSVC(),
              'RBF_SVC':svm.SVC(),
              'GaussianNB':GaussianNB(),
              'KNN':KNeighborsClassifier(),
              'AdaBoost':AdaBoostClassifier()
             }
print('dict_made')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
for train, test in sss.split(X=ds_pp_X.sample, y=ds_pp_y.Class2,groups=ds_pp_y.Class2):
    ds_pp_X_train = ds_pp_X.isel(sample=train)
    ds_pp_y_train = ds_pp_y.isel(sample=train)
    ds_pp_X_test = ds_pp_X.isel(sample=test)
    ds_pp_y_test = ds_pp_y.isel(sample=test)

with warnings.catch_warnings(record=True) as cx_manager:
    # NOTE: you can speed up the following loop by downselecting the preprocessing steps via providing an explicit
    # list of preprocessing steps (ds_pp_X_train[['AllColors-AllStates_None1_None2_None3_None4_None5_None6',
    #                                             'AllColors-AllStates_RNV_None2_None3_None4_None5_None6']])
    # instead of using ds_pp_X_train.data_vars which will supply a list of all the preprocessing steps
    for pipe in tqdm(ds_pp_X_train.data_vars, position=0, leave=False):
        print(pipe)
        df_pipe_scores = pd.DataFrame()
        for clsfr in list(clsfr_dict.keys()):
            print(clsfr)
            df_bad_X_row = pd.DataFrame()
            # print(clsfr, pipe)
            df_scores = pd.DataFrame()
            # set X and y
            X_train = ds_pp_X_train[pipe].dropna(dim='feature', how='all').dropna(dim='sample', how='all')
            y_train = ds_pp_y_train[pipe].dropna(dim='sample', how='all')
            X_test = ds_pp_X_test[pipe].dropna(dim='feature', how='all').dropna(dim='sample', how='all')
            y_test = ds_pp_y_test[pipe].dropna(dim='sample', how='all')
              
            # train
            # not all classifiers are currently built to hand OneHotEncoded labels. This converts
            # the labels back to the polymer name.
            if clsfr in ['LDA','QDA','LinearSVC', 'RBF_SVC','GaussianNB','AdaBoost']:
                y_train = ohe_dict[eval_class].inverse_transform(y_train).squeeze()
            model = clsfr_dict[clsfr]
            model.fit(X_train, y_train)
            print('model trained')
            # test
            y_pred = model.predict(X_test)
            if clsfr in ['LDA','QDA','LinearSVC', 'RBF_SVC','GaussianNB','AdaBoost']:
                y_pred = ohe_dict[eval_class].transform(y_pred.reshape(-1,1))

            y_pred = (y_pred > 0.5).astype('uint8')
            print('model prediction made')
            # score
            acc = accuracy_score(y_test, y_pred)
            prec_micro = precision_score(y_test, y_pred, average='micro', zero_division=0.0)
            rec_micro = recall_score(y_test, y_pred, average='micro')
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0.0)
            prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0.0)
            rec_macro = recall_score(y_test, y_pred, average='macro')
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0.0)
            prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0.0)
            rec_weighted = recall_score(y_test, y_pred, average='weighted')
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0.0)
            print('all model scores evaluated')
            
            # add to dataframe
            df_scores['Data'] = [pipe.split('_')[0]]
            df_scores['Preprocessing'] = '_'.join(pipe.split('_')[1:])
            df_scores['Classifier'] = clsfr
            df_scores['Accuracy'] = acc
            df_scores['Precision_micro'] = prec_micro
            df_scores['Recall_micro'] = rec_micro
            df_scores['F1_micro'] = f1_micro
            df_scores['Precision_macro'] = prec_macro
            df_scores['Recall_macro'] = rec_macro
            df_scores['F1_macro'] = f1_macro
            df_scores['Precision_weighted'] = prec_weighted
            df_scores['Recall_weighted'] = rec_weighted
            df_scores['F1_weighted'] = f1_weighted
            df_scores['Warning'] = str([i.message for i in cx_manager])  # HERE
            cx_manager.clear()
            df_pipe_scores = pd.concat([df_pipe_scores, df_scores], ignore_index=True)
            print('--------'*10)
        
        #check for location to save data
        import os
        newpath = r'ClassifierScores/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        df_pipe_scores.to_csv(f'{newpath}/{pipe}.csv')

print("FINISHED!")