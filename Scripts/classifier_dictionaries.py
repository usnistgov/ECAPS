from scipy.stats import uniform, loguniform, randint
from sklearn.tree import DecisionTreeClassifier
import xarray as xr
import misc_funcs as bps
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             balanced_accuracy_score)
from sklearn.preprocessing import OneHotEncoder

clsfr_dict = {'SIMCA':bps.SIMCA_classifier,
              'PLS-DA':PLSRegression,
              'RandomForest': RandomForestClassifier,
              'MLPC':MLPClassifier,
              'LDA':LinearDiscriminantAnalysis,
              'QDA':QuadraticDiscriminantAnalysis,
              'LinearSVC':svm.LinearSVC,
              'RBF_SVC':svm.SVC,
              'GaussianNB':GaussianNB,
              'KNN':KNeighborsClassifier,
              'AdaBoost':AdaBoostClassifier}

def objective(trial, clsfr, X_train, Y_train, eval_class):
    """
    takes in trial, the classifier to be tested,
    then X and Y training data, and finally the
    class level used to select Y
    """

    eval_class = eval_class
    ohe_c1 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_c1.fit(Y_train.Class1.values.reshape(-1,1))
    ohe_c2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_c2.fit(Y_train.Class2.values.reshape(-1,1))
    ohe_dict={'Class1': ohe_c1, 'Class2':ohe_c2}
    Y_train = Y_train[eval_class]

    #n_comps = len(pd.unique(y_train))
    # clsfr_dict = {'SIMCA':bps.SIMCA_classifier,
    #               'PLS-DA':PLSRegression,
    #               'RandomForest': RandomForestClassifier,
    #               'MLPC':MLPClassifier,
    #               'LDA':LinearDiscriminantAnalysis,
    #               'QDA':QuadraticDiscriminantAnalysis,
    #               'LinearSVC':svm.LinearSVC,
    #               'RBF_SVC':svm.SVC,
    #               'GaussianNB':GaussianNB,
    #               'KNN':KNeighborsClassifier,
    #               'AdaBoost':AdaBoostClassifier}

    abt_list = [DecisionTreeClassifier(max_depth=1,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=2,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=3,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=4,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=5,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=6,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=7,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=8,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=9,
                                       min_samples_leaf=1,
                                       max_features='sqrt'),
                DecisionTreeClassifier(max_depth=10,
                                       min_samples_leaf=1,
                                       max_features='sqrt')]

            
    hp_dict = {'SIMCA':{'n_components':['s_int', 1, 10, False],
                                'alpha': ['s_fl', 0.01, 0.1, False],
                                'cat_encoder':['s_cat', [ohe_dict[eval_class]]],
                                'simca_type':['s_cat', ["SIMCA"]]},
                       'PLS-DA':{'n_components':['s_int', 1, 10, False],
                                 'scale':['s_cat', [False]],
                                 'max_iter':["s_int", 500, 1000, True],
                                 'tol':['s_fl', 0.000001, 0.1, False]},
                       'RandomForest':{'n_estimators':["s_int", 100, 1000, False],
                                       'criterion': ["s_cat", ['gini', 'entropy']],
                                       'max_features':["s_cat",['log2', 'sqrt']],
                                       'max_depth':["s_int", 1, 50, True],
                                       'min_samples_leaf':["s_int", 1, 5, False],
                                       'bootstrap':["s_cat", [False]]},    
                       'MLPC':{'hidden_layer_sizes':["s_int", 1, 1000, True],
                               'solver':["s_cat", ['lbfgs', 'sgd', 'adam']],
                               'alpha':["s_fl", 0.000001, 0.1, False],
                               'learning_rate': ["s_cat", ['constant','invscaling','adaptive']],
                               'max_iter': ["s_cat", [500]],
                               'tol':["s_fl", 0.000001, 0.1, False],
                               'early_stopping':["s_cat", [True]]},
                       'LDA': {'n_components':["s_int", 1, 5, False], #limited by number of classes
                               'shrinkage': ["s_cat", [None]],
                               'solver': ["s_cat", ['svd', 'lsqr']],
                               'tol': ["s_fl", 0.000001, 0.1, False]},
                       'QDA':{'tol': ["s_fl", 0.000001, 0.1, False],
                             'reg_param': ["s_fl", 0, 0.99, False]},
                       'LinearSVC':{'C': ["s_fl", 0.001, 100, False], 
                                    'class_weight': ["s_cat", [None, 'balanced']],
                                    'max_iter': ["s_cat", [500]],
                                    'multi_class': ["s_cat", ['ovr', 'crammer_singer']],
                                    'penalty': ["s_cat", ['l1','l2']],
                                    'dual': ["s_cat", ["auto"]],
                                    'tol': ["s_fl", 0.000001, 0.1, False]},
                       'RBF_SVC':{'C': ["s_fl", 0.001, 100, False],
                                  'kernel':["s_cat", ['rbf']],
                                  'gamma': ["s_cat", ['scale', 'auto']],
                                  'tol':["s_fl", 0.000001, 0.1, False],
                                  'class_weight':["s_cat", [None, 'balanced']],
                                  'max_iter': ["s_cat", [500]]},
                       'GaussianNB':{'var_smoothing':["s_fl", 10**(-10), 10**(-6), False]},
                       'KNN':{'weights':["s_cat", ['uniform', 'distance']],
                              'algorithm': ["s_cat", ['auto']],
                              'leaf_size': ["s_int", 1, 50, False],
                              'n_neighbors': ["s_int", 1, 50, False],
                              'p': ["s_cat", [1,2]]},
                       'AdaBoost':{'n_estimators':["s_int", 100, 1000, False],
                                   'learning_rate': ["s_fl", 0.001, 1, True],
                                   'algorithm': ["s_cat", ['SAMME', 'SAMME.R']],
                                   # 'estimator': ["s_cat", [abt_list]]
                                   'estimator': ['s_est', 1, 10, False]}
                    }
    def make_sug(clsfr, hyper):
        sug_type = hp_dict[clsfr][hyper][0]
        if sug_type == 's_int':
            start = hp_dict[clsfr][hyper][1]
            stop = hp_dict[clsfr][hyper][2]
            log_sug = hp_dict[clsfr][hyper][3]
            sug = trial.suggest_int(hyper, start, stop, log=log_sug)
        elif sug_type == 's_fl':
            start = hp_dict[clsfr][hyper][1]
            stop = hp_dict[clsfr][hyper][2]
            log_sug = hp_dict[clsfr][hyper][3]
            sug = trial.suggest_float(hyper, start, stop, log=log_sug)  
        elif sug_type == 's_cat':
            sug_list = hp_dict[clsfr][hyper][1]
            sug = trial.suggest_categorical(hyper, sug_list)
        elif sug_type == 's_est':
            start = hp_dict[clsfr][hyper][1]
            stop = hp_dict[clsfr][hyper][2]
            log_sug = hp_dict[clsfr][hyper][3]
            md = trial.suggest_int(hyper, start, stop, log=log_sug)
            sug = DecisionTreeClassifier(max_depth=md,
                                         min_samples_leaf=1,
                                         max_features='sqrt')
        else:
            print(f'What kind of suggestion do you need for {clsfr}-{hyper}?')
        return sug
    
    #grab/create hyper parameter variables 
    params =dict()
    for hyper in hp_dict[clsfr].keys():
        params[hyper] = make_sug(clsfr, hyper)
    # create model using our optuna variables
    model = clsfr_dict[clsfr](**params)
    #score the model with the one set of HP-parameters
    cv_i_score = cross_val_score(model, X_train, Y_train, cv=5)
    accuracy = cv_i_score.mean()
    return accuracy