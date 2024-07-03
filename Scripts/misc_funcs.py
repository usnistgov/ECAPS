# imports that this needs to function
from scipy.signal import savgol_filter, detrend
import pandas as pd
import skfda
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import matplotlib.markers as pltmarkers
from tqdm import tqdm
from sklearn_xarray import Target, wrap
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score,
                            precision_recall_fscore_support)
try:
    from simca import SIMCA, DDSIMCA
except:
    from Scripts.simca import SIMCA, DDSIMCA

# make a bunch of dictionaries
dict_colors = {
    "Class1": {"PP": "#cb1b4f", "PE": "#357ba3"},
    "Class2": {
        "PE": "#357ba3",
        "PP": "#e13342",
        "LDPE": "#40b7ad",
        "LLDPE": "#348fa7",
        "MDPE": "#37659e",
        "HDPE": "#413d7b",
        "PP-co-PE": "#ad1759",
    },
}

dict_cBlind = {
    "Class1": {"PP": "#0173b2", "PE": "#d55e00"},
    "Class2": {
        "PE": "#949494",
        "PP": "#0173b2",
        "LDPE": "#ece133",
        "LLDPE": "#ca9161",
        "MDPE": "#de8f05",
        "HDPE": "#d55e00",
        "PP-co-PE": "#56b4e9",
    },
    "Color":{'Natural': '#cc78bc',
             'Gray':'#949494',
             'White':'#029e73',
             'Black':'#000000'},
    "state":{'Pellet':'#cc78bc',
             'Powder':'#949494'},
    "User":{'S.G.':'#cc78bc',
             'A.R.Y.':'#949494'},
    "repeat":{0:'#0173b2', 
              1:'#de8f05',
              2:'#029e73',
              3:'#d55e00',
              4:'#cc78bc',
              5:'#ca9161',
              6:'#fbafe4',
              7:'#949494',
              8:'#ece133',
              9:'#56b4e9',
             },
    "fPCA_loading": {2: "#029e73", 3: "#cc78bc", 1: "#949494", 0: "#fbafe4"},
    "classifiers":{'SIMCA':'#000000', 
                   'PLS-DA':'#009292',
                   'PLS-DA3':'#009292',
                   'PLS-DA5':'#009292',
                   'RandomForest':'#ff6db6',
                   'MLPC':'#ffb6db',
                   'LDA':'#490092',
                   'QDA':'#006ddb',
                   'LinearSVC':'#b66dff',
                   'RBF_SVC':'#6db6ff',
                   'GaussianNB':'#b6dbff',
                   'KNN':'#920000',
                   'AdaBoost':'#924900',
                   'max':'#838383',
                   'min':'#838383'},
    "Preprocessing":{'None1':'#000000', 
                   'RNV':'#009292',
                   'SNV':'#ff6db6',
                   'MeanCentering':'#ffb6db'},
    "DimReduction":{'None6':'#490092',
                   'PCA':'#006ddb',
                   'fPCA':'#b66dff',
                   'UMAP':'#6db6ff'},
    "Smoothing": {'None3':'#000000',
                  'SG':'#006ddb',
                  'SG1':'#ff6db6',
                  'SG2':"#ffb6db"},
    "Normalization": {'None4':'#000000',
                      'L1':'#006ddb',
                      'L2':'#ff6db6'},
    "FeatureScaling": {'None5':'#000000',
                      'MinMaxScaler':'#006ddb',
                      'StandardScaler':'#ff6db6'},
    "Detrending":{'None2':'#ffb6db',
                  'Detrending':'#b6dbff'}

}


dict_zord = {"PP": 1, "LDPE": 2, "LLDPE": 3, "PE":0,
             "MDPE": 6, "HDPE": 4, "PP-co-PE": 5}
dict_lines = {
    "Class1": {"PP": (0,(4,2)), "PE": (0,(6,2))},
    "Class2": {
        "PE": (0,(6,2)),
        "PP": (0,(4,2)),
        "iPP": (0,(4,2,2,2)),
        "rPP": (0,(4,2,4,4)),
        "LDPE": (0,(5,2,4,4)),
        "LLDPE": (0,(5,2,4,2)),
        "MDPE": (0,(4,2,4,4)),
        "HDPE": (0,(6,2,6,2)),
        "PP-co-PE": (0,(4,2,6,2))},
    "State":{"Pellet":1,
             "Powder":0.25},
    "User":{"S.G.":1,
            "A.R.Y.":0.25}
        }
dict_shapes = {
    "Class1": {"PP": "o", "PE": "v"},
    "Class2": {
        "PE": "v",
        "PP": "o",
        "iPP": "h",
        "rPP": "H",
        "LDPE": "<",
        "LLDPE": "^",
        "MDPE": "v",
        "HDPE": ">",
        "PP-co-PE": "H"},
    "Color":{'Natural': 'o',
             'Gray':'v',
             'White':'>',
             'Black':'H'},
    "state":{'Pellet':'o',
             'Powder':'^'},
    "User":{'S.G.':'o',
             'A.R.Y.':'^'},
    "repeat":{0:(3,0,90), 
              1:(3,0,120),
              2:(10,1,0),
              3:(8,0,0),
              4:(4,0,0),
              5:(4,0,45),
              6:(5,0,0),
              7:(5,0,120),
              8:(6,1,0),
              9:(3,1,0),
             },
    "Data": {'Natural-Pellet':"o",
             'Natural-AllStates':"^",
             'NoBlack-Pellet':"p",
             'NoBlack-AllStates':"<",
             'AllColors-Pellet':"H",
             'AllColors-AllStates':">"},
    "classifiers":{'SIMCA':(3,0,90), 
               'PLS-DA':(3,1,180),
               'PLS-DA3':(3,1,45),
               'PLS-DA5':(3,1,60),
               'RandomForest':(10,1,0),
               'MLPC':(8,0,0),
               'LDA':(4,0,0),
               'QDA':(4,0,45),
               'LinearSVC':(5,0,0),
               'RBF_SVC':(5,0,180),
               'GaussianNB':(6,1,0),
               'KNN':(3,1,0),
               'AdaBoost':'*',
               'max':'o',
               'min':'o'},
    "Smoothing": {'None3':"o",
                  'SG':"^",
                  'SG1':">",
                  'SG2':"<"},
    "Preprocessing":{'None1':'o', 
                   'RNV':'^',
                   'SNV':'>',
                   'MeanCentering':'<'},
    "DimReduction":{'None6':'o',
                   'PCA':'^',
                   'fPCA':'>',
                   'UMAP':'<'},
    "Normalization": {'None4':'o',
                      'L1':'^',
                      'L2':'>'},
    "FeatureScaling": {'None5':'o',
                      'MinMaxScaler':'^',
                      'StandardScaler':'>'},
    "Detrending":{'None2':'o',
                  'Detrending':'^'}
}

axis_dict = {
    "Density": "Density [g/cm$^3$]",
    "Crystallinity": "Crystallinity [%]",
    "Mw(IR) (g_per_mol)": "Mw [g/mol]",
    "PDI(IR)": "PDI(IR)",
    "Mw(MALS) g_per_mol": "Mw [g/mol]",
    "PDI(MALS)": "PDI",
    "Mw(Viscometer) g_per_mol": "Mw(Viscometer) [g/mol]",
    "PDI(Visco)": "PDI",
    "MHa": "MHa",
    "MHK dL_per_g": "MHK [dL/g]",
    "SCB_per_1000C (IR)": "$CH_3/1000 C$",
    "LCBf_per_1000C (MALS)": "Long Chain Branching/1000 C",
    "pc0": "First fPC",
    "pc1": "Second fPC",
    "pc2": "Third fPC",
    "pc3": "Fourth fPC",
}

axes_dict = {'Intensity':0,
             'MSC':4, 'MSC_dt':5, 'MSC_dt_sg':6, 'MSC_dt_sg2':7,
             'SNV':8, 'SNV_dt':9, 'SNV_dt_sg':10, 'SNV_dt_sg2':11,
             'RNV':12, 'RNV_dt':13, 'RNV_dt_sg':14, 'RNV_dt_sg2':15} 

axes_mosaic = [['Intensity','Intensity','MSC', 'MSC_dt'],
               ['Intensity','Intensity','MSC_dt_sg', 'MSC_dt_sg2'],
               ['SNV', 'SNV_dt', 'SNV_dt_sg', 'SNV_dt_sg2'],
               ['RNV', 'RNV_dt', 'RNV_dt_sg', 'RNV_dt_sg2']]
               
               
               
# now the functions
def make_linestyle(polymer, state, class_lines=dict_lines['Class2'], state_mod=dict_lines['State']):
    poly_ls = class_lines[polymer]
    state_multi = state_mod[state]
    modded_poly_ls_inner = []
    for i in poly_ls[1]:
        modded_poly_ls_inner.append(i*state_multi)
    return tuple((0, tuple(modded_poly_ls_inner)))

def fix_markers(scatter_plot, marker_vector):
    """
    A function to apply new markers to a scatter_plot
    based on a vector of markers. marker_vector must have
    the same dimension as the x and y for scatter_plot
    """
    marker_locs=[]
    for marker in marker_vector:
        if marker in [i for i in Line2D.markers]:
            obj_marker = pltmarkers.MarkerStyle(marker)
        else:
            try:
                obj_marker = pltmarkers.MarkerStyle(marker)
            except UnboundLocalError:
                print(f'{marker} is not a viable marker')
        marker_loc = obj_marker.get_path(
        ).transformed(obj_marker.get_transform())
        marker_locs.append(marker_loc)
    scatter_plot.set_paths(marker_locs)
    return scatter_plot

def fix_markers2(scatter_plot, marker_vector):
    """
    A function to apply new markers to a scatter_plot
    based on a vector of markers. marker_vector must have
    the same dimension as the x and y for scatter_plot
    """
    marker_locs=[]
    for marker in marker_vector:
        try:
            obj_marker = pltmarkers.MarkerStyle(marker)
        except:
            print(f'{marker} is not a viable marker')
        marker_loc = obj_marker.get_path(
        ).transformed(obj_marker.get_transform())
        marker_locs.append(marker_loc)
    scatter_plot.set_paths(marker_locs)
    return scatter_plot
    
# define functions to then make function transformers:
# each of this will read from an Xarray DataArray
# and return another DataArray
def data_select(ds, colors='AllColors', state='AllStates'):
    """
    ds: datatset
    colors: dimension/indexed coordinate to select from
    state: dimension/indexed coordinate to select from
    
    ________
    returns X, y
    X = features
    y = OneHotEncoded labels
    
    """
    clr_select_dict = {'Natural':ds.Color=='Natural',
                   'NoBlack':ds.Color!='Black',
                   'AllColors':ds.Color.notnull()}
    state_select_dict = {'Pellet':ds.state=='Pellet',
                         'AllStates':ds.state.notnull()}
    ds_selColor = ds.where(clr_select_dict[colors], drop=True)
    ds_selState = ds_selColor.where(state_select_dict[state], drop=True)
    
    X = ds_selState.Intensity.squeeze()
    y = ds_selState.Class2_ohe.squeeze()
    
    return X, y

def mean_center(data):
    "subtract the mean from the data"
    data2 = data.copy()
    X = data2.values
    mean = data2.mean().values
    data2.values = X-mean
    return data2

def xr_MC(data):
    """
    applies the mean_center function to an xarray dataset
    """
    return data.groupby('sample').map(mean_center)  

def SNV(data):
    """
    Standard normal variate normalizes the data to a mean of zero
    and a stdev of 1 by using the mean and stdev of the entire dataset.
    
    This function takes in an xarray dataARRAY and will apply this function
    to the dataarray.
    
    data = an xarray dataarray
    """
    data2 = data.copy()
    x0 = data2.values
    a0 = data2.mean().values
    #print(f'a0: {a0}')
    a1 = data2.std().values
    #print(f'a1: {a1}')
    data2.values = (x0-a0)/a1
    return data2

def xr_SNV(data):
    """
    applies the standard normal variate (SNV) function to an xarray dataset
    """
    return data.groupby('sample').map(SNV)

def RNV(data):
    """
    Robust normal variate normalizes the data to a mean of ~zero
    and a stdev of ~1 by using the mean and stdev of the interquartile
    range.
    
    This function takes in an xarray dataset and will apply this function
    to the dataarray neamed by var.
    
    data = an xarray dataset
    var = the name of the dataarray in the dataset that you want to normalize.
    """
    # Computing IQR
    q1 = data.quantile(0.25).values
    q3 = data.quantile(0.75).values
    # Filtering Values between Q1 and Q3
    iqr = data.where(((data > q1) & (data < q3)),
                         drop=True)
    xo = data.values
    a0 = iqr.mean().values
    a1 = iqr.std().values
    data.values = (xo-a0)/a1
    return data

def xr_RNV(data):
    """
    applies the robust normal variate (RNV) function to an xarray dataset
    """
    return data.groupby('sample').map(RNV)

def xr_detrend(data):
    """
    applies scipy.signal.detrend to remove baseline increase across the x-axis
    that is to say systematic rise in y with increasing x
    """
    return data.groupby('sample').map(detrend)


def xr_SG(data, window_length=249, delta=1.9280342251144855, polyorder=2, deriv=0):
    """
    applies the Savitzky-Golay filter for an Xarray Dataset
    the delta is set to the average difference between
    consecutive wavenumbers for this Dataset
    """
    return data.groupby("sample").map(
        savgol_filter, window_length=window_length, delta=delta,
        polyorder=polyorder, deriv=deriv
    )


def xr_SG2(data, window_length=249, delta=1.9280342251144855, polyorder=2, deriv=2):
    """
    applies the Savitzky-Golay filter and takes the second derivative for an Xarray Dataset
    """
    return data.groupby("sample").map(
        savgol_filter, window_length=window_length, delta=delta,
        polyorder=polyorder, deriv=deriv
    )


def id_func(data):
    """
    returns the dame data with no changes
    this is mostly for use as a pipeline placeholder
    for when no function is applied (no-dt, no-filter)
    """
    return data


# def my_PCA(data, n_components=0.99, min_n_components=3,
#            verbose=False, in_pipe=False):
#     """
#     This takes an Xarray DataArray and converts it to an FDataGrid for fPCA,
#     runs fPCA, and then returns a DataArray with fPCA loadings
#     and the transformed data.
    
#     Additionally, if n_components is between 0 and 1, my_fPCA will return
#     enough principal components to explain `n_components` worth of variance 
    
#     PCA_loadings: the 'functions' that are used to transform the data. 
#                   These will have dimensions of (Wavenumber, n_components, 1)
#     PCs: the values for the transformed data
#           These will have the dimensions of (samples, n_components, 1)
#     """
#     data2 = data.copy()
#     pca=wrap(PCA(n_components=n_components), sample_dim='sample',
#                            reshapes='Wavenumber') 
#     results = pca.fit_transform(data2)
#     comps = pca.estimator_.n_components_
#     if comps < min_n_components:
#         pca = wrap(PCA(n_components=min_n_components),
#                        sample_dim='sample',
#                        reshapes='Wavenumber') 
#         results = pca.fit_transform(data2)
#         comps = pca.estimator_.n_components_

        
#     if verbose ==True:
#         print(f'components: {comps}')
#         print(f'EVR: {pca.estimator_.explained_variance_ratio_}')
        
#     ds_pca = xr.Dataset(
#         {
#             "PCA_loadings": xr.DataArray(
#                 data=pca.estimator_.components_.squeeze().T,
#                 dims=["Wavenumber", "PCA_loading"],
#                 coords={
#                     "Wavenumber": data2.Wavenumber.values,
#                     "PCA_loading": np.arange(comps),
#                 },
#             ),
#             "PCs": xr.DataArray(
#                 data=results,
#                 dims=["sample", "PCA_loading"],
#                 coords={
#                     "sample": data2.sample.values,
#                     "PCA_loading": np.arange(comps),
#                 },
#             ),
#             "PCA_explainedVariance": xr.DataArray(
#                 data=pca.estimator_.explained_variance_ratio_,
#                 dims=["PCA_loading"],
#                 coords={
#                     "PCA_loading": np.arange(comps),
#                 },
#             ),
#         }
#     )
#     if in_pipe == True:
#         sdc
#     else:
#         return ds_pca


# def my_fPCA(data, n_components=0.99, min_n_components=3, verbose=False):
#     """
#     This takes an Xarray DataArray and converts it to an FDataGrid for fPCA,
#     runs fPCA, and then returns a DataArray with fPCA loadings
#     and the transformed data.
    
#     Additionally, if n_components is between 0 and 1, my_fPCA will return
#     enough principal components to explain `n_components` worth of variance 
    
#     fPCA_loadings: the 'functions' that are used to transform the data. 
#                   These will have dimensions of (Wavenumber, n_components, 1)
#     fPCs: the values for the transformed data
#           These will have the dimensions of (samples, n_components, 1)
#     """
#     data2 = data.copy()
#     fdata = skfda.FDataGrid(data2, grid_points=data2.Wavenumber.values)
#     if n_components <=0:
#         print('Invalid number of components please provide a positive number')
#     if n_components < 1:
#         comps = data2.shape[0]
#         fpca = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
#             n_components=comps)
#         results = fpca.fit(fdata)
#         evr = results.explained_variance_ratio_
#         comps = len(evr[evr.cumsum() <= n_components])+1
#     else:
#         comps = n_components
    
#     if comps < min_n_components:
#         comps = min_n_components
#     fpca = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
#         n_components=comps)
#     results = fpca.fit_transform(fdata)
#     if verbose ==True:
#         print(f'components: {comps}')
#         print(f'EVR: {fpca.explained_variance_ratio_}')
        
#     ds_fpca = xr.Dataset(
#         {
#             "fPCA_loadings": xr.DataArray(
#                 data=fpca.components_.data_matrix.squeeze().T,
#                 dims=["Wavenumber", "fPCA_loading"],
#                 coords={
#                     "Wavenumber": data2.Wavenumber.values,
#                     "fPCA_loading": np.arange(comps),
#                 },
#             ),
#             "fPCs": xr.DataArray(
#                 data=results,
#                 dims=["sample", "fPCA_loading"],
#                 coords={
#                     "sample": data2.sample.values,
#                     "fPCA_loading": np.arange(comps),
#                 },
#             ),
#             "fPCA_explainedVariance": xr.DataArray(
#                 data=fpca.explained_variance_ratio_,
#                 dims=["fPCA_loading"],
#                 coords={
#                     "fPCA_loading": np.arange(comps),
#                 },
#             ),
#         }
#     )

#     return ds_fpca

    
class SIMCA_classifier:
    """
    SIMCA or DDSIMCA classifier for multiple classes.
    Adapted from
    """

    def __init__(self, cat_encoder, n_components=3, alpha=0.05,
                 simca_type="SIMCA"):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        alpha : float
            Significance level.
        """
        self.set_params(**{"n_components": n_components, "alpha": alpha,
                           "cat_encoder":cat_encoder, "simca_type":simca_type})
        self.classes_ = dict()
        self.score_functions = {'accuracy': accuracy_score,
                        'precision': precision_score,
                        'recall': recall_score,
                        'f1': f1_score,
                        'ckappa': cohen_kappa_score,
                        'prfs': precision_recall_fscore_support}
        

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"n_components": self.n_components, "alpha": self.alpha,
                "cat_encoder":self.cat_encoder,"simca_type":self.simca_type}


    def matrix_X_(self, X):
        """Check that observations are rows of X."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert (
            X.shape[1] == self.n_features_in_
        ), "Incorrect number of features given in X."
        return X

    def fit(self, X, y):
        """
        Fit the SIMCA models.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows which correspond to the
            class being modeled - this will be converted to a numpy array
            automatically.
        y : matrix-like
            Columns of classes, using True/False indicators for in or out of class
            Should work with the output of scikit-learn's OneHotEncoder
        
        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)
        
        for simca_class in np.unique(y, axis=0):
            simca_class_tup = tuple(simca_class)
            y_sub = y[np.where((y==simca_class_tup).all(axis=1))]
            X_sub = X[np.where((y==simca_class_tup).all(axis=1))]
            simca_class_name = (self.cat_encoder
                                .inverse_transform(
                                    simca_class.reshape(1, -1)
                                )[0][0])
            if self.simca_type == "SIMCA":
                sc = SIMCA(n_components=self.n_components, alpha=self.alpha) #inits a SIMCA for this class
            if self.simca_type == "DDSIMCA":
                sc = DDSIMCA(n_components=self.n_components, alpha=self.alpha) #inits a DDSIMCA for this class

            _ = sc.fit(X_sub, y_sub) #trains the SIMCA for this class

            self.classes_[simca_class_name] = sc

        return self

    def predict(self, X, labels=False):
        """
        Predict the class(es) for a given set of features.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        Returns
        -------
        predictions : ndarray
            Bolean array of whether a point belongs to this class.
        """
        #looping through each class to see if they fit in this class
        df=pd.DataFrame()
        for test_class in list(self.classes_.keys()):
            df_mini = pd.DataFrame()
            pred = self.classes_[test_class].predict(X)
            df[test_class] = pred
            
        df = df[self.cat_encoder.categories_[0]]

        if labels is True:
            return self.cat_encoder.inverse_transform(df)
        else:
            return df
        
    def score(self, X, y_true, metric='accuracy', averaging='weighted'):
        """
        Score the prediction.

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y : array-like
            Boolean array of whether or not each point belongs to the class.
        metric: str
            Based of the scikit learn metric chosen. Choices include:
            'accuracy', 'precision', 'recall', 'f1',
            'ckappa' for cohen_kappa_score, and 
            'prfs' for precision_recall_fscore_support

        Returns
        -------
        score : float
            accuracy
        """
        
        y_true=np.array(y_true)
        
        if len(y_true.shape)==1:
            y_true = y_true.reshape(-1,1)
            y_true = self.cat_encoder.transform(y_true)

        # if y.shape[1] > 1:
        #     y = self.cat_encoder.inverse_transform(y)
            
        Y_pred = self.predict(X, labels=False)
        assert (
            y_true.shape[0] == Y_pred.shape[0]
        ), "X and y do not have the same dimensions."
        
        if metric == 'accuracy':
            return self.score_functions[metric](y_true, Y_pred)
        else:
            return self.score_functions[metric](y_true, Y_pred, average=averaging)


    def _get_tags(self):
        """For compatibility with sklearn >=0.21."""
        return {
            "allow_nan": False,
            "binary_only": True,
            "multilabel": True,
            "multioutput": True,
            "multioutput_only": False,
            "no_validation": False,
            "non_deterministic": False,
            "pairwise": False,
            "preserves_dtype": [np.float64], #this may be wrong
            "poor_score": False,
            "requires_fit": True,
            "requires_positive_X": False,
            "requires_y": False,
            "requires_positive_y": False,
            "_skip_test": True,  # Skip since get_tags is unstable anyway
            "_xfail_checks": False,
            "stateless": False,
            "X_types": ["2darray"],
        }
    
class my_fPCA:
    """
    fPCA transformer to allow for piping and setting minimum
    number of PCs as well as PCs by percent of explained
    variance.
    """

    def __init__(self, n_components=0.99, min_n_components=3):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of fPCA components to use to model this class.
        min_n_components : int
            minimum number of components
        """
        if n_components <=0:    # check that n_components is valid
            print('Invalid number of components please provide a positive number')
        self.set_params(**{"n_components": n_components,
                          "min_n_components": min_n_components})

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"n_components": self.n_components,
                "min_n_components":self.min_n_components}

    def fit(self, X, y=None):
        """
        Fit the fPCA transformer.

        Parameters
        ----------
        X : xarray DataArray
            should include 'sample' and 'Wavenumber' dimensions
                    
        Returns
        -------
        self
        """
        data2 = X.copy().squeeze()
        fdata = skfda.FDataGrid(data2, grid_points=data2.Wavenumber.values)
        # if n_components <=0:    # check that 
        #     print('Invalid number of components please provide a positive number')
        # calculate the number of components necessary to reach our threshold
        if self.n_components < 1:
            comps = data2.shape[0]
            fpca = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
                n_components=comps)
            results = fpca.fit(fdata)
            evr = results.explained_variance_ratio_
            self.comps = len(evr[evr.cumsum() <= self.n_components])+1
        else:
            self.comps = self.n_components
        # check that number of components matches
        if self.comps < self.min_n_components:
            self.comps = self.min_n_components

        self.model = skfda.preprocessing.dim_reduction.feature_extraction.FPCA(
            n_components=self.comps)
        self.model.fit(fdata)
        self.explained_variance_ratio = xr.DataArray(data = self.model.explained_variance_ratio_,
                                                     dims=["fPCA_loading"],
                                                     coords={"fPCA_loading": np.arange(self.comps)})
        self.loadings = xr.DataArray(data = self.model.components_.data_matrix.squeeze().T,
                                     dims=["Wavenumber", "fPCA_loading"],
                                     coords={"Wavenumber": data2.Wavenumber.values,
                                             "fPCA_loading": np.arange(self.comps)})
                                     
        return self

    def transform(self, X, y=None):
        """
        Transform X into the reduced dimensions

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y: ignored

        Returns
        -------
        Principal component scores.
        """
        data2 = X.copy()
        fdata = skfda.FDataGrid(data2, grid_points=data2.Wavenumber.values)   
        self.fPC_scores = xr.DataArray(data=self.model.transform(fdata, y),
                                       dims=["sample", "fPCA_loading"],
                                       coords={"sample": data2.sample.values,
                                               "fPCA_loading": np.arange(self.comps)},
                                       name='fPCs')
        
        return self.fPC_scores
        
    def fit_transform(self, X, y=None):
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.
        """
        return self.fit(X, y).transform(X, y)

    
class my_PCA:
    """
    PCA transformer to allow for piping and setting minimum
    number of PCs as well as PCs by percent of explained
    variance.
    """

    def __init__(self, n_components=0.99, min_n_components=3):
        """
        Instantiate the class.

        Parameters
        ----------
        n_components : int
            Number of PCA components to use to model this class.
        min_n_components : int
            minimum number of components
        """
        if n_components <=0:    # check that n_components is valid
            print('Invalid number of components please provide a positive number')
        self.set_params(**{"n_components": n_components,
                          "min_n_components": min_n_components})

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {"n_components": self.n_components,
                "min_n_components":self.min_n_components}

    def fit(self, X, y=None):
        """
        Fit the PCA transformer.

        Parameters
        ----------
        X : xarray DataArray
            should include 'sample' and 'Wavenumber' dimensions
                    
        Returns
        -------
        self
        """
        
        data2 = X.copy().squeeze()
        self.model=wrap(PCA(n_components=self.n_components),
                        sample_dim='sample',
                        reshapes='Wavenumber') 
        self.model.fit(data2)
        self.comps = self.model.estimator_.n_components_
        if self.comps < self.min_n_components:
            self.model = wrap(PCA(n_components=self.min_n_components),
                           sample_dim='sample',
                           reshapes='Wavenumber') 
            self.model.fit(data2)
            self.comps = self.model.estimator_.n_components_

        self.model.fit(data2)
        self.explained_variance_ratio = xr.DataArray(data = self.model.estimator_.explained_variance_ratio_,
                                                     dims=["PCA_loading"],
                                                     coords={"PCA_loading": np.arange(self.comps)})
        self.loadings = xr.DataArray(data = self.model.estimator_.components_.squeeze().T,
                                     dims=["Wavenumber", "PCA_loading"],
                                     coords={"Wavenumber": data2.Wavenumber.values,
                                             "PCA_loading": np.arange(self.comps)})
        
        return self

    def transform(self, X, y=None):
        """
        Transform X into the reduced dimensions

        Parameters
        ----------
        X : matrix-like
            Columns of features; observations are rows - will be converted to
            numpy array automatically.
        y: ignored

        Returns
        -------
        Principal component scores.
        """
        data2 = X.copy()
        
        self.PC_scores = xr.DataArray(data=self.model.transform(data2),
                                      dims=["sample", "PCA_loading"],
                                      coords={"sample": data2.sample.values,
                                              "PCA_loading": np.arange(self.comps)},
                                      name='PCs')
        
        return self.PC_scores
        
    def fit_transform(self, X, y=None):
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.

        """
        return self.fit(X, y).transform(X, y)
