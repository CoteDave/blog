import numpy as np
import pandas as pd
import scipy.stats as ss
from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist
#from math import isinf
#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.pairwise import manhattan_distances
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from rdr_scorer import RdR_scorer

"""
The idea of this index is to estimate the accuracy of direction’s changes of the projected data, 
i.e., if the future value will increase or decrease when compared to current value. 
We should use POCID in a complementary way to the analysis of the prediction errors. 
It is not advisable to make a decision based solely on POCID values. 
POCID =Ph t=1 Dt h ×100 (32) 
"""

"""
y_true = pd.DataFrame([0,
1,
2,
3,
4,
3,
2,
1,
2,
3,
4,
5,
6,
5,
4,
3,
4,
5,
6,
7,
8])

y_pred = pd.DataFrame([3,
4,
5,
10,
1,
1,
2,
3,
4,
5,
6,
7,
8,
9,
1,
2,
41,
3,
12,
11,
7])

y_train = pd.DataFrame([1,2,1,0,1,2])
"""

    
def clip_zeros(y_true):
    return np.where(y_true == 0, 0.000001, y_true)

def POCID(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    """Prediction on change of direction"""
    nobs = len(y_pred)
    pocid = (100 * (100 * np.mean((np.diff(y_true[-nobs:].ravel()) * np.diff(y_pred.ravel())) > 0)  )) / 75
    return pocid


def jensen_shannon_distance(y_true, y_pred): 
    """ 
    method to compute the Jenson-Shannon Distance  
    between two probability distributions 
    """ 

    # convert the vectors into numpy arrays in case that they aren't 
    p = np.array(y_true) 
    q = np.array(y_pred) 
     
    # calculate m 
    m = (p + q) / 2 
     
    # compute Jensen Shannon Divergence 
    divergence = (ss.entropy(p, m) + ss.entropy(q, m)) / 2 
     
    # compute the Jensen Shannon Distance 
    distance = np.sqrt(divergence) 
     
    return distance 

def MAE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_error = abs(y_pred - y_true)
    mae = y_error.mean()
    return mae

# MFE is the difference between observed and forecasted value
def MFE(y_true, y_pred):
    return np.mean(y_true - y_pred)

def MAPE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_p_error = (abs(y_pred - y_true) / y_true) * 100
    mape = y_p_error.mean()
    return mape

def SMAPE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    smape = np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred)/2))) * 100
    return smape

def FA_MAPE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_p_error = (abs(y_pred - y_true) / y_true) * 100
    mape = y_p_error.mean()
    FA = 100 - mape
    return FA

def MSE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_error = (y_pred - y_true) * (y_pred - y_true)
    mse = y_error.mean()
    return mse

def NMSE(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    return mse / (np.sum((y_true - np.mean(y_true)) ** 2)/(len(y_true)-1))


def RMSE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_error = (y_pred - y_true) * (y_pred - y_true)
    mse = y_error.mean()
    rmse = np.sqrt(mse)
    return rmse

def RMSPE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))*100

def FA_RMSPE(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    return 100 - (np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))*100)

def U_THEIL(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    y_calc_num = y_true.copy().astype(float)
    y_calc_dem = y_true.copy().astype(float)
    for row in range(0, len(y_true)):
        if row == 0:
            pass
        else:
            y_calc_num[row] = np.square((y_pred[row] - y_true[row]) / y_true[row-1])
            y_calc_dem[row] = np.square((y_true[row] - y_true[row-1]) / y_true[row-1])
    y_calc_num = y_calc_num[1:]
    y_calc_dem = y_calc_dem[1:]
    u_theil = np.sqrt(np.sum(y_calc_num) / np.sum(y_calc_dem))
    return u_theil
    

def R2(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    sse = sum((y_true - y_pred)**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score[0]

def R2_ADJ(y_true, y_pred, n_vars = 1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sse = sum((y_true - y_pred)**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    r2_adj = 1-(1-r2_score[0])*(len(y_true)-1)/(len(y_true)-n_vars-1)
    return r2_adj

# This evaluation metric is used to over come some of the problems of MAPE and
# is used to measure if the forecasting model is better than the naive model or
# not.
def MASE_LAST(y_true, y_pred, y_train):
    last_y = y_train[-1:].values.ravel()[0]
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    error = y_true - y_pred
    error = error.ravel()
    y_true_naïve = np.concatenate((np.array([last_y]).ravel(), y_true[:-1].ravel()), axis = 0)
    #print(y_true_naïve)
    error_naïve = y_true.ravel() - y_true_naïve
    abs_errors = abs(error)/abs(error_naïve)
    abs_errors = np.nan_to_num(abs_errors, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(abs_errors)

# This evaluation metric is used to over come some of the problems of MAPE and
# is used to measure if the forecasting model is better than the naive model or
# not.
def MASE_AVG(y_true, y_pred, y_train):
    last_y = np.mean(y_train)
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    error = y_true - y_pred
    error = error.ravel()
    y_true_naïve = np.concatenate((np.array([last_y]).ravel(), y_true[:-1].ravel()), axis = 0)
    error_naïve = y_true.ravel() - y_true_naïve
    abs_errors = abs(error)/abs(error_naïve)
    abs_errors = np.nan_to_num(abs_errors, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(abs_errors)

def AIC(y_true, y_pred, n_vars):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    """
    Return an AIC score for a model.
    Input:
    y: array-like of shape = (n_samples) including values of observed y
    y_pred: vector including values of predicted y
    p: int number of predictive variable(s) used in the model
    Output:
    aic_score: int or float AIC score of the model
    Raise TypeError if y or y_pred are not list/tuple/dataframe column/array.
    Raise TypeError if elements in y or y_pred are not integer or float.
    Raise TypeError if p is not int.
    Raise InputError if y or y_pred are not in same length.
    Raise InputError if length(y) <= 1 or length(y_pred) <= 1.
    Raise InputError if p < 0.
    """

    # Package dependencies
    #import numpy as np
    #import pandas as pd

    # User-defined exceptions
    class InputError(Exception):
        """
        Raised when there is any error from inputs that no base Python exceptions cover.
        """
        pass

## Length condition: length of y and y_pred should be equal, and should be more than 1
    ### check if y and y_pred have equal length
    if not len(y_true) == len(y_pred):
        raise InputError("Expect equal length of y and y_pred")
    ### check if y and y_pred length is larger than 1
    elif len(y_true) <= 1 or len(y_pred) <= 1:
        raise InputError("Expect length of y and y_pred to be larger than 1")
    else:
        n = len(y_true)

    # Calculation
    resid = np.subtract(y_pred, y_true)
    rss = np.sum(np.power(resid, 2))
    aic_score = n*np.log(rss/n) + 2*n_vars
    
    if np.isinf(aic_score):
        aic_score = 0

    return aic_score

def BIC(y_true, y_pred, n_vars):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    """
    Returns the BIC score of a model.
    Input:-
    y: the labelled data in shape of an array of size  = number of samples. 
        type = vector/array/list
    y_pred: predicted values of y from a regression model in shape of an array
        type = vector/array/list
    p: number of variables used for prediction in the model.
        type = int
    Output:-
    score: It outputs the BIC score
        type = int
    Tests:-
    Raise Error if length(y) <= 1 or length(y_pred) <= 1.
    Raise Error if length(y) != length(y_pred).
    Raise TypeError if y and y_pred are not vector.
    Raise TypeError if elements of y or y_pred are not integers.
    Raise TypeError if p is not an int.
    Raise Error if p < 0.
    """

    # package dependencies
    #import numpy as np
    #import pandas as pd
    #import collections
    
# Length exception
    if not len(y_true) == len(y_pred):
        raise TypeError("Equal length of observed and predicted values expected.")
    else:
        n = len(y_true)

    # Score

    residual = np.subtract(y_pred, y_true)
    SSE = np.sum(np.power(residual, 2))
    BIC = n*np.log(SSE/n) + n_vars*np.log(n)
    
    if np.isinf(BIC):
        BIC = 0
        
    return BIC


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
  
def replace_erronous_values(pred_ci):
    pred_ci = pred_ci.fillna(0)
    pred_ci = pred_ci.loc[pred_ci['CLE'].astype(str) != '0']
    for col in list(pred_ci.columns): 
        print(col)
        print(col.find('HB') > -1)
        if col.find('LB') > -1 :
            pred_ci[col] = pred_ci[col].fillna(np.max(pred_ci[col].loc[pred_ci[col] != np.inf]))
            pred_ci[col] = pred_ci[col].replace(np.inf, np.max(pred_ci[col].loc[pred_ci[col] != np.inf]))
            pred_ci[col] = pred_ci[col].replace(-np.inf, np.max(pred_ci[col].loc[pred_ci[col] != np.inf]))
        elif col.find('HB') > -1 : 
            pred_ci[col] = pred_ci[col].fillna(np.min(pred_ci[col].loc[pred_ci[col] != -np.inf]))
            pred_ci[col] = pred_ci[col].replace(np.inf, np.min(pred_ci[col].loc[pred_ci[col] != -np.inf]))
            pred_ci[col] = pred_ci[col].replace(-np.inf, np.min(pred_ci[col].loc[pred_ci[col] != -np.inf]))
        else:
            pred_ci[col] = pred_ci[col].fillna(0)
            pred_ci[col] = pred_ci[col].replace(np.inf,0)
            pred_ci[col] = pred_ci[col].replace(-np.inf,0)
    return pred_ci
   
#n_vars = 1
def all_metrics(y_true, y_pred, y_train, n_vars):
    #y_pred = y_pred.fillna(0)
    
    y_true_clipped = pd.DataFrame(clip_zeros(y_true))
    y_true_clipped.columns = list(y_true.columns)
    y_true_clipped.index = y_true.index
    #errors_df = pd.DataFrame(y_true)
    y_true = y_true_clipped
    #errors_df.columns = ['Y_TRUE']
    #errors_df['Y_TEST_PRED'] = y_pred.values.astype(float)
    errors_df = pd.DataFrame(y_pred.values - y_true.values) #errors_df['Y_TEST_PRED'] - errors_df['Y_TRUE']
    errors_df.columns = ['PRED_ERRORS']
    errors_df['HB_POCID'] = POCID(y_true, y_pred)
    errors_df['LB_JENSEN_SHANNON_DIST'] = jensen_shannon_distance(y_true, y_pred)[0]
    errors_df['LB_MAE'] = MAE(y_true, y_pred)
    errors_df['N0_MFE'] = MFE(y_true.values, y_pred.values)
    
    errors_df['LB_MAPE'] = MAPE(y_true, y_pred)
    errors_df['LB_SMAPE'] = SMAPE(y_true, y_pred)
    errors_df['HB_FA_MAPE'] = FA_MAPE(y_true, y_pred)
    
    errors_df['LB_MAX_ERROR'] = abs(np.max(y_pred.values - y_true.values))
    errors_df['LB_MSE'] = MSE(y_true, y_pred)
    errors_df['LB_NMSE'] = NMSE(y_true, y_pred)[0]
    errors_df['LB_RMSE'] = RMSE(y_true, y_pred)
    errors_df['LB_RMSPE'] = RMSPE(y_true, y_pred)
    errors_df['HB_FA_RMSPE'] = FA_RMSPE(y_true, y_pred)
    
    errors_df['LB_U_THEIL'] = U_THEIL(y_true, y_pred)
    
    errors_df['HB_R2'] = R2(y_true.values, y_pred.values)
    errors_df['HB_R2_ADJ'] = R2_ADJ(y_true.values, y_pred.values, n_vars = n_vars)
    #pred_ci_econom['HB_R2_ADJ'] = np.where(pred_ci_econom['HB_R2_ADJ'].isna() == True, pred_ci_econom['HB_R2'], pred_ci_econom['HB_R2_ADJ'])
    errors_df['HB_R2_ADJ'] = np.where(errors_df['HB_R2_ADJ'].isna() == True, errors_df['HB_R2'], errors_df['HB_R2_ADJ'])
    errors_df['HB_R2_ADJ'] = np.where(errors_df['HB_R2_ADJ'] == np.inf, errors_df['HB_R2'], errors_df['HB_R2_ADJ'])
    errors_df['HB_R2_ADJ'] = np.where(errors_df['HB_R2_ADJ'] == -np.inf, errors_df['HB_R2'], errors_df['HB_R2_ADJ'])
   
    
    errors_df['LB_MASE_LAST'] = MASE_LAST(y_true.values, y_pred.values, y_train)
    errors_df['LB_MASE_AVG'] = MASE_AVG(y_true.values, y_pred.values, y_train)
    
    if len(y_true) == 1:
        y_true2 = np.concatenate((y_true, y_true), axis = 0)
        y_pred2 = np.concatenate((y_pred, y_pred), axis = 0)
        errors_df['LB_AIC'] = AIC(y_true2, y_pred2, n_vars)
        errors_df['LB_BIC'] = BIC(y_true2, y_pred2, n_vars)
    else:
        errors_df['LB_AIC'] = AIC(y_true, y_pred, n_vars)
        errors_df['LB_BIC'] = BIC(y_true, y_pred, n_vars)
    
    errors_df['LB_DTW_DIST'] = dtw.distance(y_true.values, y_pred.values)
    
    for dist in ['euclidean', 'chebyshev',  'hamming', 'jaccard', 'mahalanobis', 'minkowski']:   
        d2, c2, acc2, p2 = accelerated_dtw(y_true, y_pred, dist)
        errors_df['LB_DTW_' + dist.upper()] = d2
    
    errors_df = errors_df.reindex(sorted(errors_df.columns), axis=1)
    
    errors_df.index = y_true.index
    
    #errors_df.columns = [col.replace('LB_', '') for col in errors_df.columns]
    #errors_df.columns = [col.replace('HB_', '') for col in errors_df.columns]
    
    return errors_df

def rank_by_score(errors_df):
    #errors_df = errors_df.loc[~errors_df['Y_PRED'].isna()]
    #errors_df.to_csv('error_df.csv', sep = ';')
    lb_scores = errors_df[[col for col in errors_df if col.startswith('LB_')]]
    hb_scores = errors_df[[col for col in errors_df if col.startswith('HB_')]]
    
    #errors_df = replace_erronous_values(errors_df)
    
    #errors_df = errors_df.fillna(0.0000000000001)
    
    scaler = MinMaxScaler(feature_range = (0, 100)) 
    
    lb_scores = scaler.fit_transform(lb_scores).astype(float)
    lb_scores = 100 - lb_scores
    
    hb_scores = scaler.fit_transform(hb_scores).astype(float)
    
    global_scores = np.concatenate([lb_scores, hb_scores], axis = 1)
    
    
    
    global_scores = np.mean(global_scores, axis = 1)    
    errors_df['GLOBAL_SCORE'] = global_scores
    #errors_df['GLOBAL_SCORE'] = np.where(errors_df['GLOBAL_SCORE'].isna() == True, errors_df['HB_FA_MAPE'], errors_df['GLOBAL_SCORE'])
    #errors_df['GLOBAL_SCORE'] = errors_df['GLOBAL_SCORE'].fillna(errors_df['HB_FA_MAPE'])
    global_scores = global_scores - np.max(global_scores)
    errors_df['MODEL_RANK'] = rankdata(global_scores, method='min')
    errors_df['INDCT_BETTER_THAN_NAIVE'] = np.where(errors_df['LB_U_THEIL'] < 1, 1, 0)
    
    errors_df = errors_df.sort_values(by = ['GLOBAL_SCORE'], ascending = False)
    return errors_df

def get_score_df(X_train,
                 y_true, y_pred,
                 y_ts,
                 y_colname,
                 n_leads,
                 model_name,
                 freq,
                 pred_ci):
             
    pred_test_val = pred_ci.copy()
    
    pred_test_val['Y_TRUE'] = y_true.values
    
    scores = all_metrics(y_true = pred_test_val[['Y_TRUE']], 
                           y_pred = pred_test_val[['Y_PRED']], 
                           y_train = y_ts, 
                           n_vars = len(X_train.columns))
    
    
    rdr_s = RdR_scorer()
    rdr_s.fit(y_ts.iloc[:-n_leads, :], 
              y_colname, n_leads, 
              y_true.copy(), 
              pred_test_val[['Y_PRED']], 
              model_name = model_name, 
              freq = freq)
    rdr_score_result = rdr_s.score()
    rdr_interp = rdr_s.get_rdr_interpretation()
    scores['RDR_SCORE'] = rdr_score_result
    scores['RDR_INTERPRET'] = rdr_interp
    scores['RANDOM_WALK_WITH_DRIFT'] = rdr_s._random_walk['Y_PRED'].values
    scores['RANDOM_WALK_WITH_DRIFT_RMSE'] = rdr_s._raw_rmse_rw
    
    pred_ci = pd.merge(pred_ci, scores, how = 'left', left_index = True, right_index = True)
    
    return pred_ci, rdr_s
