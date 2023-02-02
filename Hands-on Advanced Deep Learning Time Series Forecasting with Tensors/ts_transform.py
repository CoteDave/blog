from sklearn.utils.validation import check_array #, warn_if_not_float
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import pandas as pd

 
class TransformerLog():
    
    def __init__(self, copy=True):
        self.copy = copy
        self.transformed_ = None
        self.scaler_ = None
        self.X_ = None
        
    def fit_transform(self, X, y=None):
        """Don't trust the documentation of this module!
        Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, accept_sparse="csc",
                         ensure_2d=False)
    
        
        X = X.astype(np.float)
        
        if np.any(X < 0) == True:
            scaler = MinMaxScaler(feature_range = (1, 1000))
            self.scaler_ = scaler
            X = scaler.fit_transform(X)
            
        X = np.log1p(X)
        X = X.astype(np.float)
        
        self.transformed_ = 1.0
        self.X_ = X

        return X
    
    def transform(self, X, y=None):
        """Don't trust the documentation of this module!
        Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, accept_sparse="csc",
                         ensure_2d=False)
    
        
        X = X.astype(np.float)
        
        if np.any(X < 0) == True:
            scaler = MinMaxScaler(feature_range = (1, 1000))
            self.scaler_ = scaler
            X = scaler.fit_transform(X)
            
        X = np.log1p(X)
        X = X.astype(np.float)
        
        self.transformed_ = 1.0
        self.X_ = X

        return X
    
    def inverse_transform(self, X, copy=None):
        #check_is_fitted(self, 'transformed_')
        
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, accept_sparse="csc", ensure_2d=False)
        
        X = X.astype(np.float)
        
        if self.transformed_ is not None:
            X = np.expm1(X)
            if self.scaler_ is not None:
                print('print_scaler_yes')
                X = self.scaler_.inverse_transform(X)      
        
        X = X.astype(np.float)
        
        return X


class TransformerBoxCox():
    
    def __init__(self, copy=True):
        self.copy = copy
        self.transformed_ = None
        self.scaler_ = None
        self.X_ = None
        self._translation = None
        
    def fit_transform(self, X, y=None):
        """Don't trust the documentation of this module!
        Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        #X = check_array(X, copy=self.copy, accept_sparse="csc",ensure_2d=False)
        
        try:
            colnames_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
        except:
            X = pd.DataFrame(X)
            colnames_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
            
        
        X = X.astype(float)
        
        if min(np.min(X.loc[:, colnames_numerical]).dropna()) <= 0:
            self._translation = abs(min(np.min(X.loc[:, colnames_numerical]).dropna())) * 2
            X = X + self._translation
        else:
            self._translation = 0
        
        try:
            scaler = PowerTransformer(method = 'box-cox')
        except:
            scaler = PowerTransformer(method = 'yeo-johnson')
            
        X = scaler.fit_transform(X)            
        X = X.astype(float)
        
        self.transformed_ = 1.0
        self.scaler_ = scaler
        self.X_ = X

        return X
    
    def transform(self, X, y=None):
        """Don't trust the documentation of this module!
        Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        #X = check_array(X, copy=self.copy, accept_sparse="csc",ensure_2d=False)
        
        if self._translation != 0:
            X = X + self._translation
            X = self.scaler_.transform(X)           
        else:
            X = self.scaler_.transform(X)      
        
        X = X.astype(float)
        return X
    
    def inverse_transform(self, X, copy=None):
        #check_is_fitted(self, 'transformed_')
        
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, accept_sparse="csc", ensure_2d=False)
        
        X = X.astype(float)
        
        if self.transformed_ is not None and self.scaler_ is not None:
            X = self.scaler_.inverse_transform(X)        
        if self._translation != 0:
            X = X - self._translation  
        
        X = X.astype(float)
        
        return X


    
class TransformerDiff():
    
    def __init__(self, copy=True, order = 1):
        self.copy = copy
        self.order = order
        self.transformed_ = None
        #self.scaler_ = None
        self.X_ = None
        self.last_value_ = None
        
    def fit_transform(self, X, y=None):
        """Don't trust the documentation of this module!
        Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        #X = check_array(X, copy=self.copy, accept_sparse="csc",ensure_2d=False)
        
        self._x_in_f = X.copy()
        self._pred_value_test = np.array(X[-1, 0]).reshape(-1,1)
        
        
        if self.order > 3:
            return('ERROR: Order > 4 is too high and useless!!!')
            
        if self.order < 1:
            print('The time serie is already Stationary! Diff parameter set to False')
            return X
        
        X = X.astype(np.float)
        
        try:
            X.shape[1]
        except:
            try:
                X = X.reshape(-1,1)
            except:
                X = np.array(X).reshape(-1,1)
        
        '''
        if np.any(X < 0) == True:
            scaler = MinMaxScaler(feature_range = (1, 1000))
            X = scaler.fit_transform(X)
        '''
        
        self._x_in_o = X.copy()
        self._pred_value_test = np.array(X[-1, 0]).reshape(-1,1)
        
        i = 0
        #col = 1
        for col in range(X.shape[1]):
            serie = pd.Series(X[:, col])
            
            if self.order == 1:
                X_diff = serie.diff()
                if i == 0:
                    diff = np.array(X_diff)
                    last_value = X[0, col]
                    
                    pred_value = np.array(X[-1, 0]).reshape(-1,1)
                else:
                    diff = np.column_stack((diff, np.array(X_diff)))
                    last_value = np.column_stack((last_value, X[0, col]))
                    
                    pred_value = np.column_stack((pred_value, X[-1, col]))
                    
            elif self.order == 2:
                X_diff = serie.diff()
                X_diff2 = X_diff.diff()
                if i == 0:
                    diff = np.array(X_diff2)
                    last_value = np.array(X[0, col]).reshape(-1,1)
                    last_value = np.concatenate((last_value, np.array(X_diff.dropna().head(1)).reshape(-1,1)), axis = 0)
                    
                    pred_value = np.array(X[-1, col]).reshape(-1,1)
                    pred_value = np.concatenate((pred_value, np.array(X_diff.dropna().tail(1)).reshape(-1,1)), axis = 0)
                else:
                    diff = np.column_stack((diff, np.array(X_diff2)))
                    last_value2 = np.array(X[0, col]).reshape(-1,1)
                    last_value2 = np.concatenate((last_value2, np.array(X_diff.dropna().head(1)).reshape(-1,1)), axis = 0)
                    last_value = np.column_stack((last_value, last_value2))
                    
                    pred_value2 = np.array(X[-1, col]).reshape(-1,1)
                    pred_value2 = np.concatenate((pred_value2, np.array(X_diff.dropna().tail(1)).reshape(-1,1)), axis = 0)
                    pred_value = np.column_stack((pred_value, pred_value2))
                    
            elif self.order == 3:
                X_diff = serie.diff()
                X_diff2 = X_diff.diff()
                X_diff3 = X_diff2.diff()
                if i == 0:
                    diff = np.array(X_diff3)
                    last_value = np.array(X[0, col]).reshape(-1,1)
                    last_value = np.concatenate((last_value, np.array(X_diff.dropna().head(1)).reshape(-1,1)), axis = 0)
                    last_value = np.concatenate((last_value, np.array(X_diff2.dropna().head(1)).reshape(-1,1)), axis = 0)
                    
                    pred_value = np.array(X[-1, col]).reshape(-1,1)
                    pred_value = np.concatenate((pred_value, np.array(X_diff.dropna().tail(1)).reshape(-1,1)), axis = 0)
                    pred_value = np.concatenate((pred_value, np.array(X_diff2.dropna().tail(1)).reshape(-1,1)), axis = 0)
                else:
                    diff = np.column_stack((diff, np.array(X_diff3)))
                    last_value2 = np.array(X[0, col]).reshape(-1,1)
                    last_value2 = np.concatenate((last_value2, np.array(X_diff.dropna().head(1)).reshape(-1,1)), axis = 0)
                    last_value2 = np.concatenate((last_value2, np.array(X_diff2.dropna().head(1)).reshape(-1,1)), axis = 0)
                    last_value = np.column_stack((last_value, last_value2))
                    
                    pred_value2 = np.array(X[-1, col]).reshape(-1,1)
                    pred_value2 = np.concatenate((pred_value2, np.array(X_diff.dropna().tail(1)).reshape(-1,1)), axis = 0)
                    pred_value2 = np.concatenate((pred_value2, np.array(X_diff2.dropna().tail(1)).reshape(-1,1)), axis = 0)
                    pred_value = np.column_stack((pred_value, pred_value2))

            i = i + 1

        X = np.array(diff)
        X = X.astype(np.float)
        
        self.transformed_ = True
        #self.scaler_ = scaler
        self.X_ = X
        self.last_value_ = np.array(last_value)
        self.pred_value_ = np.array(pred_value)
        
  
        return X
    
    def inverse_transform(self, X, copy=None):
        #check_is_fitted(self, 'transformed_')
        
        if self.order < 1:
            return X
        
        #X = X_diff
        #X = X[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(X)]
        
        copy = copy if copy is not None else self.copy
        #X = check_array(X, copy=copy, accept_sparse="csc", ensure_2d=False)
        
        X = X.astype(np.float)
        
        try:
            X.shape[1]
        except:
            X = X.reshape(-1,1)
        #last_value[-3, 0]
        
        if self.transformed_ is not None:
            i = 0
            #j = 0
            #col = 1
            #col = 1
            for col in range(X.shape[1]):
                #serie = pd.Series(X[:, col])
                
                if self.order == 1:
                    try:
                        self.last_value_[0, col]
                    except:
                        self.last_value_ = self.last_value_.reshape(-1, 1)
                    X_first = np.array(self.last_value_[0, col]).reshape(-1,1)
                    undiff = np.array([]).reshape(-1,1)
                    undiff = np.concatenate((undiff, X_first), axis = 0)
                    for j in range(0, len(X[:, col])):
                        undiff = np.concatenate((undiff, np.array(X[j, col] + undiff[j, 0]).reshape(-1, 1)), axis = 0)
                        #print(i,col,j)
                    if i == 0:
                        undiff_all = np.array(undiff)
                    else:
                        undiff_all = np.column_stack((undiff_all, np.array(undiff)))
                        
                elif self.order == 2:
                    X_first1 = np.array(self.last_value_[-1, col]).reshape(-1,1)
                    X_first2 = np.array(self.last_value_[-2, col]).reshape(-1,1)
                    undiff = np.array([]).reshape(-1,1)
                    undiff = np.concatenate((undiff, X_first2), axis = 0)
                    undiff = np.concatenate((undiff, X_first2 + X_first1), axis = 0)
                    for j in range(0, len(X[:, col])):
                        undiff = np.concatenate((undiff, np.array(X[j, col] + (undiff[j+1, 0]-undiff[j, 0]) + undiff[j+1, 0]).reshape(-1, 1)), axis = 0)
                        #print(i,col,j)
                    if i == 0:
                        undiff_all = np.array(undiff)
                    else:
                        undiff_all = np.column_stack((undiff_all, np.array(undiff)))
                        
                elif self.order == 3:
                    X_first1 = np.array(self.last_value_[-1, col]).reshape(-1,1)
                    X_first2 = np.array(self.last_value_[-2, col]).reshape(-1,1)
                    X_first3 = np.array(self.last_value_[-3, col]).reshape(-1,1)
                    undiff = np.array([]).reshape(-1,1)
                    undiff = np.concatenate((undiff, X_first3), axis = 0)
                    undiff = np.concatenate((undiff, X_first2 + X_first3), axis = 0)
                    undiff = np.concatenate((undiff, X_first3 + X_first2 + X_first2 + X_first1), axis = 0)
                    for j in range(0, len(X[:, col])):
                        undiff = np.concatenate((undiff, np.array((X[j, col] + (undiff[j+2, 0] - undiff[j+1, 0]) - (undiff[j+1, 0] - undiff[j, 0]) ) + (undiff[j+2, 0] - undiff[j+1, 0]) + undiff[j+2, 0]).reshape(-1, 1)), axis = 0)
                        #print(i,col,j)
                    if i == 0:
                        undiff_all = np.array(undiff)
                    else:
                        undiff_all = np.column_stack((undiff_all, np.array(undiff)))
                
                i = i + 1           
        return undiff_all
    
    def forecast_transform(self, y_pred, copy=None):
        #check_is_fitted(self, 'transformed_')        
        
        copy = copy if copy is not None else self.copy
        
        y_pred = y_pred.astype(np.float)
        #print(self.y_pred)
        #print(self.transformed_)
        try:
            y_pred.shape[1]
        except:
            y_pred = y_pred.reshape(-1, 1)
            
        if self.transformed_ == True:
            #print(self.transformed_)
            i = 0
            for col in range(y_pred.shape[1]):
                #print(col)
                pred_first = y_pred[0, col]
                for j in range(0, len(y_pred[:, col])):
                    #print(pred_first,col,j)
                    if col == 0:
                        if j == 0:
                            try:
                                self.pred_value_[:, col]
                            except:
                                self.pred_value_ = self.pred_value_.reshape(-1, 1)
                                
                            y_pred_undiff = np.array(np.sum(self.pred_value_[:, col]) + pred_first).reshape(-1,1)
                            #print(y_pred_undiff)
                        else:
                            #print(y_pred_undiff)
                            y_pred_undiff = np.concatenate((y_pred_undiff, np.array(y_pred_undiff[j-1, col] + y_pred[j, col]).reshape(-1,1) ), axis = 0)
                        y_pred_undiff_all = np.array(y_pred_undiff)
                    else:
                        if j == 0:
                            
                            try:
                                self.pred_value_[:, col]
                            except:
                                self.pred_value_ = self.pred_value_.reshape(-1, 1)
                                
                        
                            y_pred_undiff = np.array(np.sum(self.pred_value_[:, col]) + pred_first).reshape(-1,1)
                        else:
                            y_pred_undiff = np.concatenate((y_pred_undiff, np.array(y_pred_undiff[j-1, 0] + y_pred[j, col]).reshape(-1,1) ), axis = 0)
                        
                        y_pred_undiff_all2 = np.array(y_pred_undiff)    
                        #y_pred_undiff_all = np.column_stack((y_pred_undiff_all, y_pred_undiff ))
                        if j == len(y_pred[:, col])-1:
                            y_pred_undiff_all = np.concatenate((y_pred_undiff_all, y_pred_undiff_all2 ), axis = 1)
                        else:
                            pass
                
                i = i + 1           
        return y_pred_undiff_all
    
class TransformerLogDiff:       
    
    def __init__(self, copy=True):
        self.copy = copy
        self.log_ = TransformerLog()
        self.diff_ = TransformerDiff()
        #self.scaler_ = None
        self.X_ = None
        self.last_value_ = None
        
    def fit_transform(self, X, y=None):
        X_log = self.log_.fit_transform(X)
        #X_logdiff = self.diff_.fit_transform(X_log)
        
        return X_log
    
    def inverse_transform(self, X_logdiff, y=None):
        #X_log = self.diff_.inverse_transform(X_logdiff)
        X = self.log_.inverse_transform(X_logdiff)
        
        return X_log
    
    def forecast_transform(self, y_pred):
        #y_undiff = self.diff_.forecast_transform(y_pred)
        y_undiff_unlog = self.log_.inverse_transform(y_pred)
        
        return y_undiff_unlog
    
        
