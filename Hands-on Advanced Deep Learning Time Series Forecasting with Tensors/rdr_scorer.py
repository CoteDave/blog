import pandas as pd
import numpy as np
from dtw_funcs import slow_dtw_distance
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


from dateutil.relativedelta import relativedelta

from pandas.tseries.offsets import MonthEnd
from datetime import datetime, date
from datetime import date, timedelta
#####################
#CREATED BY DAVE COTE
#####################

"""
__author__ = "Dave Cote"
__copyright__ = "Copyright 2020, The RdR score experiment"
__credits__ = ["Dave Cote"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "None"
__email__ = "None"
__status__ = "Experimental"
"""




#from func_utils import get_pred_dates


def get_pred_dates(freq, X_pred, date_colname, n_leads):
    
    if freq == 12:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(months=1)+ MonthEnd(0) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(months=i) + MonthEnd(0)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]  
    
    elif freq == 52:
        print(X_pred.index)
        print(max(X_pred.index))
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(weeks=1) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(weeks=i)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]    
            
    elif freq == 4:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(months=3)+ MonthEnd(0) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(months=i*3) + MonthEnd(0)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]    
    
    elif freq == 2:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(months=6)+ MonthEnd(0) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(years=i*6) + MonthEnd(0)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]  
            
    elif freq == 1:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(years=1)+ MonthEnd(0) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(years=i) + MonthEnd(0)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]  
            
    elif freq >= 250 and freq <= 368:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(days=1) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(days=i)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]   
            
    elif freq == 1638:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=60)])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i*60) ])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]   
            
    elif freq == 3276:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=30) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i*30) ])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]   
    
    elif freq == 6552:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=15) ])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i*15)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]    
            
    elif freq == 9828:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=10)])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i*10)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]  
            
    elif freq == 19656:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=5)])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i*5)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]    
            
    elif freq == 98280:
        X_pred_date = pd.DataFrame([max(X_pred.index) + relativedelta(minutes=1)])
        if n_leads >= 2:
            for i in range(2, n_leads+1):
                X_pred_date = X_pred_date.append([max(X_pred.index) + relativedelta(minutes=i)])    
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]             
        else:
            X_pred_date.columns = [date_colname] 
            X_pred_date.index = X_pred_date[date_colname]  
            
    return X_pred_date

class NaiveRandomWalk(BaseEstimator, RegressorMixin):
    def __init__(self, with_drift = False):
        super().__init__()
        
        self.with_drift = with_drift
        
        
    def fit(self, dataset, y_colname, date_colname, n_leads, freq, ci = 0.95):
        #self._dataset = dataset
        #self._y_colname = y_colname
        #self._date_colname = date_colname
        #self._n_leads = n_leads
        #self._freq = freq
        self._ci = ci
        
        dataset[date_colname] = pd.to_datetime(dataset[date_colname])
        dataset.index = pd.to_datetime(dataset[date_colname])
        dataset.index.name = 'index'
        
        start_date = np.min(get_pred_dates(freq, 
                                           dataset[y_colname], 
                                           date_colname, 
                                           n_leads))[0]
        end_date = np.max(get_pred_dates(freq, 
                                         dataset[y_colname], 
                                         date_colname, 
                                         n_leads))[0]
        
        print(start_date, end_date)
        #print(dataset)
        self._start_date = start_date
        self._end_date = end_date
        
        if self.with_drift == False:
            try:
                model = sm.tsa.statespace.SARIMAX(endog = dataset[y_colname], 
                                                      order=(0,1,0), 
                                                      trend = 'n',
                                                      seasonal_order = (0,0,0,0))
                model = model.fit(method = 'lbfgs') 
                arima_pred = model.get_prediction(start=start_date, 
                            end=end_date,
                            dynamic=False)
            except:
                arima_pred = model.get_prediction(len(dataset) + 1, len(dataset) + n_leads,
                            dynamic=False)
        else:
            try:
                model = sm.tsa.statespace.SARIMAX(endog = dataset[y_colname], 
                                                      order=(0,1,0), 
                                                      trend = 'c',
                                                      seasonal_order = (0,0,0,0))
                model = model.fit(method = 'lbfgs') 
                arima_pred = model.get_prediction(start=start_date, 
                            end=end_date,
                            dynamic=False)
            except:
                arima_pred = model.get_prediction(len(dataset) + 1, len(dataset) + n_leads,
                            dynamic=False)
        
        pred_ci2 = arima_pred.conf_int(alpha = 1-self._ci)
        y_forecasted = pd.DataFrame(arima_pred.predicted_mean).astype(float)

        arima_pred = pd.concat([y_forecasted, pred_ci2.astype(float)], axis = 1)
        arima_pred.columns = ['Y_PRED', 'Y_PRED_LOWER', 'Y_PRED_UPPER']
        
        arima_pred.index = get_pred_dates(freq, 
                                   dataset[y_colname], 
                                   date_colname, 
                                   n_leads)[date_colname]
        
        arima_pred.index = pd.to_datetime(arima_pred.index)
        
        self._pred = arima_pred.copy()

    def predict(self):

        return self._pred
    

#y_true = ts_test
#y_pred = holt_pred['Y_PRED']
class RdR_scorer:
    def __init__(self):
        return
    
    def fit(self, ts, y_colname, n_step, y_true, y_pred, model_name = 'MODEL', freq = 12):
        
        self._model_name = model_name
        self._y_true = y_true
        
        ts_orig = ts.copy()
        ts_orig['DATE'] = pd.to_datetime(ts_orig.index.values)
        #print(ts_orig)
        start_date = min(pd.to_datetime(y_true.index))
        end_date = max(pd.to_datetime(y_true.index))
        #print(start_date, end_date)
        y_true_1darray = y_true.iloc[:, 0].ravel()
        y_pred_1darray = y_pred.iloc[:, 0].ravel()
        
        RandomModel = NaiveRandomWalk(with_drift = False)
        RandomModel.fit(ts_orig, y_colname, 'DATE', n_step, freq, ci = 0.95)
        RandomWalkPred = RandomModel.predict()
        
        RandomWalkPred.columns = ['Y_PRED', 'Y_PRED_LOWER', 'Y_PRED_UPPER']
        RandomWalkPred.index = pd.to_datetime(RandomWalkPred.index)

        rmse_model = np.sqrt(mean_squared_error(y_true_1darray, y_pred_1darray))
        dtw_distance_model = slow_dtw_distance(y_true_1darray, y_pred_1darray)
        
        rmse_rw = np.sqrt(mean_squared_error(y_true, RandomWalkPred['Y_PRED']))
        self._random_walk = RandomWalkPred.copy()
        self._raw_rmse_rw = rmse_rw
        
        dtw_distance_rw = slow_dtw_distance(y_true_1darray, RandomWalkPred['Y_PRED'].values.ravel())
        
        rmse_scaler = MinMaxScaler(feature_range = (0,1))
        dtw_scaler = MinMaxScaler(feature_range = (0,1))
        
        df_scale = pd.DataFrame([[0, 0],[ rmse_rw, dtw_distance_rw]])
        df_scale.columns = ['RMSE','DTW']
        df_scale['RMSE_SCORE'] = 1 - rmse_scaler.fit_transform(df_scale[['RMSE']].values).ravel()
        df_scale['DTW_SCORE'] = 1 - dtw_scaler.fit_transform(df_scale[['DTW']].values).ravel()
        df_scale['RdR_SCORE'] = (df_scale['RMSE_SCORE'] + df_scale['DTW_SCORE']) / 2
        
        df = pd.DataFrame([[rmse_model, dtw_distance_model]])
        df.columns = ['RMSE','DTW']
        df['RMSE_SCORE'] = 1 - rmse_scaler.transform(df[['RMSE']].values).ravel()
        df['DTW_SCORE'] = 1 - dtw_scaler.transform(df[['DTW']].values).ravel()
        df['RdR_SCORE'] = (df['RMSE_SCORE'] + df['DTW_SCORE']) / 2
        
        df_all = pd.concat([df_scale, df], axis = 0, ignore_index = True).reset_index(drop = True)
        
        rmse_pourc = y_true.copy()
        rmse_pourc['RMSE_POURC'] = df['RMSE'].values / rmse_pourc[y_colname].values
        rmse_pourc['RMSE_ACCURACY'] = 1 - (df['RMSE'].values / rmse_pourc[y_colname].values)
        rmse_pourc['RMSE_ACCURACY'].clip(lower = 0)
        forecast_accuracy_mean = rmse_pourc['RMSE_ACCURACY'].mean()
        forecast_accuracy_min = rmse_pourc['RMSE_ACCURACY'].min()
        forecast_accuracy_max = rmse_pourc['RMSE_ACCURACY'].max()
        
        self._rmse = df['RMSE'].values[0]
        self._dtw = df['DTW'].values[0]
        self._rdr = df['RdR_SCORE'].values[0]
        
        self._rmse_rw = df_scale['RMSE'].values[0]
        self._dtw_rw = df_scale['DTW'].values[0]
        self._rdr_rw = df_scale['RdR_SCORE'].values[0]
        
        self._model_df = df[['DTW', 'RMSE', 'RdR_SCORE']]
        self._model_df['MODEL_NAME'] = str(model_name)
        self._rw_df = df_scale.loc[df_scale['RMSE'] > 0][['DTW', 'RMSE', 'RdR_SCORE']]
        self._rw_df['MODEL_NAME'] = 'RandomWalk'
        self._all_df = df_all[['DTW', 'RMSE']]
        
        self._pred_rw = RandomWalkPred
        
        self._fa_mean = forecast_accuracy_mean
        self._fa_min = forecast_accuracy_min
        self._fa_max = forecast_accuracy_max
        
        
        self._def_viz = pd.concat([self._model_df, self._rw_df], axis = 0, ignore_index = True).reset_index(drop = True)

    def rename_model(self, model_name):
        self._model_name = model_name
        self._model_df['MODEL_NAME'] = str(model_name)
        self._def_viz = pd.concat([self._model_df, self._rw_df], axis = 0, ignore_index = True).reset_index(drop = True)
        
    def score(self):
        return self._rdr
    
    def get_rmse_score(self):
        return self._rmse
    
    def get_dtw_score(self):
        return self._dtw
    
    def get_randomwalk_rmse_score(self):
        return self._rmse_rw
    
    def get_randomwalk_dtw_score(self):
        return self._dtw_rw
    
    def get_randomwalk_pred(self):
        return self._pred_rw
    
    def get_df_viz(self):
        return self._def_viz
    
    def add_rdr(self, rdr_object):
        self._def_viz = pd.concat([self._def_viz, rdr_object._def_viz], axis = 0, ignore_index = True).reset_index(drop = True)
        self._def_viz = self._def_viz.drop_duplicates()
        return self._def_viz
    
    def get_rdr_interpretation(self):
        if self._rdr > 0 and self._fa_mean >= 0.75:
            qual = 'GOOD PERFORMANCE'
        elif self._rdr > 0 and self._fa_mean < 0.75:
            qual = 'AVERAGE PERFORMANCE'
        elif self._rdr < 0:
            qual = 'BAD PEFORMANCE'
        else:
            pass
        
        if self._rdr > 0 :
            trust = 'better'
        elif self._rdr == 0:
            trust = '(equal)'
        else:
            trust = 'worst'
            
        texte = qual + ': ' + 'With a stable trend and no major unpredictable changes, the model is ' + str(round((self._rdr * 100), 2)) + '% ' + trust + ' than a naïve random decision. The mean forecast accuracy is ' + str(round(self._fa_mean * 100,2)) + '% (around ' + str(round(self._fa_min * 100,2)) + '% and ' + str(round(self._fa_max * 100,2)) + '% of accuracy per forecasted datapoint)'
        
        return texte
    
    def plot_rdr_rank(self, models = list([])):
        if len(models) == 0:
            models = self._def_viz
            
        models = models.append(pd.DataFrame([[0,0,1.0,'Perfect Score']], index = [max(models.index)+1], columns = models.columns))
        
        model_ranking = models.sort_values(by = 'RdR_SCORE', ascending = True)
        model_ranking['RdR_SCORE'] = model_ranking['RdR_SCORE'] * 100
        #model_ranking = model_ranking.loc[~model_ranking['MODEL_NAME'].isin(['PERFECT_SCORE', 'WORST_SCORE'])]
         
        colors = []
        for value in model_ranking.loc[:, 'RdR_SCORE']: # keys are the names of the boys
            if value < 0:
                colors.append('r')
            elif value <= np.max(model_ranking.loc[model_ranking['MODEL_NAME'] == 'RandomWalk']['RdR_SCORE']):
                colors.append('y')
            else:
                colors.append('g')
        
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 10, forward=True)
        ax.barh(model_ranking['MODEL_NAME'], model_ranking['RdR_SCORE'], color = colors, alpha = 0.65)
        ax.tick_params(axis="y", labelsize=10)
        #ax.set_xticks(rotation = 60, fontsize = 8)
        #for i in enumerate('RMSE:' + model_ranking['LB_RMSE'].round(4).astype(str)):
        #    plt.text(i + 0, str(v), fontweight='bold')
        # find the values and append to list
        totals = []
        for i in ax.patches:
            totals.append(i.get_width())
        # set individual bar lables using above list
        #total = sum(totals)
        for i in ax.patches:
            # get_width pulls left or right; get_y pushes up or down
            ax.text(i.get_width(), i.get_y()+.38, \
                    str(round(i.get_width(), 2)) + '%', fontsize=12,
        color='black',weight="bold"
        )
        plt.title('Model Ranking based on RdR score' + '\n' 
                  + 'Best possible model = 100%' + '\n' 
                  + '0% = Naïve RandomWalk' + '\n' 
                  + '< 0% = Worst than Naïve RandomWalk (Red)' + '\n'
                  + '> 0% = Better than Naïve RandomWalk (Green)')
        plt.tight_layout()
    
    def plot_rdr(self, models = list([]), 
                 scatter_size = 80, 
                 scatter_label = True, 
                 scatter_label_size = 6, 
                 scatter_alpha = 0.65,
                 figsize = (10, 10)):
        if len(models) == 0:
            models = self._def_viz
        #############################################################################################
        ################### PLOT MODEL VALIDATION GRAPH GRID ########################################
        #############################################################################################
        models = models.append(pd.DataFrame([[0,0,1.0,'Perfect Score']], index = [max(models.index)+1], columns = models.columns))
        
        import matplotlib.patches as mpatches
        #rectangle = [(0,0),(0,1),(1,1),(1,0)]
        fig1 = plt.figure(figsize = figsize)
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.add_patch(mpatches.Rectangle((0, 0), self._rw_df['DTW'].values[0], self._rw_df['RMSE'].values[0], alpha = 0.1, color = 'green', linestyle = '--', linewidth = 1.5, edgecolor = 'grey'))
        ax2 = fig1.add_subplot(111, aspect='equal')
        ax2.add_patch(mpatches.Rectangle((0, self._rw_df['RMSE'].values[0]), np.max(self._def_viz['DTW']) * 1.15, np.max(self._def_viz['RMSE']) * 1.15, alpha = 0.07, color = 'red', edgecolor = None, linewidth = 0))
        ax3 = fig1.add_subplot(111, aspect='equal')
        ax3.add_patch(mpatches.Rectangle((self._rw_df['DTW'].values[0], 0), np.max(self._def_viz['DTW']) * 1.15, self._rw_df['RMSE'].values[0], alpha = 0.07, color = 'red', edgecolor = None, linewidth = 0))
        plt.scatter(self._rw_df['DTW'], self._rw_df['RMSE'],s = scatter_size, label = 'Naïve Random Walk Score', color = 'red')
        
        for i in range(0, len(models)):
            model = pd.DataFrame(models.iloc[i:i+1, :])
            #print(model)
            #print(model.columns.tolist())
            if ((model['DTW'].values[0] >= self._rw_df['DTW'].values[0]) or (model['RMSE'].values[0] >= self._rw_df['RMSE'].values[0])) == True:
                plt.scatter(model['DTW'], model['RMSE'], color = 'red', s = scatter_size, label = model['MODEL_NAME'], alpha = scatter_alpha)
            else:
                plt.scatter(model['DTW'], model['RMSE'], color = 'green', s = scatter_size, label = model['MODEL_NAME'], alpha = scatter_alpha)
            if scatter_label == True:
                if len(model['MODEL_NAME'][i]) > 15:
                    model_name = model['MODEL_NAME'][i][:15]+str('[...]')
                else:
                    model_name = model['MODEL_NAME'][i]
                plt.annotate(model_name, (model['DTW'][i], model['RMSE'][i]), fontsize = scatter_label_size)
            
        plt.ylim(0, np.max(self._def_viz['RMSE'])  * 1.15)
        plt.xlim(0, np.max(self._def_viz['DTW']) * 1.15)
        plt.xlabel('Dynamic Time Warping Distance' + '\n' + '(Metric for time series shape similarity, prediction vs test set)', fontsize = 15)
        plt.ylabel('RMSE score' + '\n' + '(Metric for penalized prediction errors on test set)', fontsize = 10)
        plt.title('MODELS A PERFORMANCE COMPARISON for multistep forecasting in the future'+ '\n' + 'TIME SERIE' + '\n' + 'RED ZONES: Not better than Naïve model'+ '\n' + 'GREEN ZONE: Better than Naïve model')
        #plt.legend(loc='upper left', fontsize = 15)
        plt.show()
		#print('TEST')
        #plt.savefig('all_best_preds_test.png', dpi = 800)
        #plt.tight_layout()
