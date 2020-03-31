import pandas as pd
import numpy as np
from dtaidistance import dtw
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

#y_true = ts_test
#y_pred = holt_pred['Y_PRED']
class RdR_scorer:
    def __init__(self):
        return
    
    def fit(self, ts, y_colname, n_step, y_true, y_pred, model_name = 'MODEL', freq = 12):
        
        start_date = min(y_true.index)
        end_date = max(y_true.index)
        
        model1 = sm.tsa.statespace.SARIMAX(ts[y_colname],
                                           order=(0,1,0),
                                           seasonal_order=(0, 0, 0, freq),
                                           enforce_stationarity=False,
                                           enforce_invertibility=False
                                           #exog = exog
                                           #trend='c'
                                           #,time_varying_regression = True
                                           )
        model1 = model1.fit(method = 'lbfgs') 
        
        RandomWalkPred = model1.get_prediction(start=start_date, 
                                               end=end_date,
                                               dynamic=False)
        pred_ci2 = RandomWalkPred.conf_int()
        y_forecasted = pd.DataFrame(RandomWalkPred.predicted_mean).astype(float)
        RandomWalkPred = pd.concat([y_forecasted, pred_ci2.astype(float)], axis = 1)
        RandomWalkPred.columns = ['Y_PRED', 'Y_PRED_LOWER', 'Y_PRED_UPPER']
        RandomWalkPred.index = pd.to_datetime(RandomWalkPred.index)

        rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
        dtw_distance_model = dtw.distance(y_true.values, y_pred.values)
        
        rmse_rw = np.sqrt(mean_squared_error(y_true, RandomWalkPred['Y_PRED']))
        dtw_distance_rw = dtw.distance(y_true.values, RandomWalkPred['Y_PRED'].values)
        
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
    
    def plot_rdr(self, models = list([])):
        if len(models) == 0:
            models = self._def_viz
        #############################################################################################
        ################### PLOT MODEL VALIDATION GRAPH GRID ########################################
        #############################################################################################
        models = models.append(pd.DataFrame([[0,0,1.0,'Perfect Score']], index = [max(models.index)+1], columns = models.columns))
        
        import matplotlib.patches as mpatches
        #rectangle = [(0,0),(0,1),(1,1),(1,0)]
        fig1 = plt.figure(figsize = (18, 9))
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.add_patch(mpatches.Rectangle((0, 0), self._rw_df['DTW'].values[0], self._rw_df['RMSE'].values[0], alpha = 0.1, color = 'green', linestyle = '--', linewidth = 1.5, edgecolor = 'grey'))
        ax2 = fig1.add_subplot(111, aspect='equal')
        ax2.add_patch(mpatches.Rectangle((0, self._rw_df['RMSE'].values[0]), np.max(self._def_viz['DTW']) * 1.15, np.max(self._def_viz['RMSE']) * 1.15, alpha = 0.07, color = 'red', edgecolor = None, linewidth = 0))
        ax3 = fig1.add_subplot(111, aspect='equal')
        ax3.add_patch(mpatches.Rectangle((self._rw_df['DTW'].values[0], 0), np.max(self._def_viz['DTW']) * 1.15, self._rw_df['RMSE'].values[0], alpha = 0.07, color = 'red', edgecolor = None, linewidth = 0))
        plt.scatter(self._rw_df['DTW'], self._rw_df['RMSE'],s = 80, label = 'Naïve Random Walk Score', color = 'red')
        
        for i in range(0, len(models)):
            model = pd.DataFrame(models.iloc[i:i+1, :])
            #print(model)
            #print(model.columns.tolist())
            if ((model['DTW'].values[0] >= self._rw_df['DTW'].values[0]) or (model['RMSE'].values[0] >= self._rw_df['RMSE'].values[0])) == True:
                plt.scatter(model['DTW'], model['RMSE'], color = 'red', s = 80, label = model['MODEL_NAME'])
            else:
                plt.scatter(model['DTW'], model['RMSE'], color = 'green', s = 80, label = model['MODEL_NAME'])
            plt.annotate(model['MODEL_NAME'][i], (model['DTW'][i], model['RMSE'][i]))
            
        plt.ylim(-0.1, np.max(self._def_viz['RMSE']) * 1.15)
        plt.xlim(-0.1, np.max(self._def_viz['DTW']) * 1.15)
        plt.xlabel('Dynamic Time Warping Distance' + '\n' + '(Metric for time series shape similarity, prediction vs test set)', fontsize = 15)
        plt.ylabel('RMSE score' + '\n' + '(Metric for penalized prediction errors on test set)', fontsize = 15)
        plt.title('MODELS PERFORMANCE COMPARISON for multistep forecasting in the future'+ '\n' + 'TIME SERIE' + '\n' + 'RED ZONES: Not better than Naïve model'+ '\n' + 'GREEN ZONE: Better than Naïve model')
        #plt.legend(loc='upper left', fontsize = 15)
        plt.show()
        #plt.savefig('all_best_preds_test.png', dpi = 800)
        plt.tight_layout()
