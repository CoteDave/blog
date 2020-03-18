import numpy as np
import pandas as pd
from pyod.models.knn import KNN 
from pyod.models.iforest import IForest 
from pyod.models.ocsvm import OCSVM

import multiprocessing
n_cpu = multiprocessing.cpu_count()

import sys
import category_encoders as encoders

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from collections import defaultdict

from rulefit import RuleFit

from xgboost import XGBRegressor, XGBClassifier

def get_numerical_cols(dataframe):
        colnames_numerical = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        return colnames_numerical
    
def get_categorical_cols(dataframe):
    colnames_categorical = dataframe.select_dtypes(include='object').columns.tolist()
    return colnames_categorical

def duplicate_columns(dataframe):
    groups = dataframe.columns.to_series().groupby(dataframe.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = dataframe[v].columns
        vs = dataframe[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if pd.DataFrame(ia).equals(pd.DataFrame(ja)):
                    dups.append(cs[i])
                    break
    return dups

def outlier_std(s, nstd=3.0, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using standard deviation, works column-wise.
    param nstd:
        Set number of standard deviations from the mean
        to consider an outlier
    :type nstd: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]

def outlier_iqr(s, k=1.5, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using interquartile range, works column-wise.
    param k:
        some cutoff to multiply by the iqr
    :type k: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    # calculate interquartile range
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]

def replace_nulls(dataframe):
        
    colnames_numerical = get_numerical_cols(dataframe)
    colnames_categorical = get_categorical_cols(dataframe)
    
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    for col in colnames_numerical:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mean())

    for col in colnames_categorical:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
        
    return dataframe

def drop_duplicated_columns(dataframe):
    dups = duplicate_columns(dataframe)
    dataframe = dataframe.drop(dups, axis=1)
    return dataframe

def drop_constant_columns(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    keep_columns = dataframe.columns[dataframe.nunique()>1]
    return dataframe.loc[:,keep_columns].copy()









def concatenate_list_data(liste):
    result= ''
    i = 0
    for element in liste:
        if i == len(liste)-1:
            result += str(element)
        else:
            result += str(element) + ' & '
        i = i +1
    return result


def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)
       
def round_decimal_string(string):
 
    position_p_list = [pos for pos, char in enumerate(string) if char == '.']
    i = 1
    for i in range(1, len(position_p_list)+1):   
        next_p_position = find_nth(string, '.', i)
        string_concat1 = string[0:next_p_position+3]
        next_space_position = string[next_p_position+3:].find(' ')
        if next_space_position <=2:
            string_concat2 = string[next_p_position+4:]
        if next_space_position == -1:
            string_concat2 = string[next_p_position+3:next_p_position+3]
        else:
            string_concat2 = string[next_p_position+2+next_space_position:]
        string = string_concat1 + string_concat2
        string = string.replace('  ', ' ')
    return string



def rule_to_feature(dataset, text_rule = ''):
    data = dataset.copy()
    dataset_variable_name = 'data'
    text_rule = text_rule.replace(' = ', ' == ')
    position_and_list = [pos for pos, char in enumerate(text_rule) if char == '&']
    position_h_list = [pos for pos, char in enumerate(text_rule) if char == '>']
    position_he_list = [pos for pos, char in enumerate(text_rule) if char == '>=']
    position_l_list = [pos for pos, char in enumerate(text_rule) if char == '<']
    position_le_list = [pos for pos, char in enumerate(text_rule) if char == '<=']
    position_e_list = [pos for pos, char in enumerate(text_rule) if char == '==']
    
    all_positions = position_and_list + position_h_list + position_he_list + position_l_list + position_le_list + position_e_list
    all_positions = sorted(all_positions)
    
    #feature_list = " ".join(re.split("[^a-zA-Z, _]*", text_rule)).split() 
    
    
    all_eq_positions = position_h_list + position_he_list + position_l_list + position_le_list + position_e_list
    all_eq_positions = sorted(all_eq_positions)

    max_var = len(position_and_list)
    max_eq = len(all_eq_positions)
    var_list = []
    
    string_list = []
    
    #i = 0
    for i in range(0, len(position_and_list)+1):
        #print(i)
        if i == 0 and len(position_and_list) > 0:
            string_list.append(text_rule[0:position_and_list[i]])
        elif i == 0 and len(position_and_list) == 0:
            string_list.append(text_rule)
        elif i == len(position_and_list):
            string_list.append(text_rule[position_and_list[i-1]:])
        else:
            string_list.append(text_rule[position_and_list[i-1]:position_and_list[i]])
    
    string_list_feature = [w.replace('& ', '') for w in string_list]
    feature_list = [w[:find_nth(w, ' ', 1)] for w in string_list_feature]
    
    i = 0
    for string_var in string_list:
        if i == 0 and len(position_and_list) > 0:
            rule = '((' + dataset_variable_name +"['" + feature_list[i] + "']" + ' ' + text_rule[all_eq_positions[0]:position_and_list[i]] + ')' 
        elif i == 0 and len(position_and_list) == 0:
            rule = '((' + dataset_variable_name +"['" + feature_list[i] + "']" + ' ' + text_rule[all_eq_positions[0]:] + '))'
        elif i == len(string_list) - 1:
            rule = '(' + dataset_variable_name +"['" + feature_list[i] + "']" + ' ' + text_rule[all_eq_positions[i]:] + '))' 
        else:
            rule = '(' + dataset_variable_name +"['" + feature_list[i] + "']" + ' ' + text_rule[all_eq_positions[i]:all_eq_positions[i+1]] + ')'
            if find_nth(rule, '&', 1) > -1:
                rule = rule[:find_nth(rule, '&', 1)]
                rule = rule + ')'
                
        var_list.append(rule) 
        i = i +1

    statement = concatenate_list_data(var_list)
    #print(statement)
    #print(feature_list)
    #eval("dataset['bmi']") 
    return np.where(eval(statement), 1, 0), statement

def rules_feature_engineering(data_X,
                              data_y,
                              feature_names,
                              #rulefit_model,
                              n_best_rules,
                              base_model, 
                              random_state,
                              max_rules):
    ###### RULEFIT ENSEMBLE Model
    model = RuleFit(tree_generator= base_model,
                    random_state= random_state, 
                    max_rules= max_rules)
    
    try:
        model.fit(data_X.values, data_y.values, feature_names=feature_names.copy())
    except:
        model.fit(data_X, data_y, feature_names=feature_names.copy())
        
    ##### MODEL INTERPRETATION - RULES !
    rules = model.get_rules()
    rules = rules[rules.coef != 0].sort_values(by="support")
    num_rules_rule=len(rules[rules.type=='rule'])
    num_rules_linear=len(rules[rules.type=='linear'])
    
    rules['rule'] = rules['rule'].apply(round_decimal_string)
    linear_rules = rules.loc[rules['type'] == 'linear']
    rules = rules.loc[rules['type'] != 'linear']
    rules.index = rules['rule'].str.wrap(105)
    #rules.index = rules['rule'].str.wrap(40)
    rules_copy = rules.copy()
    rules = rules.sort_values(['importance'], ascending = False)
    rules['rank'] = rules['importance'].rank(ascending=False)
    n_rules = int(max(rules['rank']))
    #rules = rules.loc[rules['rank'] <= 25]
    
    if len(rules) < n_best_rules:
        n_best_rules = len(rules)
    
    rules_X = data_X.copy()
    selected_rules = list(rules.sort_values(by = ['rank'])['rule'])[0:n_best_rules]
    #i = 0
    selected_rules2 = pd.DataFrame(columns=['FEATURE_NAME','RULE', 'PYTHON_CODE'])
    for i in range(0, n_best_rules):
        #print(i)
        #print(list(rules.sort_values(by = ['rank'])['rule'])[i])
        #print(list(X.columns))
        text_rule= list(rules.sort_values(by = ['rank'])['rule'])[i]
        text_rule_cleaned = text_rule.replace(' ', '_').replace(' ', '_').replace(' ', '_').replace('&','AND').replace('.','_')
        text_rule_cleaned = text_rule_cleaned.replace('_<_','_LOWER_')
        text_rule_cleaned = text_rule_cleaned.replace('_>_','_GREATER_')
        text_rule_cleaned = text_rule_cleaned.replace('_<=_','_LOWER_EQUAL_')
        text_rule_cleaned = text_rule_cleaned.replace('_>=_','_GREATER_EQUAL_')
        text_rule_cleaned = text_rule_cleaned.replace('_==_','_EQUAL_')
        rules_X['RULE_EXTRACT_' + str(i+1)], statement = rule_to_feature(dataset = data_X.copy(), text_rule = text_rule)
        df_temp = pd.DataFrame([['RULE_EXTRACT_' + str(i+1), selected_rules[i], statement]], columns=['FEATURE_NAME','RULE', 'PYTHON_CODE'])
        selected_rules2 = selected_rules2.append(df_temp, ignore_index = True)
        #rules_X[text_rule_cleaned+ '_R' + str(i+1)] = rule_to_feature(dataset = data_X.copy(), text_rule = text_rule)
       
    return rules_X, selected_rules2

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import shap

def feature_selection(X_train, y_train):
    
    i = 0
    for col in y_train.columns:
        xgb_model = XGBRegressor(n_estimators=200, 
                                learning_rate=0.03, 
                                subsample=0.61,  
                                max_depth=9, random_state=0).fit(X_train, y_train[col])
        features_train = pd.DataFrame(list(X_train.columns), columns = ['FEATURE'])
        features_train['index'] = features_train.index
        df_allfeatures = pd.DataFrame(X_train.columns, columns = ['FEATURE'])
        df_allfeatures.index = df_allfeatures['FEATURE']
        df_fscore_xgb_list = pd.DataFrame(xgb_model.get_booster().get_score(importance_type= 'gain'), index=[0]).T.sort_values(by = (0), ascending = False)
        df_fscore_xgb_list.columns = ['FSCORE_GAIN_XGB']
        df_fscore_xgb_list = pd.merge(df_allfeatures, df_fscore_xgb_list, how = 'left', left_index = True, right_index = True)
        df_fscore_xgb_list = df_fscore_xgb_list.fillna(0)
        df_fscore_xgb_list = df_fscore_xgb_list.sort_values(by = (['FSCORE_GAIN_XGB']), ascending = False)
        df_fscore_xgb_list['Y_TRUE'] = col
        shap_explainer = shap.TreeExplainer(xgb_model)
        
        if i == 0:
            df_fscore_xgb = df_fscore_xgb_list.copy()
        else:
            df_fscore_xgb = df_fscore_xgb.append(df_fscore_xgb_list, ignore_index = True)
            
        if i == 0:
            global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
            global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
            global_shap_values_train.columns = ['SHAP_MEAN']
            global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
            global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
            del global_shap_values_train['index']
            global_shap_values_train['Y_VALUE'] = col
      
            global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
            global_shap_values_train.index = global_shap_values_train['FEATURE']
            df_shap_mean = global_shap_values_train.copy()
            
        else:
            global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
            global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
            global_shap_values_train.columns = ['SHAP_MEAN']
            global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
            global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
            del global_shap_values_train['index']
            global_shap_values_train['Y_VALUE'] = col
      
            global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
            global_shap_values_train.index = global_shap_values_train['FEATURE']
            df_shap_mean = df_shap_mean.append(global_shap_values_train, ignore_index = True)
        i = i+1
        #print(col)
           
    df_shap_mean['SHAP_MEAN_ABS'] = abs(df_shap_mean['SHAP_MEAN'])
    df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] = abs(df_fscore_xgb['FSCORE_GAIN_XGB'])
    df_shap_mean.index.name = 'index'
    df_shap_mean2 = df_shap_mean[['SHAP_MEAN_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
    df_fscore_xgb.index.name = 'index'
    df_fscore_xgb2 = df_fscore_xgb[['FSCORE_GAIN_XGB_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
    
    df_shap_mean = pd.DataFrame(df_shap_mean2.sort_values(by = ['SHAP_MEAN_ABS'], ascending = False))
    df_shap_mean = df_shap_mean.loc[df_shap_mean['SHAP_MEAN_ABS'] >= df_shap_mean['SHAP_MEAN_ABS'].quantile(0.45)]
    df_fscore_xgb = pd.DataFrame(df_fscore_xgb2.sort_values(by = ['FSCORE_GAIN_XGB_ABS'], ascending = False))
    df_fscore_xgb = df_fscore_xgb.loc[df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] >= df_fscore_xgb['FSCORE_GAIN_XGB_ABS'].quantile(0.45)]
    
    
    #Multicolinearity
    corr = pd.DataFrame(spearmanr(X_train).correlation).fillna(0)
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    
    #selected_features = X_train.columns.to_list()
    for i in range(0, len(y_train.columns)):
        #print(str(i+1) + ' / ' + str(len(y_train.columns)) + '...')
        model = Lasso(alpha=0.001, max_iter=25000)
        model.fit(X_train.iloc[:, selected_features], y_train.iloc[:, i])
        
        result = permutation_importance(model, X_train.iloc[:, selected_features], y_train.iloc[:, i], n_repeats=8,
                                        random_state=0)
        perm_sorted_idx = result.importances_mean.argsort()
        perm_df = pd.DataFrame(result.importances[perm_sorted_idx])
        perm_df['FEATURE'] = X_train.iloc[:, selected_features].iloc[:, perm_sorted_idx].columns.values
        
        if i == 0:
            perm_df_score = perm_df[['FEATURE']]
            perm_df_score['PERM_SCORE'] = perm_df.iloc[:, 2]
        else:
            perm_df_score2 = perm_df[['FEATURE']]
            perm_df_score2['PERM_SCORE'] = perm_df.iloc[:, 2]
            
            perm_df_score = pd.concat([perm_df_score, perm_df_score2], axis = 0, ignore_index = True)
    
    perm_df_score = pd.DataFrame(perm_df_score.groupby(['FEATURE']).mean().reset_index() ).sort_values(by = ['PERM_SCORE'], ascending = False)
    perm_df_score = perm_df_score.loc[perm_df_score['PERM_SCORE'] >= perm_df_score['PERM_SCORE'].quantile(0.45)]
    
    all_features = pd.DataFrame(X_train.columns)
    all_features.columns = ['FEATURE']
    
    all_features = pd.merge(all_features, perm_df_score, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
    all_features = pd.merge(all_features, df_shap_mean, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
    all_features = pd.merge(all_features, df_fscore_xgb, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])

    scaler = MinMaxScaler(feature_range=(0,100))
    all_features_scaled = pd.DataFrame(scaler.fit_transform(all_features.iloc[:, 1:]), columns = all_features.iloc[:, 1:].columns, index = all_features.index)
    all_features_scaled['FEATURE'] = all_features['FEATURE']
    all_features_mean = pd.DataFrame(np.mean(all_features_scaled, axis = 1))
    all_features_mean.columns = ['SCORE_MEAN']
    all_features_mean['FEATURE'] = all_features['FEATURE']
    all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] > 0]
    all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] >= all_features_mean['SCORE_MEAN'].quantile(0.2)]
    all_features_mean = all_features_mean.dropna().sort_values(by = ['SCORE_MEAN'], ascending = False)
    
    X_train = X_train.loc[:, all_features_mean['FEATURE']]
    
    corr = pd.DataFrame(spearmanr(X_train).correlation).fillna(0)
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    
    X_train = X_train.iloc[:, selected_features]
    
    return X_train.columns.to_list()






class DataPreparator:
    
    def __init__(self, replace_null = True,
            drop_duplicate_columns = True,
            drop_duplicate_rows = True,
            drop_constant_columns = True,
            drop_outliers = True,
            categorical_encoding = True,
            
            encoding_strategy = 'dummy',
            outliers_strategy = 'IQR',
            outliers_cutoff = 3,
            outliers_contamination = 0.01,
            copy=True):
        
        self.copy = copy
        self.transformed_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.anomaly_ = None
        self.X_ = None
        self.replace_null_ = replace_null
        self.categorical_encoding_ = categorical_encoding
        self.encoding_strategy_ = encoding_strategy
        
        self.drop_duplicate_columns_ = drop_duplicate_columns
        self.drop_duplicate_rows_  = drop_duplicate_rows
        self.drop_constant_columns_  = drop_constant_columns
        self.drop_outliers_  = drop_outliers
        self.outliers_strategy_  = outliers_strategy
        self.outliers_cutoff_  = outliers_cutoff
        self.outliers_contamination_  = outliers_contamination
        
        return
        
    
    def categorical_encoding(self, X_train, y_train,
                             strategy = 'dummy' #['dummy', 'catboost', 'target']
                             ):
        
        colnames_numerical = get_numerical_cols(X_train)
        colnames_categorical = get_categorical_cols(X_train)
        #y_colname = 'charges'
        
        if strategy == 'dummy':
            X_train = pd.get_dummies(X_train, 
                                     prefix_sep='_', 
                                     columns=colnames_categorical, 
                                     drop_first=True)
        elif strategy == 'catboost':
            encoder = encoders.CatBoostEncoder()
            encoded_cat = pd.DataFrame(encoder.fit_transform(X_train[colnames_categorical], y_train))
            encoded_cat.columns = list(colnames_categorical)
            
            #encoded_cat_test = pd.DataFrame(encoder.transform(dataset[colnames_categorical]))
            #encoded_cat_test.columns = list(colnames_categorical)
            
            self.encoder_ = encoder
            
            for col in encoded_cat.columns.to_list():
                X_train[col] = encoded_cat[col]
            
        elif strategy == 'target':    
            encoder = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
            encoded_cat = pd.DataFrame(encoder.fit_transform(X_train[colnames_categorical], y_train))
            encoded_cat.columns = list(colnames_categorical)
            
            #encoded_cat_test = pd.DataFrame(encoder.transform(dataset[colnames_categorical]))
            #encoded_cat_test.columns = list(colnames_categorical)
            
            self.encoder_ = encoder
            
            for col in encoded_cat.columns.to_list():
                X_train[col] = encoded_cat[col]
            
        elif strategy == 'None':  
            return X_train
        
        else:
            print("ERROR: Encoding strategy does not exists. Please select one of ['None','dummy', 'catboost', 'target']")
            sys.exit() 
            
        return X_train
            
    def drop_outliers(self, X_train, y_train,
                      strategy = 'IQR', #['STD', 'IQR', 'IsolationForest', 'OneClassSVM', 'KNN']
                      cutoff = 3,
                      contamination = 0.01):
        
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        dataframe = pd.concat([X_train, y_train], axis = 1)
        colnames_numerical = get_numerical_cols(dataframe)
        bool_cols = [col for col in dataframe if np.isin(dataframe[col].dropna().unique(), [0, 1]).all()]
        colnames_numerical = list(set(colnames_numerical) - set(bool_cols))
        
        if strategy == 'IQR':
            iqr = dataframe[colnames_numerical].apply(outlier_iqr, k=cutoff)
            cols_to_remove = iqr.loc[:, iqr.any()].columns.to_list()
            for col in cols_to_remove:
                dataframe = dataframe.loc[iqr[col] == False]
            
        elif strategy == 'STD':
            std = dataframe[colnames_numerical].apply(outlier_std,nstd=cutoff)
            cols_to_remove = std.loc[:, std.any()].columns.to_list()
            for col in cols_to_remove:
                dataframe = dataframe.loc[std[col] == False]   
                
        elif strategy == 'IsolationForest':
            model = IForest(n_estimators=100, 
                            max_samples='auto', 
                            contamination=contamination, 
                            max_features=1.0, 
                            bootstrap=False, 
                            n_jobs=n_cpu, 
                            random_state=0, 
                            verbose=0)
            model.fit(dataframe)
            outliers = pd.DataFrame(model.labels_)
            dataframe = dataframe.iloc[outliers.loc[outliers[0] == 0].index.values, :]
            
            self.anomaly_ = model
            
        elif strategy == 'OneClassSVM':
            model = OCSVM(contamination=contamination,
                          kernel = 'rbf')
            model.fit(dataframe)
            outliers = pd.DataFrame(model.labels_)
            dataframe = dataframe.iloc[outliers.loc[outliers[0] == 0].index.values, :]
            
            self.anomaly_ = model
            
        elif strategy == 'KNN':
            model = KNN(contamination=contamination,
                            n_jobs=n_cpu)
            model.fit(dataframe)
            outliers = pd.DataFrame(model.labels_)
            dataframe = dataframe.iloc[outliers.loc[outliers[0] == 0].index.values, :]
            
            self.anomaly_ = model
            
        elif strategy == 'None':
            return  dataframe  
        else:
            print("ERROR: Outlier strategy does not exists. Please select one of ['None', 'STD', 'IQR', 'IsolationForest', 'OneClassSVM', 'KNN']")
            sys.exit() 
        
        X_train = dataframe[X_train.columns.to_list()]
        return X_train
      
    def fit_transform(self, X_train, y_train):

        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        
        y_colname = y_train.columns.to_list()
        
        dataframe = pd.concat([X_train, y_train], axis = 1)
        
        if self.replace_null_ == True:
            dataframe = replace_nulls(dataframe)
        
        if self.drop_constant_columns_ == True:
            dataframe = drop_constant_columns(dataframe)
            
        if self.drop_duplicate_columns_ == True:
            dataframe = drop_duplicated_columns(dataframe)
        
        X_train = dataframe.loc[:, dataframe.columns.difference(y_colname).to_list()]
        y_train = dataframe[y_train.columns.to_list()]
        
        if self.drop_duplicate_rows_ == True:
            X_train = X_train.drop_duplicates()
            y_train = y_train.loc[y_train.index.isin(X_train.index.values)]
        
        if self.categorical_encoding_ == True:
            X_train = self.categorical_encoding(X_train, y_train,
                                                 strategy = self.encoding_strategy_
                                                 )
        
        
        uint_cols = X_train.select_dtypes(include='uint8').columns.tolist()
        
        for col in uint_cols:
            X_train[col] = X_train[col].astype(int)
        
        
        if self.drop_outliers_ == True:
            X_train = self.drop_outliers(X_train, y_train,
                                        strategy = self.outliers_strategy_,
                                        cutoff = self.outliers_cutoff_,
                                        contamination = self.outliers_contamination_)
            
            y_train = y_train.loc[y_train.index.isin(X_train.index.values)]
        
        
        self.colnames_numerical_ = get_numerical_cols(X_train)
        self.colnames_categorical_ = get_categorical_cols(X_train)
        self.all_columns_ = X_train.columns.to_list()
            
        return X_train, y_train
    
    def transform(self, X_test, y_test):   
        
        colnames_categorical = get_categorical_cols(X_test)
        
        if self.replace_null_ == True:
            X_test = replace_nulls(X_test)
        else:
            pass
        
        if self.categorical_encoding_ == True:
            if self.encoding_strategy_ == 'None':
                return X_test
            elif self.encoding_strategy_ == 'dummy':
                X_test = pd.get_dummies(X_test, 
                                     prefix_sep='_', 
                                     columns=colnames_categorical, 
                                     drop_first=True)
                uint_cols = X_test.select_dtypes(include='uint8').columns.tolist()
        
                for col in uint_cols:
                    X_test[col] = X_test[col].astype(int)
            
            elif self.encoder_ != None:
                encoded = pd.DataFrame(self.encoder_.transform(X_test[colnames_categorical]))
            #encoded_cat_test.columns = list(colnames_categorical)
                for col in encoded.columns.to_list():
                    X_test[col] = encoded[col]
                    
            else:
                pass
            
            
        
        else:
            return X_test, y_test
        
        X_test = X_test.loc[:, X_test.columns.isin(self.all_columns_)]
        
        to_delete_cols = X_test.loc[:, ~X_test.columns.isin(self.all_columns_)].columns.to_list()
        
        if len(to_delete_cols) > 0:
            for col in to_delete_cols:
                del X_test[col]
                
        return X_test



class FeatureSelectorRegressor:
    
    def __init__(self, selection_strategy = 'light', #['light', 'severe', 'shap', 'permutation']
                 quantile_cutoff = 0.45,
                 alpha = 0.001,
                 scaler = 'Standard', #['None', MinMax, Robust]
                 copy=True):
        self.copy = copy
        self.transformed_ = None
        #self.scaler_ = None
        self.X_ = None
        
        self.selection_strategy_ = selection_strategy
        self.quantile_cutoff_ = quantile_cutoff
        self.alpha_ = alpha
        self.scaler_ = scaler
        
    def fit_transform(self, X_train, y_train):

        X_cols = X_train.columns.tolist()
        X_index = X_train.index
        
        if self.scaler_ == 'Standard':   
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        elif self.scaler_ == 'MinMax':
            scaler = MinMaxScaler(feature_range=(0,100))
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        elif self.scaler_ == 'Robust':
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        else:
            X_train_scaled = X_train.copy()
            
        if self.selection_strategy_ == 'lasso':
            
            model = Lasso(alpha = self.alpha_) #self.alpha_)#89
            model.fit(X_train_scaled, y_train)

            X_weights = pd.DataFrame(model.coef_)
            X_weights = X_weights.T
            X_weights.columns = X_train_scaled.columns
            X_weights = X_weights.T
            X_weights.columns = ['FEATURE_WEIGHT']
            X_weights['FEATURE_NAME'] = X_weights.index
            X_weights.groupby(['FEATURE_NAME']).sum().sort_values(['FEATURE_WEIGHT'])#.plot.barh()
            #plt.tight_layout()
           # model.intercept_ 
            selected_features = X_weights.loc[abs(X_weights['FEATURE_WEIGHT']) > 0]['FEATURE_NAME'].tolist()

        
        elif self.selection_strategy_  == 'severe':
            i = 0
            for col in y_train.columns:
                xgb_model = XGBRegressor(n_estimators=200, 
                                        learning_rate=0.03, 
                                        subsample=0.61,  
                                        max_depth=9, random_state=0).fit(X_train, y_train[col])
                features_train = pd.DataFrame(list(X_train.columns), columns = ['FEATURE'])
                features_train['index'] = features_train.index
                df_allfeatures = pd.DataFrame(X_train.columns, columns = ['FEATURE'])
                df_allfeatures.index = df_allfeatures['FEATURE']
                df_fscore_xgb_list = pd.DataFrame(xgb_model.get_booster().get_score(importance_type= 'gain'), index=[0]).T.sort_values(by = (0), ascending = False)
                df_fscore_xgb_list.columns = ['FSCORE_GAIN_XGB']
                df_fscore_xgb_list = pd.merge(df_allfeatures, df_fscore_xgb_list, how = 'left', left_index = True, right_index = True)
                df_fscore_xgb_list = df_fscore_xgb_list.fillna(0)
                df_fscore_xgb_list = df_fscore_xgb_list.sort_values(by = (['FSCORE_GAIN_XGB']), ascending = False)
                df_fscore_xgb_list['Y_TRUE'] = col
                shap_explainer = shap.TreeExplainer(xgb_model)
                
                if i == 0:
                    df_fscore_xgb = df_fscore_xgb_list.copy()
                else:
                    df_fscore_xgb = df_fscore_xgb.append(df_fscore_xgb_list, ignore_index = True)
                    
                if i == 0:
                    global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
                    global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
                    global_shap_values_train.columns = ['SHAP_MEAN']
                    global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
                    global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
                    del global_shap_values_train['index']
                    global_shap_values_train['Y_VALUE'] = col
              
                    global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
                    global_shap_values_train.index = global_shap_values_train['FEATURE']
                    df_shap_mean = global_shap_values_train.copy()
                    
                else:
                    global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
                    global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
                    global_shap_values_train.columns = ['SHAP_MEAN']
                    global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
                    global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
                    del global_shap_values_train['index']
                    global_shap_values_train['Y_VALUE'] = col
              
                    global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
                    global_shap_values_train.index = global_shap_values_train['FEATURE']
                    df_shap_mean = df_shap_mean.append(global_shap_values_train, ignore_index = True)
                i = i+1
                #print(col)
                   
            df_shap_mean['SHAP_MEAN_ABS'] = abs(df_shap_mean['SHAP_MEAN'])
            df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] = abs(df_fscore_xgb['FSCORE_GAIN_XGB'])
            df_shap_mean.index.name = 'index'
            df_shap_mean2 = df_shap_mean[['SHAP_MEAN_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
            df_fscore_xgb.index.name = 'index'
            df_fscore_xgb2 = df_fscore_xgb[['FSCORE_GAIN_XGB_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
            
            df_shap_mean = pd.DataFrame(df_shap_mean2.sort_values(by = ['SHAP_MEAN_ABS'], ascending = False))
            df_shap_mean_copy = df_shap_mean.copy()
            df_shap_mean = df_shap_mean.loc[df_shap_mean['SHAP_MEAN_ABS'] >= df_shap_mean['SHAP_MEAN_ABS'].quantile(self.quantile_cutoff_)]
            
            df_fscore_xgb = pd.DataFrame(df_fscore_xgb2.sort_values(by = ['FSCORE_GAIN_XGB_ABS'], ascending = False))
            df_fscore_xgb_copy = df_fscore_xgb.copy()
            df_fscore_xgb = df_fscore_xgb.loc[df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] >= df_fscore_xgb['FSCORE_GAIN_XGB_ABS'].quantile(self.quantile_cutoff_)]
            
            #print(len(df_shap_mean))
            #print(len(df_fscore_xgb))
            
            if len(df_shap_mean_copy) == 0:
                df_shap_mean = df_shap_mean_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            if len(df_fscore_xgb) == 0:
                df_fscore_xgb = df_fscore_xgb_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            #Multicolinearity
            corr = pd.DataFrame(spearmanr(X_train_scaled).correlation).fillna(0)
            corr_linkage = hierarchy.ward(corr)
            cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            
            #selected_features = X_train.columns.to_list()
            for i in range(0, len(y_train.columns)):
                #print(str(i+1) + ' / ' + str(len(y_train.columns)) + '...')
                model = Lasso(alpha=0.001, 
                              max_iter=25000)
                model.fit(X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i])
                
                result = permutation_importance(model, X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i], n_repeats=8,
                                                random_state=0)
                perm_sorted_idx = result.importances_mean.argsort()
                perm_df = pd.DataFrame(result.importances[perm_sorted_idx])
                perm_df['FEATURE'] = X_train.iloc[:, selected_features].iloc[:, perm_sorted_idx].columns.values
                
                if i == 0:
                    perm_df_score = perm_df[['FEATURE']]
                    perm_df_score['PERM_SCORE'] = perm_df.iloc[:, 2]
                else:
                    perm_df_score2 = perm_df[['FEATURE']]
                    perm_df_score2['PERM_SCORE'] = perm_df.iloc[:, 2]
                    
                    perm_df_score = pd.concat([perm_df_score, perm_df_score2], axis = 0, ignore_index = True)
            
            perm_df_score = pd.DataFrame(perm_df_score.groupby(['FEATURE']).mean().reset_index() ).sort_values(by = ['PERM_SCORE'], ascending = False)
            perm_df_score_copy = perm_df_score.copy()
            perm_df_score = perm_df_score.loc[perm_df_score['PERM_SCORE'] >= perm_df_score['PERM_SCORE'].quantile(self.quantile_cutoff_)]
            
            if len(perm_df_score) == 0:
                perm_df_score = perm_df_score_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            all_features = pd.DataFrame(X_train.columns)
            all_features.columns = ['FEATURE']
            
            all_features = pd.merge(all_features, perm_df_score, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_shap_mean, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_fscore_xgb, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
        
            scaler2 = MinMaxScaler(feature_range=(0,100))
            all_features_scaled = pd.DataFrame(scaler2.fit_transform(all_features.iloc[:, 1:]), columns = all_features.iloc[:, 1:].columns, index = all_features.index)
            all_features_scaled['FEATURE'] = all_features['FEATURE']
            all_features_mean = pd.DataFrame(np.mean(all_features_scaled, axis = 1))
            all_features_mean.columns = ['SCORE_MEAN']
            all_features_mean['FEATURE'] = all_features['FEATURE']
            all_features_mean_copy = all_features_mean.copy()
            
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] > 0]
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] >= all_features_mean['SCORE_MEAN'].quantile(0.2)]
            all_features_mean = all_features_mean.dropna().sort_values(by = ['SCORE_MEAN'], ascending = False)
            
            #print(len(all_features_mean))
            if len(all_features_mean) == 0:
                all_features_mean = all_features_mean_copy.sort_values(by = ['SCORE_MEAN'], ascending = False).iloc[0, :]['FEATURE']
                all_features_mean = [all_features_mean]
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean]
                #print(all_features_mean)
            else:
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean['FEATURE']]

        else:
            pass
        
        all_features = pd.DataFrame(X_train.columns)
        all_features.columns = ['FEATURE']
        
        if self.selection_strategy_ in list(['severe']):    
            all_features = pd.merge(all_features, df_shap_mean, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_fscore_xgb, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, perm_df_score, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            
        elif self.selection_strategy_ == 'lasso':  
            all_features = pd.DataFrame(selected_features)
            all_features.columns = ['FEATURE']
        else:
            print("ERROR: Feature selection strategy does not exists. Please select one of ['shap','light', 'severe', 'permutation']")
            sys.exit()
        
        if self.selection_strategy_ != 'lasso':  
            scaler2 = MinMaxScaler(feature_range=(0,100))
            all_features_scaled = pd.DataFrame(scaler2.fit_transform(all_features.iloc[:, 1:]), columns = all_features.iloc[:, 1:].columns, index = all_features.index)
            all_features_scaled['FEATURE'] = all_features['FEATURE']
            all_features_mean = pd.DataFrame(np.mean(all_features_scaled, axis = 1))
            all_features_mean.columns = ['SCORE_MEAN']
            all_features_mean['FEATURE'] = all_features['FEATURE']
            all_features_mean_copy = all_features_mean.copy()
            
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] > 0] 
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] >= all_features_mean['SCORE_MEAN'].quantile(0.2)]
            all_features_mean = all_features_mean.dropna().sort_values(by = ['SCORE_MEAN'], ascending = False)
            
            #print(len(all_features_mean))
            if len(all_features_mean) == 0:
                all_features_mean = all_features_mean_copy.sort_values(by = ['SCORE_MEAN'], ascending = False).iloc[0, :]['FEATURE']
                all_features_mean = [all_features_mean]
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean]
                #print(all_features_mean)
            else:
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean['FEATURE']]

            
            try:
                if self.selection_strategy_ in list(['severe']):
                    corr = pd.DataFrame(spearmanr(X_train_scaled2).correlation).fillna(0)
                    corr_linkage = hierarchy.ward(corr)
                    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
                    cluster_id_to_feature_ids = defaultdict(list)
                    for idx, cluster_id in enumerate(cluster_ids):
                        cluster_id_to_feature_ids[cluster_id].append(idx)
                    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
                    
                    if len(selected_features) == 0:
                        pass
                    else:
                        X_train_scaled2 = X_train_scaled2.iloc[:, selected_features]
                    
                    
                    selected_features = X_train_scaled2.columns.tolist()
                    
                    
                
                    if self.scaler_ in ['Standard', 'MinMax', 'Robust']:
                        X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                        X_train.columns = X_cols
                        X_train = X_train.loc[:, selected_features]
                        X_train.index = X_index
                    else:
                        X_train = X_train.iloc[:, selected_features]
                    
                else:
                    X_train = X_train_scaled.loc[:, all_features_mean['FEATURE']]
            except:
                X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                X_train.columns = X_cols
                try:
                    X_train = X_train.loc[:, all_features_mean['FEATURE']]
                except:
                    X_train = X_train.loc[:, all_features_mean]
                    
                X_train.index = X_index
                
            self.selected_features_ = X_train.columns.to_list()
        
        else:
            #Multicolinearity
            corr = pd.DataFrame(spearmanr(X_train_scaled).correlation).fillna(0)
            corr_linkage = hierarchy.ward(corr)
            cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
           
            #selected_features = X_train.columns.to_list()
            for i in range(0, len(y_train.columns)):
                #print(str(i+1) + ' / ' + str(len(y_train.columns)) + '...')
                model = Lasso(alpha=0.001, max_iter=25000)
                model.fit(X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i])
                
                result = permutation_importance(model, X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i], n_repeats=8,
                                                random_state=0)
                perm_sorted_idx = result.importances_mean.argsort()
                perm_df = pd.DataFrame(result.importances[perm_sorted_idx])
                perm_df['FEATURE'] = X_train.iloc[:, selected_features].iloc[:, perm_sorted_idx].columns.values
                
                if i == 0:
                    perm_df_score = perm_df[['FEATURE']]
                    perm_df_score['PERM_SCORE'] = perm_df.iloc[:, 2]
                else:
                    perm_df_score2 = perm_df[['FEATURE']]
                    perm_df_score2['PERM_SCORE'] = perm_df.iloc[:, 2]
                    
                    perm_df_score = pd.concat([perm_df_score, perm_df_score2], axis = 0, ignore_index = True)
            
            perm_df_score = pd.DataFrame(perm_df_score.groupby(['FEATURE']).mean().reset_index() ).sort_values(by = ['PERM_SCORE'], ascending = False)
            perm_df_score_copy = perm_df_score.copy()
            perm_df_score = perm_df_score.loc[perm_df_score['PERM_SCORE'] >= perm_df_score['PERM_SCORE'].quantile(self.quantile_cutoff_)]
            
            X_train_scaled2 = X_train_scaled.loc[:, perm_df_score['FEATURE']]
            selected_features = X_train_scaled2.columns.tolist()
            
            if len(selected_features) == 0:
                X_train_scaled2 = X_train_scaled.loc[:, perm_df_score_copy.iloc[0, :]['FEATURE']]
                selected_features = X_train_scaled2.columns.tolist()
            else:
                pass
            
            if self.scaler_ in ['Standard', 'MinMax', 'Robust']:
                X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                X_train.columns = X_cols
                X_train = X_train.loc[:, selected_features]
                X_train.index = X_index
            else:
                X_train = X_train_scaled.loc[:, selected_features]
                X_train.index = X_index
                
            self.selected_features_ = X_train.columns.to_list()
            
        
        return X_train
    
    def transform(self, X_test):
        return X_test.loc[:, self.selected_features_]
    
    
    
class FeatureSelectorClassifier:
    
    def __init__(self, selection_strategy = 'light', #['light', 'severe', 'shap', 'permutation']
                 quantile_cutoff = 0.45,
                 alpha = 0.001,
                 scaler = 'Standard', #['None', MinMax, Robust]
                 copy=True):
        self.copy = copy
        self.transformed_ = None
        #self.scaler_ = None
        self.X_ = None
        
        self.selection_strategy_ = selection_strategy
        self.quantile_cutoff_ = quantile_cutoff
        self.alpha_ = alpha
        self.scaler_ = scaler
        
    def fit_transform(self, X_train, y_train):

        X_cols = X_train.columns.tolist()
        X_index = X_train.index
        
        if self.scaler_ == 'Standard':   
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        elif self.scaler_ == 'MinMax':
            scaler = MinMaxScaler(feature_range=(0,100))
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        elif self.scaler_ == 'Robust':
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
            X_train_scaled.columns = X_cols
        else:
            X_train_scaled = X_train.copy()
            
        if self.selection_strategy_ == 'lasso':
            
            model = LogisticRegression(penalty='l1', l1_ratio = self.alpha_) #self.alpha_)#89
            model.fit(X_train_scaled, y_train)

            X_weights = pd.DataFrame(model.coef_)
            X_weights = X_weights.T
            X_weights.columns = X_train_scaled.columns
            X_weights = X_weights.T
            X_weights.columns = ['FEATURE_WEIGHT']
            X_weights['FEATURE_NAME'] = X_weights.index
            X_weights.groupby(['FEATURE_NAME']).sum().sort_values(['FEATURE_WEIGHT'])#.plot.barh()
            #plt.tight_layout()
           # model.intercept_ 
            selected_features = X_weights.loc[abs(X_weights['FEATURE_WEIGHT']) > 0]['FEATURE_NAME'].tolist()

        
        elif self.selection_strategy_  == 'severe':
            i = 0
            for col in y_train.columns:
                xgb_model = XGBClassifier(n_estimators=200, 
                                        learning_rate=0.03, 
                                        subsample=0.61,  
                                        max_depth=9, random_state=0).fit(X_train, y_train[col])
                features_train = pd.DataFrame(list(X_train.columns), columns = ['FEATURE'])
                features_train['index'] = features_train.index
                df_allfeatures = pd.DataFrame(X_train.columns, columns = ['FEATURE'])
                df_allfeatures.index = df_allfeatures['FEATURE']
                df_fscore_xgb_list = pd.DataFrame(xgb_model.get_booster().get_score(importance_type= 'gain'), index=[0]).T.sort_values(by = (0), ascending = False)
                df_fscore_xgb_list.columns = ['FSCORE_GAIN_XGB']
                df_fscore_xgb_list = pd.merge(df_allfeatures, df_fscore_xgb_list, how = 'left', left_index = True, right_index = True)
                df_fscore_xgb_list = df_fscore_xgb_list.fillna(0)
                df_fscore_xgb_list = df_fscore_xgb_list.sort_values(by = (['FSCORE_GAIN_XGB']), ascending = False)
                df_fscore_xgb_list['Y_TRUE'] = col
                shap_explainer = shap.TreeExplainer(xgb_model)
                
                if i == 0:
                    df_fscore_xgb = df_fscore_xgb_list.copy()
                else:
                    df_fscore_xgb = df_fscore_xgb.append(df_fscore_xgb_list, ignore_index = True)
                    
                if i == 0:
                    global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
                    global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
                    global_shap_values_train.columns = ['SHAP_MEAN']
                    global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
                    global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
                    del global_shap_values_train['index']
                    global_shap_values_train['Y_VALUE'] = col
              
                    global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
                    global_shap_values_train.index = global_shap_values_train['FEATURE']
                    df_shap_mean = global_shap_values_train.copy()
                    
                else:
                    global_shap_values_train = pd.DataFrame(shap_explainer.shap_values(X_train))
                    global_shap_values_train = pd.DataFrame(global_shap_values_train.abs().mean(axis=0))
                    global_shap_values_train.columns = ['SHAP_MEAN']
                    global_shap_values_train['index'] = global_shap_values_train.index.astype(int)
                    global_shap_values_train = global_shap_values_train.merge(features_train, left_on='index', right_on='index')
                    del global_shap_values_train['index']
                    global_shap_values_train['Y_VALUE'] = col
              
                    global_shap_values_train = global_shap_values_train.sort_values(by=['SHAP_MEAN'], ascending=False)
                    global_shap_values_train.index = global_shap_values_train['FEATURE']
                    df_shap_mean = df_shap_mean.append(global_shap_values_train, ignore_index = True)
                i = i+1
                #print(col)
                   
            df_shap_mean['SHAP_MEAN_ABS'] = abs(df_shap_mean['SHAP_MEAN'])
            df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] = abs(df_fscore_xgb['FSCORE_GAIN_XGB'])
            df_shap_mean.index.name = 'index'
            df_shap_mean2 = df_shap_mean[['SHAP_MEAN_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
            df_fscore_xgb.index.name = 'index'
            df_fscore_xgb2 = df_fscore_xgb[['FSCORE_GAIN_XGB_ABS','FEATURE']].groupby('FEATURE').mean().reset_index()
            
            df_shap_mean = pd.DataFrame(df_shap_mean2.sort_values(by = ['SHAP_MEAN_ABS'], ascending = False))
            df_shap_mean_copy = df_shap_mean.copy()
            df_shap_mean = df_shap_mean.loc[df_shap_mean['SHAP_MEAN_ABS'] >= df_shap_mean['SHAP_MEAN_ABS'].quantile(self.quantile_cutoff_)]
            
            df_fscore_xgb = pd.DataFrame(df_fscore_xgb2.sort_values(by = ['FSCORE_GAIN_XGB_ABS'], ascending = False))
            df_fscore_xgb_copy = df_fscore_xgb.copy()
            df_fscore_xgb = df_fscore_xgb.loc[df_fscore_xgb['FSCORE_GAIN_XGB_ABS'] >= df_fscore_xgb['FSCORE_GAIN_XGB_ABS'].quantile(self.quantile_cutoff_)]
            
            #print(len(df_shap_mean))
            #print(len(df_fscore_xgb))
            
            if len(df_shap_mean_copy) == 0:
                df_shap_mean = df_shap_mean_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            if len(df_fscore_xgb) == 0:
                df_fscore_xgb = df_fscore_xgb_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            #Multicolinearity
            corr = pd.DataFrame(spearmanr(X_train_scaled).correlation).fillna(0)
            corr_linkage = hierarchy.ward(corr)
            cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            
            #selected_features = X_train.columns.to_list()
            for i in range(0, len(y_train.columns)):
                #print(str(i+1) + ' / ' + str(len(y_train.columns)) + '...')
                model = LogisticRegression(penalty='l1', l1_ratio = 0.001)
                model.fit(X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i])
                
                result = permutation_importance(model, X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i], n_repeats=8,
                                                random_state=0)
                perm_sorted_idx = result.importances_mean.argsort()
                perm_df = pd.DataFrame(result.importances[perm_sorted_idx])
                perm_df['FEATURE'] = X_train.iloc[:, selected_features].iloc[:, perm_sorted_idx].columns.values
                
                if i == 0:
                    perm_df_score = perm_df[['FEATURE']]
                    perm_df_score['PERM_SCORE'] = perm_df.iloc[:, 2]
                else:
                    perm_df_score2 = perm_df[['FEATURE']]
                    perm_df_score2['PERM_SCORE'] = perm_df.iloc[:, 2]
                    
                    perm_df_score = pd.concat([perm_df_score, perm_df_score2], axis = 0, ignore_index = True)
            
            perm_df_score = pd.DataFrame(perm_df_score.groupby(['FEATURE']).mean().reset_index() ).sort_values(by = ['PERM_SCORE'], ascending = False)
            perm_df_score_copy = perm_df_score.copy()
            perm_df_score = perm_df_score.loc[perm_df_score['PERM_SCORE'] >= perm_df_score['PERM_SCORE'].quantile(self.quantile_cutoff_)]
            
            if len(perm_df_score) == 0:
                perm_df_score = perm_df_score_copy.iloc[0, :]['FEATURE']
            else:
                pass
            
            all_features = pd.DataFrame(X_train.columns)
            all_features.columns = ['FEATURE']
            
            all_features = pd.merge(all_features, perm_df_score, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_shap_mean, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_fscore_xgb, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
        
            scaler2 = MinMaxScaler(feature_range=(0,100))
            all_features_scaled = pd.DataFrame(scaler2.fit_transform(all_features.iloc[:, 1:]), columns = all_features.iloc[:, 1:].columns, index = all_features.index)
            all_features_scaled['FEATURE'] = all_features['FEATURE']
            all_features_mean = pd.DataFrame(np.mean(all_features_scaled, axis = 1))
            all_features_mean.columns = ['SCORE_MEAN']
            all_features_mean['FEATURE'] = all_features['FEATURE']
            all_features_mean_copy = all_features_mean.copy()
            
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] > 0]
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] >= all_features_mean['SCORE_MEAN'].quantile(0.2)]
            all_features_mean = all_features_mean.dropna().sort_values(by = ['SCORE_MEAN'], ascending = False)
            
            #print(len(all_features_mean))
            if len(all_features_mean) == 0:
                all_features_mean = all_features_mean_copy.sort_values(by = ['SCORE_MEAN'], ascending = False).iloc[0, :]['FEATURE']
                all_features_mean = [all_features_mean]
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean]
                #print(all_features_mean)
            else:
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean['FEATURE']]
            
            """
            try:
                corr = pd.DataFrame(spearmanr(X_train_scaled2).correlation).fillna(0)
                corr_linkage = hierarchy.ward(corr)
                cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
                cluster_id_to_feature_ids = defaultdict(list)
                for idx, cluster_id in enumerate(cluster_ids):
                    cluster_id_to_feature_ids[cluster_id].append(idx)
                selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            except:
                
            """
            
            #X_train = X_train.iloc[:, selected_features]

        else:
            pass
        
        all_features = pd.DataFrame(X_train.columns)
        all_features.columns = ['FEATURE']
        
        if self.selection_strategy_ in list(['severe']):    
            all_features = pd.merge(all_features, df_shap_mean, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, df_fscore_xgb, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            all_features = pd.merge(all_features, perm_df_score, how = 'left', left_on = ['FEATURE'], right_on = ['FEATURE'])
            
        elif self.selection_strategy_ == 'lasso':  
            all_features = pd.DataFrame(selected_features)
            all_features.columns = ['FEATURE']
        else:
            print("ERROR: Feature selection strategy does not exists. Please select one of ['shap','light', 'severe', 'permutation']")
            sys.exit()
        
        if self.selection_strategy_ != 'lasso':  
            scaler2 = MinMaxScaler(feature_range=(0,100))
            all_features_scaled = pd.DataFrame(scaler2.fit_transform(all_features.iloc[:, 1:]), columns = all_features.iloc[:, 1:].columns, index = all_features.index)
            all_features_scaled['FEATURE'] = all_features['FEATURE']
            all_features_mean = pd.DataFrame(np.mean(all_features_scaled, axis = 1))
            all_features_mean.columns = ['SCORE_MEAN']
            all_features_mean['FEATURE'] = all_features['FEATURE']
            all_features_mean_copy = all_features_mean.copy()
            
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] > 0] 
            all_features_mean = all_features_mean.loc[all_features_mean['SCORE_MEAN'] >= all_features_mean['SCORE_MEAN'].quantile(0.2)]
            all_features_mean = all_features_mean.dropna().sort_values(by = ['SCORE_MEAN'], ascending = False)
            
            #print(len(all_features_mean))
            if len(all_features_mean) == 0:
                all_features_mean = all_features_mean_copy.sort_values(by = ['SCORE_MEAN'], ascending = False).iloc[0, :]['FEATURE']
                all_features_mean = [all_features_mean]
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean]
                #print(all_features_mean)
            else:
                X_train_scaled2 = X_train_scaled.loc[:, all_features_mean['FEATURE']]

            
            try:
                if self.selection_strategy_ in list(['severe']):
                    corr = pd.DataFrame(spearmanr(X_train_scaled2).correlation).fillna(0)
                    corr_linkage = hierarchy.ward(corr)
                    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
                    cluster_id_to_feature_ids = defaultdict(list)
                    for idx, cluster_id in enumerate(cluster_ids):
                        cluster_id_to_feature_ids[cluster_id].append(idx)
                    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
                    
                    if len(selected_features) == 0:
                        pass
                    else:
                        X_train_scaled2 = X_train_scaled2.iloc[:, selected_features]
                    
                    
                    selected_features = X_train_scaled2.columns.tolist()
                    
                    
                
                    if self.scaler_ in ['Standard', 'MinMax', 'Robust']:
                        X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                        X_train.columns = X_cols
                        X_train = X_train.loc[:, selected_features]
                        X_train.index = X_index
                    else:
                        X_train = X_train.iloc[:, selected_features]
                    
                else:
                    X_train = X_train_scaled.loc[:, all_features_mean['FEATURE']]
            except:
                X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                X_train.columns = X_cols
                try:
                    X_train = X_train.loc[:, all_features_mean['FEATURE']]
                except:
                    X_train = X_train.loc[:, all_features_mean]
                    
                X_train.index = X_index
                
            self.selected_features_ = X_train.columns.to_list()
        
        else:
            #Multicolinearity
            corr = pd.DataFrame(spearmanr(X_train_scaled).correlation).fillna(0)
            corr_linkage = hierarchy.ward(corr)
            cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
           
            #selected_features = X_train.columns.to_list()
            for i in range(0, len(y_train.columns)):
                #print(str(i+1) + ' / ' + str(len(y_train.columns)) + '...')
                model = LogisticRegression(penalty='l1', l1_ratio = 0.001)
                model.fit(X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i])
                
                result = permutation_importance(model, X_train_scaled.iloc[:, selected_features], y_train.iloc[:, i], n_repeats=8,
                                                random_state=0)
                perm_sorted_idx = result.importances_mean.argsort()
                perm_df = pd.DataFrame(result.importances[perm_sorted_idx])
                perm_df['FEATURE'] = X_train.iloc[:, selected_features].iloc[:, perm_sorted_idx].columns.values
                
                if i == 0:
                    perm_df_score = perm_df[['FEATURE']]
                    perm_df_score['PERM_SCORE'] = perm_df.iloc[:, 2]
                else:
                    perm_df_score2 = perm_df[['FEATURE']]
                    perm_df_score2['PERM_SCORE'] = perm_df.iloc[:, 2]
                    
                    perm_df_score = pd.concat([perm_df_score, perm_df_score2], axis = 0, ignore_index = True)
            
            perm_df_score = pd.DataFrame(perm_df_score.groupby(['FEATURE']).mean().reset_index() ).sort_values(by = ['PERM_SCORE'], ascending = False)
            perm_df_score_copy = perm_df_score.copy()
            perm_df_score = perm_df_score.loc[perm_df_score['PERM_SCORE'] >= perm_df_score['PERM_SCORE'].quantile(self.quantile_cutoff_)]
            
            X_train_scaled2 = X_train_scaled.loc[:, perm_df_score['FEATURE']]
            selected_features = X_train_scaled2.columns.tolist()
            
            if len(selected_features) == 0:
                X_train_scaled2 = X_train_scaled.loc[:, perm_df_score_copy.iloc[0, :]['FEATURE']]
                selected_features = X_train_scaled2.columns.tolist()
            else:
                pass
            
            if self.scaler_ in ['Standard', 'MinMax', 'Robust']:
                X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled.values))
                X_train.columns = X_cols
                X_train = X_train.loc[:, selected_features]
                X_train.index = X_index
            else:
                X_train = X_train_scaled.loc[:, selected_features]
                X_train.index = X_index
                
            self.selected_features_ = X_train.columns.to_list()
            
        
        return X_train
    
    def transform(self, X_test):
        return X_test.loc[:, self.selected_features_]


class FeatureBoosterRegressor():
    def __init__(self, 
                 n_best_rules = 35,
                 base_model = RandomForestRegressor(criterion='friedman_mse', 
                                                    max_depth=5,
                                                    max_features=None, 
                                                    max_leaf_nodes=2,
                                                    min_samples_leaf=1, 
                                                    verbose=0, 
                                                    n_estimators = 850,
                                                    warm_start=True,
                                                    random_state = 0),
                 random_state=0, 
                 max_rules=3000,
                 selection_strategy = 'severe',  
                 alpha = 0.001,
                 scaler = 'None',
                 warm_start=True,
                 quantile_cutoff = 0.45,
                 original_features_selection = False,
                 copy=True):
        self.copy = copy
        self.transformed_ = None
        self.scaler_ = scaler
        self.X_ = None
        
        self.n_best_rules_ = n_best_rules
        self.base_model_ = base_model
        self.random_state_ = random_state
        self.max_rules_ = max_rules
        self.selection_strategy_ = selection_strategy
        self.quantile_cutoff_ = quantile_cutoff
        self.alpha_ = alpha
        self.original_features_selection_ = original_features_selection
        #self.reset() 
        
    def reset(self):
        self.__init__()
        # set all members to their initial value
        
    def fit_transform(self, X, y):
        #X =X_train.copy()
        #y = y_train.copy()
        if self.original_features_selection_ == True:
            X = FeatureSelectorRegressor(selection_strategy = 'severe',  
                                         quantile_cutoff = 0.1,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_
                                         ).fit_transform(X, y)
            self.original_features_selection_list_ = X.columns.tolist() 
        else:
            pass
        
        poly = PolynomialFeatures(2, include_bias = False)
        poly.fit(X)
        poly_X = pd.DataFrame(poly.transform(X))
        poly_X.columns = poly.get_feature_names(input_features = X.columns.tolist())
        poly_X = poly_X.loc[:, poly_X.columns.difference(X.columns.tolist())]
        
        self.poly_ = poly
        
        
        #feature_selection  = FeatureSelector(selection_strategy = 'severe',  quantile_cutoff = 0.45) 
        poly_X = FeatureSelectorRegressor(selection_strategy = self.selection_strategy_,  
                                         quantile_cutoff = self.quantile_cutoff_,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_
                                         ).fit_transform(poly_X, y)
        #print(self.selection_strategy_)
        #print(self.quantile_cutoff_)
        poly_X.index = X.index
        self.poly_features_ = poly_X.columns.to_list()
        #poly_X = poly_X[poly_features]
        #print(self.poly_features_)
        
        new_X = pd.concat([X, poly_X], axis = 1)
        new_X.columns = new_X.columns.str.replace(" ", "_")
        new_X.columns = new_X.columns.str.replace("^", "")
        feature_names = list(new_X.columns)
        
        rule_X, rules = rules_feature_engineering(new_X.copy(),
                                                  y.copy(),
                                                  feature_names,
                                                  n_best_rules = self.n_best_rules_,
                                                  base_model = self.base_model_, 
                                                  random_state = self.random_state_,
                                                  max_rules = self.max_rules_) 
           
        rule_X = rule_X.loc[:, rule_X.columns.difference(new_X.columns.tolist())]
        
        #feature_selection  = FeatureSelector(selection_strategy = 'severe',  quantile_cutoff = 0.45) 
        rule_X = FeatureSelectorRegressor(selection_strategy = self.selection_strategy_,  
                                         quantile_cutoff = self.quantile_cutoff_,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_).fit_transform(rule_X, y)
        #print(self.selection_strategy_)
        #print(self.quantile_cutoff_)
        
        #rule_X = rule_X[rule_features]
        new_X = pd.concat([new_X, rule_X], axis = 1)
        rule_X.columns
        #print(new_X.columns.to_list())
        
        #final_features = feature_selection(new_X, y)
        #new_X = new_X[final_features]
        
        """
        new_X = FeatureSelector(selection_strategy = 'severe',  
                                 quantile_cutoff = self.quantile_cutoff_,
                                 alpha = self.alpha_,
                                 scaler = self.scaler_).fit_transform(new_X, y)
        """
        
        self.final_features_ = new_X.columns.to_list()
        
        rules['SELECTED'] = np.where(rules['FEATURE_NAME'].isin(new_X.columns.to_list()) == True, 1, 0)
        #rules = rules.loc[rules['FEATURE_NAME'].isin(new_X.columns.to_list())]
        
        self.rules_df_ = rules
        
        return new_X, rules
    
    def transform(self, X):
        if self.original_features_selection_ == True:
            X = X.loc[:, self.original_features_selection_list_]
        else:
            pass
        
        poly_X = pd.DataFrame(self.poly_.transform(X))
        poly_X.columns = self.poly_.get_feature_names(input_features = X.columns.tolist())
        poly_X = poly_X.loc[:, self.poly_features_]
        poly_X.index = X.index
 
        
        new_X = pd.concat([X, poly_X], axis = 1)
        new_X.columns = new_X.columns.str.replace(" ", "_")
        new_X.columns = new_X.columns.str.replace("^", "")
        
        data = new_X.copy()
        #print(data.columns)
        
        for i in range(0, len(self.rules_df_)):
            if i == 0:
                rule_X = np.where(eval(self.rules_df_[['PYTHON_CODE']].iloc[i, :].values[0]), 1, 0)
                rule_X = pd.DataFrame(rule_X)
                rule_X.columns = [self.rules_df_[['FEATURE_NAME']].iloc[i, :].values[0]]
            else:
                rule_X2 = np.where(eval(self.rules_df_[['PYTHON_CODE']].iloc[i, :].values[0]), 1, 0)
                rule_X2 = pd.DataFrame(rule_X2)
                rule_X2.columns = [self.rules_df_[['FEATURE_NAME']].iloc[i, :].values[0]]
                rule_X = pd.concat([rule_X, rule_X2], axis = 1)
        
        rule_X.index = new_X.index
        
        new_X = pd.concat([new_X, rule_X], axis = 1)

        new_X = new_X.loc[:, self.final_features_]
        
        
        return new_X
    
    
class FeatureBoosterClassifier():
    def __init__(self, 
                 n_best_rules = 35,
                 base_model = RandomForestClassifier(criterion='gini', 
                                                    max_depth=5,
                                                    max_features=None, 
                                                    max_leaf_nodes=2,
                                                    min_samples_leaf=1, 
                                                    verbose=0, 
                                                    n_estimators = 850,
                                                    warm_start=True,
                                                    random_state = 0),
                 random_state=0, 
                 max_rules=3000,
                 selection_strategy = 'severe',  
                 alpha = 0.001,
                 scaler = 'None',
                 warm_start=True,
                 quantile_cutoff = 0.45,
                 original_features_selection = False,
                 copy=True):
        self.copy = copy
        self.transformed_ = None
        self.scaler_ = scaler
        self.X_ = None
        
        self.n_best_rules_ = n_best_rules
        self.base_model_ = base_model
        self.random_state_ = random_state
        self.max_rules_ = max_rules
        self.selection_strategy_ = selection_strategy
        self.quantile_cutoff_ = quantile_cutoff
        self.alpha_ = alpha
        self.original_features_selection_ = original_features_selection
        #self.reset() 
        
    def reset(self):
        self.__init__()
        # set all members to their initial value
        
    def fit_transform(self, X, y):
        #X =X_train.copy()
        #y = y_train.copy()
        if self.original_features_selection_ == True:
            X = FeatureSelectorClassifier(selection_strategy = 'severe',  
                                         quantile_cutoff = 0.1,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_
                                         ).fit_transform(X, y)
            self.original_features_selection_list_ = X.columns.tolist() 
        else:
            pass
        
        poly = PolynomialFeatures(2, include_bias = False)
        poly.fit(X)
        poly_X = pd.DataFrame(poly.transform(X))
        poly_X.columns = poly.get_feature_names(input_features = X.columns.tolist())
        poly_X = poly_X.loc[:, poly_X.columns.difference(X.columns.tolist())]
        
        self.poly_ = poly
        
        
        #feature_selection  = FeatureSelector(selection_strategy = 'severe',  quantile_cutoff = 0.45) 
        poly_X = FeatureSelectorClassifier(selection_strategy = self.selection_strategy_,  
                                         quantile_cutoff = self.quantile_cutoff_,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_
                                         ).fit_transform(poly_X, y)
        #print(self.selection_strategy_)
        #print(self.quantile_cutoff_)
        poly_X.index = X.index
        self.poly_features_ = poly_X.columns.to_list()
        #poly_X = poly_X[poly_features]
        #print(self.poly_features_)
        
        new_X = pd.concat([X, poly_X], axis = 1)
        new_X.columns = new_X.columns.str.replace(" ", "_")
        new_X.columns = new_X.columns.str.replace("^", "")
        feature_names = list(new_X.columns)
        
        rule_X, rules = rules_feature_engineering(new_X.copy(),
                                                  y.copy(),
                                                  feature_names,
                                                  n_best_rules = self.n_best_rules_,
                                                  base_model = self.base_model_, 
                                                  random_state = self.random_state_,
                                                  max_rules = self.max_rules_) 
           
        rule_X = rule_X.loc[:, rule_X.columns.difference(new_X.columns.tolist())]
        
        #feature_selection  = FeatureSelector(selection_strategy = 'severe',  quantile_cutoff = 0.45) 
        rule_X = FeatureSelectorClassifier(selection_strategy = self.selection_strategy_,  
                                         quantile_cutoff = self.quantile_cutoff_,
                                         alpha = self.alpha_,
                                         scaler = self.scaler_).fit_transform(rule_X, y)
        #print(self.selection_strategy_)
        #print(self.quantile_cutoff_)
        
        #rule_X = rule_X[rule_features]
        new_X = pd.concat([new_X, rule_X], axis = 1)
        rule_X.columns
        #print(new_X.columns.to_list())
        
        #final_features = feature_selection(new_X, y)
        #new_X = new_X[final_features]
        
        """
        new_X = FeatureSelector(selection_strategy = 'severe',  
                                 quantile_cutoff = self.quantile_cutoff_,
                                 alpha = self.alpha_,
                                 scaler = self.scaler_).fit_transform(new_X, y)
        """
        
        self.final_features_ = new_X.columns.to_list()
        
        rules = rules.loc[rules['FEATURE_NAME'].isin(new_X.columns.to_list())]
        
        self.rules_df_ = rules
        
        return new_X, rules
    
    def transform(self, X):
        if self.original_features_selection_ == True:
            X = X.loc[:, self.original_features_selection_list_]
        else:
            pass
        
        poly_X = pd.DataFrame(self.poly_.transform(X))
        poly_X.columns = self.poly_.get_feature_names(input_features = X.columns.tolist())
        poly_X = poly_X.loc[:, self.poly_features_]
        poly_X.index = X.index
 
        
        new_X = pd.concat([X, poly_X], axis = 1)
        new_X.columns = new_X.columns.str.replace(" ", "_")
        new_X.columns = new_X.columns.str.replace("^", "")
        
        data = new_X.copy()
        #print(data.columns)
        
        for i in range(0, len(self.rules_df_)):
            if i == 0:
                rule_X = np.where(eval(self.rules_df_[['PYTHON_CODE']].iloc[i, :].values[0]), 1, 0)
                rule_X = pd.DataFrame(rule_X)
                rule_X.columns = [self.rules_df_[['FEATURE_NAME']].iloc[i, :].values[0]]
            else:
                rule_X2 = np.where(eval(self.rules_df_[['PYTHON_CODE']].iloc[i, :].values[0]), 1, 0)
                rule_X2 = pd.DataFrame(rule_X2)
                rule_X2.columns = [self.rules_df_[['FEATURE_NAME']].iloc[i, :].values[0]]
                rule_X = pd.concat([rule_X, rule_X2], axis = 1)
        
        rule_X.index = new_X.index
        
        new_X = pd.concat([new_X, rule_X], axis = 1)

        new_X = new_X.loc[:, self.final_features_]
        
        
        return new_X



