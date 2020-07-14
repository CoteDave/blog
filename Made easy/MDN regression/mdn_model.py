import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import math  

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

import scipy.stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
from umap import UMAP

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from hdbscan import HDBSCAN


tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_v2_behavior()

def sample_from_mixture_f(x, pred_weights, pred_means, pred_std, dist):
          """Draws samples from mixture model.
        
          Returns 2 d array with input X and sample from prediction of mixture model.
          """
          amount = len(x)
          samples = np.zeros((amount, 1))
          n_mix = len(pred_weights[0])
          to_choose_from = np.arange(n_mix)
          for j, (weights, means, std_devs) in enumerate(
                  zip(pred_weights, pred_means, pred_std)):
            index = np.random.choice(to_choose_from, p=weights)
            if dist.lower() == 'normal':
                samples[j, 0] = np.random.normal(means[index], std_devs[index], size=1)
            elif (dist.lower() == 'laplace' or dist.lower() == 'laplacian') == True:
                samples[j, 0] = np.random.laplace(means[index], std_devs[index], size=1)
            #samples[j, 0] = x[j]
            if j == amount - 1:
              break
          return samples

def listToString(s):  
            # traverse in the string  
            liste = []
            for ele in s:  
                ele = str(ele)
                ele = 'X' + ele
                liste.append(ele)
            # return string   
            return liste
        
class MDN:
    def __init__(self, n_mixtures = -1, 
                 dist = 'laplace',
                 #n_layers = 1, 
                 input_neurons = 1000, 
                 hidden_neurons = [25], 
                 gmm_boost = False,
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 early_stopping = 10,
                 tf_mixture_family = True,
                 input_activation = 'relu',
                 hidden_activation = 'tanh'):
        
        tf.compat.v1.reset_default_graph()
        self.n_mixtures = n_mixtures
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        #self.n_layers = n_layers
        self.gmm_boost = gmm_boost
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.tf_mixture_family = tf_mixture_family
        self.optimizer = optimizer
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.dist = dist
        
    def fit(self, X, Y, epochs, batch_size):
        
        EPOCHS = epochs
        BATCH_SIZE = batch_size
        n = len(X)
        XY = np.concatenate((X, Y), axis = 1)
        #df = n - 1
        
        self._X = X.copy()
        hidden_neurons = self.hidden_neurons
        
        if self.n_mixtures == -1:
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, 7)
            cv_types = ['spherical', 'tied', 'diag', 'full']
            for cv_type in cv_types:
                for n_components in n_components_range:
                    # Fit a Gaussian mixture with EM
                    gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,
                                          max_iter = 10000)
                    gmm.fit(XY)
                    bic.append(gmm.bic(XY))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                        self.n_mixtures = n_components
            
            clusterer = HDBSCAN()
            clusterer.fit(XY)
            clusterer.labels_
            
            if len(np.unique(clusterer.labels_)) < self.n_mixtures:
                self.n_mixtures = len(np.unique(clusterer.labels_)) 
            else:
                pass
            
            if self.gmm_boost == True:
                if len(np.unique(clusterer.labels_)) < self.n_mixtures:
                    clusterer = HDBSCAN()
                    clusterer.fit(X)
                    clusters = clusterer.labels_
                else:
                    clusterer = best_gmm
                    clusterer.fit(X)
                    clusters = clusterer.predict_proba(X)
                    
                self._clusterer = clusterer
                
                X = np.concatenate((X, clusters), axis = 1)
            
            else:
                pass
            
        elif self.gmm_boost == True:

            clusterer1 = BayesianGaussianMixture(n_components=self.n_mixtures, covariance_type='full', max_iter=10000)
            clusterer1.fit(X)
            clusters = clusterer1.predict_proba(X)
            self._clusterer = clusterer1
            
            clusterer2 = HDBSCAN()
            clusterer2.fit(X)
            
            if len(np.unique(clusterer2.labels_)) < self.n_mixtures:
                clusters = clusterer2.labels_
                self._clusterer = clusterer2
            else:
                pass

            X = np.concatenate((X, clusters), axis = 1)
        
        else:
            pass
        
        
        self._y = Y.copy()
        
        dataset = tf.compat.v1.data.Dataset \
    					.from_tensor_slices((X, Y)) \
    					.repeat(EPOCHS).shuffle(len(X)).batch(BATCH_SIZE)
        iter_ = tf.compat.v1.data.make_one_shot_iterator(dataset)
        
        x, y = iter_.get_next()

        K = self.n_mixtures
        
        self.K = K
        self.x = x
        
        input_activation = self.input_activation
        hidden_activation = self.hidden_activation
        
        if input_activation.lower() == 'crelu':
            input_actv = tf.nn.crelu
        elif input_activation.lower() == 'relu6':
            input_actv = tf.nn.relu6
        elif input_activation.lower() == 'elu':
            input_actv = tf.nn.elu
        elif input_activation.lower() == 'selu':
            input_actv = tf.nn.selu
        elif input_activation.lower() == 'leaky_relu':
            input_actv = tf.nn.leaky_relu
        elif input_activation.lower() == 'relu':
            input_actv = tf.nn.relu
        elif input_activation.lower() == 'swish':
            input_actv = tf.nn.swish
        elif input_activation.lower() == 'tanh':
            input_actv = tf.nn.tanh
        elif input_activation.lower() == 'linear':
            input_actv = None
        elif input_activation.lower() == 'softplus':
            input_actv = tf.nn.softplus
        elif input_activation.lower() == 'sigmoid':
            input_actv = tf.nn.sigmoid
        elif input_activation.lower() == 'softmax':
            input_actv = tf.nn.softmax
        else:
            input_actv = tf.nn.relu
            
        if hidden_activation.lower() == 'crelu':
            h_actv = tf.nn.crelu
        elif hidden_activation.lower() == 'relu6':
            h_actv = tf.nn.relu6
        elif hidden_activation.lower() == 'elu':
            h_actv = tf.nn.elu
        elif hidden_activation.lower() == 'selu':
            h_actv = tf.nn.selu
        elif hidden_activation.lower() == 'leaky_relu':
            h_actv = tf.nn.leaky_relu
        elif hidden_activation.lower() == 'relu':
            h_actv = tf.nn.relu
        elif hidden_activation.lower() == 'swish':
            h_actv = tf.nn.swish
        elif hidden_activation.lower() == 'tanh':
            h_actv = tf.nn.tanh
        elif hidden_activation.lower() == 'linear':
            h_actv = None
        elif hidden_activation.lower() == 'softplus':
            h_actv = tf.nn.softplus
        elif hidden_activation.lower() == 'sigmoid':
            h_actv = tf.nn.sigmoid
        elif hidden_activation.lower() == 'softmax':
            h_actv = tf.nn.softmax
        else:
            h_actv = tf.nn.relu
        
        n_layer = len(hidden_neurons)
        
        if n_layer < 1:
            self.layer_last = tf.layers.dense(x, units=self.input_neurons, activation=input_actv)
            self.mu = tf.layers.dense(self.layer_last, units=K, activation=None, name="mu")
            self.var = tf.exp(tf.layers.dense(self.layer_last, units=K, activation=None, name="sigma"))
            self.pi = tf.layers.dense(self.layer_last, units=K, activation=tf.nn.softmax, name="mixing")
            
        else:
            self.layer_1 = tf.layers.dense(x, units=self.input_neurons, activation=input_actv)
            for i in range(2, n_layer + 2):
                
                n_neurons = hidden_neurons[i-2]
                
                if i == n_layer+1:
                    print('last', i)
                    string_var = 'self.layer_last = tf.layers.dense(self.layer_' + str(i-1) + ', units=n_neurons, activation=h_actv)'
                else:
                    print(i)
                    string_var = 'self.layer_' + str(i) + ' = tf.layers.dense(self.layer_' + str(i-1) + ', units=n_neurons, activation=h_actv)'
                    
                exec(string_var)
            
            self.mu = tf.layers.dense(self.layer_last, units=K, activation=None, name="mu")
            self.var = tf.exp(tf.layers.dense(self.layer_last, units=K, activation=None, name="sigma"))
            self.pi = tf.layers.dense(self.layer_last, units=K, activation=tf.nn.softmax, name="mixing")

        if self.tf_mixture_family == False:
            #---------------- Not using TF Mixture Family ------------------------
            if self.dist.lower() == 'normal':
                self.likelihood = tfp.distributions.Normal(loc=self.mu, scale=self.var)
            elif (self.dist.lower() == 'laplacian' or self.dist.lower() == 'laplace') == True:    
                self.likelihood = tfp.distributions.Laplace(loc=self.mu, scale=self.var)
            elif self.dist.lower() == 'lognormal' :    
                self.likelihood = tfp.distributions.LogNormal(loc=self.mu, scale=self.var)
            elif self.dist.lower() == 'gamma':
                alpha = (self.mu**2)/ self.var
                beta = self.var / self.mu
                self.likelihood = tfp.distributions.Gamma(concentration = alpha, rate = beta)
            else:
                self.likelihood = tfp.distributions.Normal(loc=self.mu, scale=self.var)
                
            self.out = self.likelihood.prob(y)
            self.out = tf.multiply(self.out, self.pi)
            self.out = tf.reduce_sum(self.out, 1, keepdims=True)
            self.out = -tf.log(self.out + 1e-10)
            self.mean_loss = tf.reduce_mean(self.out)

        else:
    		# -------------------- Using TF Mixture Family ------------------------
            self.mixture_distribution = tfp.distributions.Categorical(probs=self.pi)
            
            if self.dist.lower() == 'normal':
                self.distribution = tfp.distributions.Normal(loc=self.mu, scale=self.var)
            elif (self.dist.lower() == 'laplacian' or self.dist.lower() == 'laplace') == True:    
                self.distribution = tfp.distributions.Laplace(loc=self.mu, scale=self.var)
            elif self.dist.lower() == 'lognormal':    
                #self.distribution = tfp.edward2.LogNormal(loc=self.mu, scale=self.var)
                self.distribution = tfp.distributions.LogNormal(loc=self.mu, scale=self.var)
            elif self.dist.lower() == 'gamma':
                alpha = (self.mu**2)/ self.var
                beta = self.var / self.mu
                self.distribution = tfp.distributions.Gamma(concentration = alpha, rate = beta)
            else:
                self.distribution = tfp.distributions.Normal(loc=self.mu, scale=self.var)
            
            self.likelihood = tfp.distributions.MixtureSameFamily(
    															mixture_distribution=self.mixture_distribution,
            													components_distribution=self.distribution)
            self.log_likelihood = -self.likelihood.log_prob(tf.transpose(y))

            self.mean_loss = tf.reduce_mean(self.log_likelihood)

		# ----------------------------------------------------------------------

        self.global_step = tf.Variable(0, trainable=False)
        
        if self.optimizer.lower() == 'adam':
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'adadelta':
            self.train_op = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'adagradda':
            self.train_op = tf.compat.v1.train.AdagradDAOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'adagrad':
            self.train_op = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'ftrl':
            self.train_op = tf.compat.v1.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'gradientdescent':
            self.train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'momentum':
            self.train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        
        elif self.optimizer.lower() == 'proximaladagrad':
            self.train_op = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'proximalgradientdescent':
            self.train_op = tf.compat.v1.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'rmsprop':
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
      
        else:
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
            
        self.init = tf.compat.v1.global_variables_initializer()

		# Initialize coefficients
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init)
        
        
        best_loss = 1e+10
        self.stopping_step = 0    
        for i in range(EPOCHS * (n//BATCH_SIZE)): 
            _, loss, mu, var, pi, x__ = self.sess.run([self.train_op,
        													self.mean_loss,
        													self.mu, 
                                                            self.var, 
                                                            self.pi,
        													self.x])
        
            if loss < best_loss:
                self.stopping_step = 0
                self.best_loss = loss
            
                best_mu = mu
                best_var = var
                best_pi = pi
                best_mean_y = mu[:,0]
                best_x = x__
                best_loss = loss
                print("Epoch: {} Loss: {:3.3f}".format(i, loss))
            else:
                self.stopping_step += 1
                
            if self.stopping_step >= self.early_stopping:
                self.should_stop = True
                print("Early stopping is trigger at step: {} loss:{}".format(i,loss))
                return
            else:
                pass
            
            self._mean_y_train = mu[:,0]
            self._dist_mu_train = mu
            self._dist_var_train = var
            self._dist_pi_train = pi
            self._x_data_train = x__
            
                
        

        
    def predict_best(self, X_pred, q = 0.95, y_scaler = None):
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        best_mean_y, best_mu, best_var, best_pi, best_x  =  self.sess.run([self.mu[:,0], 
                      self.mu, 
                      self.var, 
                      self.pi,
                      self.x], 
                      feed_dict={self.x: X_pred})
        
        self._mean_y_pred = best_mean_y
        self._dist_mu_pred = best_mu
        self._dist_var_pred = best_var
        self._dist_pi_pred = best_pi
        self._x_data_pred = best_x
        
        cluster_probs = self._dist_pi_pred
        best_cluster = np.argmax(cluster_probs, axis = 1)
        best_cluster_prob = np.max(cluster_probs, axis = 1)
        
        list_dist_mu = []
        for i in range(0, len(self._dist_mu_pred)):
            list_dist_mu.append(self._dist_mu_pred[i, np.argmax(cluster_probs, axis = 1)[i]])
        
        list_dist_var = []
        for i in range(0, len(self._dist_var_pred)):
            list_dist_var.append(self._dist_var_pred[i, np.argmax(cluster_probs, axis = 1)[i]])
            
        
        if y_scaler != None:
            y_pred_mean = y_scaler.inverse_transform(np.array(list_dist_mu).reshape(1, -1))
            y_pred_upper = y_scaler.inverse_transform((np.array(list_dist_mu) + (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))).reshape(1, -1))
            y_pred_lower = y_scaler.inverse_transform((np.array(list_dist_mu) - (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))).reshape(1, -1))

        else:
            y_pred_mean = np.array(list_dist_mu)
            y_pred_upper = np.array(list_dist_mu) + (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
            y_pred_lower = np.array(list_dist_mu) - (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))

        
        all_preds = pd.DataFrame(y_pred_mean.reshape(-1,1))
        all_preds.columns = ['Y_PRED_MEAN']
        all_preds['Y_PRED_LOWER'] = y_pred_lower.reshape(-1,1)
        all_preds['Y_PRED_UPPER'] = y_pred_upper.reshape(-1,1)
        
        all_preds['BEST_CLUSTER'] = best_cluster
        all_preds['DIST_PI'] = best_cluster_prob
        
        all_preds['DIST_MU'] = np.array(list_dist_mu)
        all_preds['DIST_SIGMA'] = np.array(list_dist_var)
        
        if y_scaler != None:
            all_preds['DIST_MU_UNSCALED'] = y_scaler.inverse_transform(np.array(list_dist_mu))
            all_preds['DIST_SIGMA_UNSCALED'] = y_scaler.inverse_transform(np.array(list_dist_var))
        else:
            pass
        
        return all_preds#, samples
    
    def predict_mixed(self, X_pred, q = 0.95, y_scaler = None):
        
        

        all_preds = self.predict_dist(X_pred = X_pred, q=q, y_scaler = y_scaler)
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        i = 0
        for elem in all_preds:
            if i == 0:
                y_pred_mean = elem['Y_PRED_MEAN'] * elem['DIST_PI']
                y_pred_lower = elem[['Y_PRED_LOWER']]
                y_pred_upper = elem[['Y_PRED_UPPER']]
            else:
                y_pred_mean = y_pred_mean + (elem['Y_PRED_MEAN'] * elem['DIST_PI'])
                y_pred_lower2 = pd.concat([y_pred_lower, elem[['Y_PRED_LOWER']]], axis = 1)
                y_pred_lower2 = np.nanmin(y_pred_lower2.values, axis=1)
                y_pred_lower['Y_PRED_LOWER'] = y_pred_lower2
                
                y_pred_upper2 = pd.concat([y_pred_upper, elem[['Y_PRED_UPPER']]], axis = 1)
                y_pred_upper2 = np.nanmax(y_pred_upper2.values, axis=1)
                y_pred_upper['Y_PRED_UPPER'] = y_pred_upper2
            i = i + 1
        
        all_preds = pd.concat([y_pred_mean, y_pred_lower, y_pred_upper], axis = 1)
        all_preds.columns = ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']
        
        return all_preds#, samples
    
    def predict_with_overlaps(self, X_pred, q=0.95, y_scaler = None, pi_threshold = 0.215):

        all_preds = self.predict_dist(X_pred = X_pred, q=q, y_scaler = y_scaler)
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        X_df = pd.DataFrame(X_pred)
        xcol_list = list(range(0, len(X_df.columns)))
        xcol_list = listToString(xcol_list)

        X_df.columns = xcol_list
        
        i = 0
        for elem in all_preds:
            if i == 0:
                y_mix = elem[['DIST_PI', 'Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']]
                y_mix.columns = ['M' + str(i) + '_' + 'DIST_PI', 'M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER']
                full_data = pd.concat([X_df, y_mix], axis = 1)
                full_data['MIXTURE_' + str(i)] = np.where(full_data['M' + str(i) + '_' + 'DIST_PI'] >= pi_threshold, 'M' + str(i), '')
            else:
                y_mix = elem[['DIST_PI', 'Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']]
                y_mix.columns = ['M' + str(i) + '_' + 'DIST_PI', 'M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER']
                full_data = pd.concat([full_data, y_mix], axis = 1)
                full_data['MIXTURE_' + str(i)] = np.where(full_data['M' + str(i) + '_' + 'DIST_PI'] >= pi_threshold, 'M' + str(i), '')
            i = i +1
                
        mixtures_cols = full_data.loc[:, full_data.columns.str.contains('MIXTURE')].columns.tolist()
        full_data['POSSIBLE_MIX'] = full_data[mixtures_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
        
        for i in range(0, self.n_mixtures+10):
            full_data['POSSIBLE_MIX'] = full_data['POSSIBLE_MIX'].str.replace(',,',',')
            
        full_data['POSSIBLE_MIX'] = np.where(full_data['POSSIBLE_MIX'].str[0:1] == ',', full_data['POSSIBLE_MIX'].str[1:], full_data['POSSIBLE_MIX'])
        full_data['POSSIBLE_MIX'] = np.where(full_data['POSSIBLE_MIX'].str[-1:] == ',', full_data['POSSIBLE_MIX'].str[:-1], full_data['POSSIBLE_MIX'])
        
        i = 0
        renamed_cols = X_df.columns.tolist() + ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER', 'DIST_PI', 'POSSIBLE_MIX', 'SOURCE']
        for col in mixtures_cols:
            all_cols = X_df.columns.tolist() + ['M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER', 'M' + str(i) + '_' + 'DIST_PI', 'POSSIBLE_MIX']
            if i == 0:
                to_merge_data = full_data[full_data[col] != ''][all_cols]
                to_merge_data['SOURCE'] = 'M' + str(i)
                to_merge_data.columns = renamed_cols
            else:
                to_merge_data2 = full_data[full_data[col] != ''][all_cols]
                to_merge_data2['SOURCE'] = 'M' + str(i)
                to_merge_data2.columns = renamed_cols
                to_merge_data = pd.concat([to_merge_data, to_merge_data2], axis = 0, ignore_index = True)
            i = i +1
        
        return to_merge_data


    def predict_dist(self, X_pred, q = 0.95, y_scaler = None):
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        best_mean_y, best_mu, best_var, best_pi, best_x  =  self.sess.run([self.mu[:,0], 
                      self.mu, 
                      self.var, 
                      self.pi,
                      self.x], 
                      feed_dict={self.x: X_pred})
        
        self._mean_y_pred = best_mean_y
        self._dist_mu_pred = best_mu
        self._dist_var_pred = best_var
        self._dist_pi_pred = best_pi
        self._x_data_pred = best_x
        
        cluster_probs = self._dist_pi_pred
        best_cluster = np.argmax(cluster_probs, axis = 1)
        best_cluster_prob = np.max(cluster_probs, axis = 1)
        
        cluster_preds = []
        for i in range(0, self._dist_mu_pred.shape[1]):
            if y_scaler != None:
                y_pred_mean = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]))
                y_pred_upper = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]) + (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1)))
                y_pred_lower = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]) - (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1)))
                y_mu = np.array(self._dist_mu_pred[:, i])
                y_var = np.array(self._dist_var_pred[:, i])
            else:
                y_pred_mean = np.array(self._dist_mu_pred[:, i])
                y_pred_upper = np.array(self._dist_mu_pred[:, i]) + (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
                y_pred_lower = np.array(self._dist_mu_pred[:, i]) - (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
                y_mu = np.array(self._dist_mu_pred[:, i])
                y_var = np.array(self._dist_var_pred[:, i])
            #preds = np.concatenate((y_pred_mean, y_pred_lower, y_pred_upper, self._dist_pi_pred[:, i].reshape(-1,1)), axis = 1)
            preds = pd.concat([pd.DataFrame(y_pred_mean),
                              pd.DataFrame(y_pred_lower),
                              pd.DataFrame(y_pred_upper),
                              pd.DataFrame(self._dist_pi_pred[:, i]),
                              pd.DataFrame(y_mu),
                              pd.DataFrame(y_var),
                              ], axis = 1, ignore_index = True)
            #preds = pd.DataFrame(preds)
            preds.columns = ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER', 'DIST_PI', 'DIST_MU', 'DIST_VAR']
            preds['N_CLUSTER'] = i
            
            cluster_preds.append(preds)
            
        
        
        
        return cluster_preds#, samples

    def sample_from_mixture(self, X_pred, n_samples_batch = 1, y_scaler = None):
        
        all_preds = self.predict_dist(X_pred = X_pred, q = 0.95, y_scaler = y_scaler)
        
        i = 0
        for elem in all_preds:
            if i == 0:
                out_pi_test = elem[['DIST_PI']]
                out_mu_test = elem[['DIST_MU']]
                out_sigma_test = elem[['DIST_VAR']]
            else:
                out_pi_test2 = elem[['DIST_PI']]
                out_mu_test2 = elem[['DIST_MU']]
                out_sigma_test2 = elem[['DIST_VAR']]
                out_pi_test = pd.concat([out_pi_test, out_pi_test2], axis = 1)
                out_mu_test = pd.concat([out_mu_test, out_mu_test2], axis = 1)
                out_sigma_test = pd.concat([out_sigma_test, out_sigma_test2], axis = 1)
            i = i +1
    

        for i in range(0, n_samples_batch):
            if i == 0:
                samples = pd.DataFrame(sample_from_mixture_f(X_pred, np.array(out_pi_test), np.array(out_mu_test), np.array(out_sigma_test), self.dist))
                samples.columns = ['sample_' + str(i+1)]
                print(str(i+1) + '... /'+ str(n_samples_batch))
            else:
                samples2 = pd.DataFrame(sample_from_mixture_f(X_pred, np.array(out_pi_test), np.array(out_mu_test), np.array(out_sigma_test), self.dist))
                samples2.columns = ['sample_' + str(i+1)]
                samples = pd.concat([samples, samples2], axis = 1)
            if (i+1)%100 == 0:
                print(str(i+1) + '... /'+ str(n_samples_batch))
                
        return samples
    
    def plot_pred_fit(self, y_pred, y_true, y_scaler = None):
        if y_scaler != None:
            y_true = y_scaler.inverse_transform(y_true)
        else:
            pass
        
        all_preds = pd.DataFrame(y_true)
        all_preds.columns = ['Y_TRUE']
        all_preds['Y_PRED_MEAN'] = y_pred['Y_PRED_MEAN'] #y_pred['Y_PRED_MEAN']
        all_preds['Y_PRED_LOWER'] =  y_pred['Y_PRED_LOWER'] #y_pred['Y_PRED_LOWER']
        all_preds['Y_PRED_UPPER'] =  y_pred['Y_PRED_UPPER'] #y_pred['Y_PRED_UPPER']
        
        all_preds = all_preds.sort_values(by = ['Y_TRUE'])
        all_preds = all_preds.reset_index(drop = True)
        
        stat_r2 = r2_score(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        stat_rmse = math.sqrt(mean_squared_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN']))
        stat_mae = mean_absolute_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        
        plt.scatter(all_preds.index, all_preds['Y_TRUE'], label = 'Y_TRUE')
        plt.scatter(all_preds.index, all_preds['Y_PRED_MEAN'], label = 'Y_PRED_MEAN')
        plt.fill_between(all_preds.index, all_preds['Y_PRED_LOWER'], all_preds['Y_PRED_UPPER'], alpha = 0.4, label = 'PREDICTION_INTERVAL')
        plt.legend()
        plt.title('Y_TRUE vs Y_PRED : FIT' + '\n' + 'R2:' + str(round(stat_r2, 4)) + '\n' + 'RMSE:' + str(round(stat_rmse, 2)) + '\n' + 'MAE:' + str(round(stat_mae, 2)))
    
    def plot_pred_vs_true(self, y_pred, y_true, y_scaler = None):
        if y_scaler != None:
            y_true = y_scaler.inverse_transform(y_true)
        else:
            pass
        
        all_preds = pd.DataFrame(y_true)
        all_preds.columns = ['Y_TRUE']
        all_preds['Y_PRED_MEAN'] = y_pred['Y_PRED_MEAN'] #y_pred['Y_PRED_MEAN']
        all_preds['Y_PRED_LOWER'] =  y_pred['Y_PRED_LOWER'] #y_pred['Y_PRED_LOWER']
        all_preds['Y_PRED_UPPER'] =  y_pred['Y_PRED_UPPER'] #y_pred['Y_PRED_UPPER']
        
        all_preds = all_preds.sort_values(by = ['Y_TRUE'])
        all_preds = all_preds.reset_index(drop = True)
        
        stat_r2 = r2_score(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        stat_rmse = math.sqrt(mean_squared_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN']))
        stat_mae = mean_absolute_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        
        
        
        plt.scatter(y_pred['Y_PRED_MEAN'], y_true)
        plt.title('Y_TRUE vs Y_PRED : LINEAR RELATION' + '\n' + 'R2:' + str(round(stat_r2, 4)) + '\n' + 'RMSE:' + str(round(stat_rmse, 2)) + '\n' + 'MAE:' + str(round(stat_mae, 2)))
     
    def plot_distribution_fit(self, n_samples_batch = 1, alpha = 0.2, y_scaler = None):
        
        X_pred = self._X.copy()

        y_sampled = self.sample_from_mixture(X_pred = X_pred, n_samples_batch = n_samples_batch, y_scaler = y_scaler)
        
        for col in y_sampled.columns:
            sns.kdeplot(y_sampled[col], shade=True, alpha = alpha)      
        sns.kdeplot(self._y.ravel(), shade=True, linewidth = 2.5, label = 'True dist')
    
    def plot_all_distribution_fit(self, n_samples_batch = 1, alpha = 0.2, y_scaler = None):
        
        X_pred = self._X.copy()
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        all_preds = self.predict_dist(X_pred, q = 0.95, y_scaler = None)

        i = 0
        for elem in all_preds:
            sns.kdeplot(elem['Y_PRED_MEAN'], shade=True, alpha = 0.15, label = 'fitted_mixture_' + str(i))      
            i = i +1
        sns.kdeplot(self._y.ravel(), shade=True, label = 'True dist')
        
    def plot_samples_vs_true(self, X_pred, y_pred, alpha = 0.4, non_linear = False, y_scaler = None):
       
        #X_pred = self._X
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        
        y_sampled = self.sample_from_mixture(X_pred = X_pred, n_samples_batch = 1, y_scaler = y_scaler)
        

        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()
        
        plt.scatter(X_1d, y_sampled, alpha = alpha, label = 'Generated sample')
        plt.scatter(X_1d, y_pred, alpha = alpha, label = 'True')
        plt.legend()
        
    def plot_predict_best(self, X_pred, q = None, y_scaler = None, non_linear = False):
        
        if q == None:
            q_d  = 0.95
        else:
            q_d = q
        
        y_pred = self.predict_best(X_pred = X_pred, q = q_d, y_scaler = y_scaler)
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()
            
        #plt.scatter(X_1d, y, alpha = 0.3)
        #plt.scatter(X_1d_test['X'], X_1d_test['Y'], s = 25, alpha = 0.6)
        plt.scatter(X_1d, y_pred['Y_PRED_MEAN'], c = 'blue', alpha = 0.2, label = 'y_pred')
        if q != None:
            plt.scatter(X_1d, y_pred['Y_PRED_LOWER'], c = 'blue', alpha = 0.2)
            plt.scatter(X_1d, y_pred['Y_PRED_UPPER'], c = 'blue', alpha = 0.2)
        plt.legend()
        #stat_rmse = math.sqrt(mean_squared_error(y, y_pred['Y_PRED_MEAN']))
        self.X_1d = X_1d
    
    def plot_predict_mixed(self, X_pred, q = None, y_scaler = None, non_linear = False):
        
        if q == None:
            q_d  = 0.95
        else:
            q_d = q
        
        y_pred = self.predict_mixed(X_pred = X_pred, q = q_d, y_scaler = y_scaler)
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()
            
        
        #plt.scatter(X_1d, y, alpha = 0.3)
        #plt.scatter(X_1d_test['X'], X_1d_test['Y'], s = 25, alpha = 0.6)
        plt.scatter(X_1d, y_pred['Y_PRED_MEAN'], c = 'blue', alpha = 0.2, label = 'y_pred', s = 10)
        if q != None:
            plt.scatter(X_1d, y_pred['Y_PRED_LOWER'], c = 'blue', alpha = 0.2, s = 10)
            plt.scatter(X_1d, y_pred['Y_PRED_UPPER'], c = 'blue', alpha = 0.2, s = 10)
        plt.legend()
        #stat_rmse = math.sqrt(mean_squared_error(y, y_pred['Y_PRED_MEAN']))
        self.X_1d = X_1d
        
    def plot_predict_with_overlaps(self, X_pred, q=None, y_scaler = None, pi_threshold = 0.215, non_linear = False):
        
        if q == None:
            q_d  = 0.95
        else:
            q_d = q
        
        y_pred = self.predict_with_overlaps(X_pred = X_pred, q = q_d, y_scaler = y_scaler, pi_threshold = pi_threshold)
        
        X_cols = y_pred.loc[:, y_pred.columns.str.startswith('X')].columns.tolist()
        
        X_pred2 = y_pred[X_cols]
        if X_pred2.shape[1] > 1:
            if non_linear == False:
                X_1d2 = PCA(n_components = 1).fit_transform(X_pred2)
            else:
                X_1d2 = UMAP(n_components = 1).fit_transform(X_pred2)
        else:
            X_1d2 = X_pred2.copy()
            
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()
            
        #plt.scatter(X_1d, self._y, alpha = 0.3)
        #plt.scatter(X_1d_test['X'], X_1d_test['Y'], s = 25, alpha = 0.6)
        plt.scatter(X_1d2, y_pred['Y_PRED_MEAN'], c = 'blue', alpha = 0.2, label = 'y_pred')
        if q != None:
            plt.scatter(X_1d2, y_pred['Y_PRED_LOWER'], c = 'blue', alpha = 0.2)
            plt.scatter(X_1d2, y_pred['Y_PRED_UPPER'], c = 'blue', alpha = 0.2)
        plt.legend()
        #stat_rmse = math.sqrt(mean_squared_error(y, y_pred['Y_PRED_MEAN']))
        self.X_1d2 = X_1d2
        self.X_1d = X_1d

    def plot_predict_dist(self, X_pred, q = 0.95, y_scaler = None, non_linear = False, with_weights = True, size = 400):
        all_preds = self.predict_dist(X_pred, q = 0.95, y_scaler = None)
        
        if self.gmm_boost == True:
            clusters = self._clusterer.predict_proba(X_pred)
            X_pred = np.concatenate((X_pred, clusters), axis = 1)
        else:
            pass
        
        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()
        
        if with_weights == False:
            color=iter(cm.rainbow(np.linspace(0,1,self.n_mixtures)))
            i = 0
            for elem in all_preds:
                c=next(color)
                plt.scatter(X_1d, elem['Y_PRED_MEAN'], s = 10, alpha = 0.2, c = c, label = 'Mixture_' + str(i))
                plt.scatter(X_1d, elem['Y_PRED_LOWER'], s = 10, alpha = 0.3, c = c)
                plt.scatter(X_1d, elem['Y_PRED_UPPER'], s = 10, alpha = 0.3, c = c)
                i = i +1
            #plt.scatter(X_1d, self._y, s = 20, alpha = 0.2)
            plt.legend()
        else:
            i = 0
            for elem in all_preds:
                
                if i == 0:
                    out_pi_test = elem[['DIST_PI']]
                    out_mu_test = elem[['DIST_MU']]
                    out_sigma_test = elem[['DIST_VAR']]
                else:
                    out_pi_test2 = elem[['DIST_PI']]
                    out_mu_test2 = elem[['DIST_MU']]
                    out_sigma_test2 = elem[['DIST_VAR']]
                    out_pi_test = pd.concat([out_pi_test, out_pi_test2], axis = 1)
                    out_mu_test = pd.concat([out_mu_test, out_mu_test2], axis = 1)
                    out_sigma_test = pd.concat([out_sigma_test, out_sigma_test2], axis = 1)
                i = i +1
                
            n_mixtures = self.n_mixtures

            # Let's plot the variances and weightings of the means as well.
            #fig = plt.figure(figsize=(8, 8))
            #ax1 = fig.add_subplot(111)
            # ax1.scatter(data[0], data[1], marker='o', c='b', s=data[2], label='the data')
            #ax1.scatter(X,y,marker='o', c='black', alpha=0.1)
            color=iter(cm.rainbow(np.linspace(0,1,self.n_mixtures)))
            for i in range(n_mixtures):
                c=next(color)
                plt.scatter(X_1d, out_mu_test.iloc[:,i], marker='o', s=size*out_sigma_test.iloc[:,i]*out_pi_test.iloc[:,i],alpha=0.3, label = 'MIXTURE_' + str(i), c = c)
            plt.legend()

        self.X_1d = X_1d