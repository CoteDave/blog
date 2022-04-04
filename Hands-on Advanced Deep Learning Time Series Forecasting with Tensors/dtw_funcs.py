import numpy as np
from numba import jit

from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score

#from utils.progbar import progress_bar

try:
    from IPython.core.display import HTML, Javascript, display
except ImportError:
    pass


__all__ = ['dtw_distance', 'KnnDTW']


def slow_dtw_distance(series1, series2):
    """
        Returns the DTW similarity distance between two 1-D
        timeseries numpy arrays.
        Arguments:
            series1, series2 : array of shape [n_timepoints]
                Two arrays containing n_samples of timeseries data
                whose DTW distance between each sample of A and B
                will be compared
        Returns:
            DTW distance between sequence 1 and 2
        """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    E[0][0] = np.square(series1[0] - series2[0])

    # Fill First Column
    for i in range(1, l1):
        E[i][0] = E[i - 1][0] + np.square(series1[i] - series2[0])

    # Fill First Row
    for i in range(1, l2):
        E[0][i] = E[0][i - 1] + np.square(series1[0] - series2[i])

    for i in range(1, l1):
        for j in range(1, l2):
            v = np.square(series1[i] - series2[j])

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])


@jit(nopython=True)
def dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.
    Arguments:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
    Returns:
        DTW distance between sequence 1 and 2
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    E[0][0] = np.square(series1[0] - series2[0])

    # Fill First Column
    for i in range(1, l1):
        E[i][0] = E[i - 1][0] + np.square(series1[i] - series2[0])

    # Fill First Row
    for i in range(1, l2):
        E[0][i] = E[0][i - 1] + np.square(series1[0] - series2[i])

    for i in range(1, l1):
        for j in range(1, l2):
            v = np.square(series1[i] - series2[j])

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])


# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class KnnDTW(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    Arguments
    ---------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for KNN
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        """Fit the model using x as training data and y as class labels
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
        y : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = np.copy(x)
        self.y = np.copy(y)

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        y : array of shape [n_samples, n_timepoints]
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            #p = progress_bar(np.shape(dm)[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = dtw_distance(x[i], y[j])

                    dm_count += 1
                    #p.update(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            #p = progress_bar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = dtw_distance(x[i], y[j])
                    # Update progress bar
                    dm_count += 1
                    #p.update(dm_count)

            return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data
        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """
        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.y[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

    def evaluate(self, x, y):
        """
        Predict the class labels or probability estimates for
        the provided data and then evaluates the accuracy score.
        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          y : array of shape [n_samples]
              Array containing the labels of the testing dataset to be classified
        Returns
        -------
          1 floating point value representing the accuracy of the classifier
        """
        # Predict the labels and the probabilities
        pred_labels, pred_probas = self.predict(x)

        # Ensure labels are integers
        y = y.astype('int32')
        pred_labels = pred_labels.astype('int32')

        # Compute accuracy measure
        accuracy = accuracy_score(y, pred_labels)
        return accuracy

    def predict_proba(self, x):
        """Predict the class labels probability estimates for
        the provided data
        Arguments
        ---------
            x : array of shape [n_samples, n_timepoints]
                Array containing the testing data set to be classified
        Returns
        -------
            2 arrays representing:
                (1) the predicted class probabilities
                (2) the knn labels
        """
        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        # Invert the distance matrix
        dm = -dm

        # Compute softmax probabilities
        dm_exp = np.exp(dm - dm.max())
        dm = dm_exp / np.sum(dm_exp, axis=-1, keepdims=True)

        classes = np.unique(self.y)
        class_dm = []

        # Partition distance matrix by class
        for i, cls in enumerate(classes):
            idx = np.argwhere(self.y == cls)[:, 0]
            cls_dm = dm[:, idx]  # [N_test, N_train_c]

            # Take maximum distance vector due to softmax probabilities
            cls_dm = np.max(cls_dm, axis=-1)  # [N_test,]

            class_dm.append([cls_dm])

        # Concatenate the classwise distance matrices and transpose
        class_dm = np.concatenate(class_dm, axis=0)  # [C, N_test]
        class_dm = class_dm.transpose()  # [N_test, C]

        # Compute softmax probabilities
        class_dm_exp = np.exp(class_dm - class_dm.max())
        class_dm = class_dm_exp / np.sum(class_dm_exp, axis=-1, keepdims=True)

        probabilities = class_dm
        knn_labels = np.argmax(probabilities, axis=-1)

        return probabilities, knn_labels