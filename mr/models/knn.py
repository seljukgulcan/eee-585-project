import math

import numpy as np
from scipy.sparse import coo_matrix


class KNN:
    def __init__(self, row_count, col_count, K, prediction_mode='simple_average'):
        """
        User-user collaborative filtering model
        :param row_count: Row count
        :param col_count: Column count
        :param K: The number of nearest neighbor to consider
        :param prediction_mode: 'all', 'simple_average', 'weighted_average', 'baseline_average'
        """
        self.row_count = row_count
        self.col_count = col_count
        self.K = K
        self.prediction_mode = prediction_mode

        self.X_train_coo = None
        self.X_train_csr = None
        self.row_deviation = None
        self.col_deviation = None
        self.overall_mean = None

    def _pearson_sparse(self, x, y):
        x_mean = x.sum() / x.getnnz()
        y_mean = y.sum() / y.getnnz()

        x = x.astype(float)
        y = y.astype(float)

        ind = x.nonzero()
        x[ind] -= x_mean

        ind = y.nonzero()
        y[ind] -= y_mean

        top = x.dot(y.T)[0, 0]
        bottom = math.sqrt(x.dot(x.T)[0, 0] * y.dot(y.T)[0, 0])

        if bottom == 0:
            return 0

        return top / bottom

    def fit(self, X, y):
        """
        X is Nx2,
        :param X: two dimensional np array where each line is [row_index, column_index]
        :param y: one dimensional np array
        """

        self.X_train_coo = coo_matrix((y, (X[:, 0], X[:, 1])),
                                      shape=(self.row_count, self.col_count))

        self.X_train_csr = self.X_train_coo.tocsr()

        overall_mean = self.X_train_csr.sum() / self.X_train_csr.getnnz()
        col_deviation = self.X_train_csr.sum(axis=0).A1 / self.X_train_csr.getnnz(axis=0) - overall_mean
        self.col_deviation = np.nan_to_num(col_deviation, copy=False, nan=0)
        row_deviation = self.X_train_csr.sum(axis=1).A1 / self.X_train_csr.getnnz(axis=1) - overall_mean
        self.row_deviation = np.nan_to_num(row_deviation, copy=False, nan=0)
        self.overall_mean = overall_mean

    def predict(self, X):
        """
        Warning: X should be sorted on X[:, 0] then X[:, 1]
        :param X: Nx2
        :return: y_predict
        """

        simple_average_predict_lst = []
        weighted_average_predict_lst = []
        baseline_average_predict_lst = []

        for i in range(self.row_count):
            sim_vect = np.zeros(self.row_count)

            row_i = self.X_train_csr[i, :]

            for j in range(self.row_count):
                row_j = self.X_train_csr[j, :]
                sim_vect[j] = self._pearson_sparse(row_i, row_j)

            # Go over test ratigns of user i
            # test, col_lst = M_test_csr[i, :].nonzero()
            col_lst = X[X[:, 0] == i][:, 1]

            for col in col_lst:

                # Find K-neighbors
                row_lst, temp = self.X_train_csr[:, col].nonzero()

                # Get max K values
                # We're using this solution to find max K values in linear time:
                # https://stackoverflow.com/a/23734295/9320666

                if len(row_lst) < self.K:
                    neighbor_lst = row_lst
                else:
                    ind = np.argpartition(sim_vect[row_lst], -self.K)[-self.K:]
                    neighbor_lst = row_lst[ind]
                neighbor_sim_lst = sim_vect[neighbor_lst]

                # Estimate
                baseline_estimate = self.overall_mean + self.row_deviation[i] + self.col_deviation[col]

                # Simple Average
                rating_estimate = 0
                for neighbor in neighbor_lst:
                    rating_estimate += self.X_train_csr[neighbor, col]

                if len(neighbor_lst) > 0:
                    rating_estimate /= len(neighbor_lst)
                else:
                    rating_estimate = baseline_estimate

                simple_average_predict_lst.append(rating_estimate)

                # Weighted Average
                rating_estimate = 0
                weight_sum = 0
                for neighbor, weight in zip(neighbor_lst, neighbor_sim_lst):
                    if weight <= 0:
                        continue
                    rating_estimate += self.X_train_csr[neighbor, col] * weight
                    weight_sum += weight

                if weight_sum == 0:
                    rating_estimate = baseline_estimate
                else:
                    rating_estimate /= weight_sum

                weighted_average_predict_lst.append(rating_estimate)

                # Row-Col Average Considered Weighted Average
                rating_estimate = 0
                weight_sum = 0

                for neighbor, weight in zip(neighbor_lst, neighbor_sim_lst):
                    if weight <= 0:
                        continue
                    rating_estimate += (self.X_train_csr[neighbor, col] -
                                        (self.overall_mean + self.row_deviation[neighbor] + self.col_deviation[
                                            col])) * weight
                    weight_sum += weight

                if weight_sum == 0:
                    rating_estimate = baseline_estimate
                else:
                    rating_estimate = baseline_estimate + (rating_estimate / weight_sum)

                baseline_average_predict_lst.append(rating_estimate)

        if self.prediction_mode == 'simple_average':
            return np.array(simple_average_predict_lst)
        elif self.prediction_mode == 'weighted_average':
            return np.array(weighted_average_predict_lst)
        elif self.prediction_mode == 'baseline_average':
            return np.array(baseline_average_predict_lst)
        elif self.prediction_mode == 'all':
            retval = list()

            retval.append(np.array(simple_average_predict_lst))
            retval.append(np.array(weighted_average_predict_lst))
            retval.append(np.array(baseline_average_predict_lst))

            return retval

        raise ValueError('Unexpected predtion_mode: {}'.format(self.prediction_mode))
