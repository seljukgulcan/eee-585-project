import numpy as np


class BaselineModel:
    """
    Baseline recommendation model

    It doesn't use and collaborative information. It predict a missing rating as the mean rating of all training set,
    row mean, column mean or row-column deviation considered mean depending on the prediction mode
    """

    def __init__(self, row_count, col_count, prediction_mode='row_col_mean'):
        """
        :param row_count: Row count
        :param col_count: Column count
        :param prediction_mode: str -> Method of prediction. Possible values:
         - 'mean' : A missing rating is predicted as global mean
         - 'row_mean' : A missing rating is predicted as row mean
         - 'col_mean' : A missing rating is predicted as col mean
         - 'row_col_mean' : A missing rating is predicted by using global mean, row deviation from mean and column
         deviation from mean
        """
        self.row_count = row_count
        self.col_count = col_count
        self.prediction_mode = prediction_mode
        self.mean = None
        self.row2deviation = None
        self.col2deviation = None

    def fit(self, X, y):
        """
        X is Nx2,
        :param X: two dimensional np array where each line is [row_index, column_index]
        :param y: one dimensional np array
        """
        mean = y.mean()
        self.mean = mean

        row2mean = np.zeros(self.row_count)
        row2count = np.zeros(self.row_count)

        col2mean = np.zeros(self.col_count)
        col2count = np.zeros(self.col_count)

        for (row, col), rating in zip(X, y):
            row2count[row] += 1
            row2mean[row] += rating

            col2count[col] += 1
            col2mean[col] += rating

        row2mean[row2count == 0] = mean
        col2mean[col2count == 0] = mean
        row2count[row2count == 0] = 1
        col2count[col2count == 0] = 1

        row2mean /= row2count
        col2mean /= col2count

        self.row2deviation = row2mean - mean
        self.col2deviation = col2mean - mean

    def predict(self, X):
        """
        :param X: Nx2
        :return: y_predict
        """
        y_predict = np.ones(X.shape[0]) * self.mean

        if self.prediction_mode == 'mean':
            return y_predict
        elif self.prediction_mode == 'row_mean':
            for i, (row, col) in enumerate(X):
                y_predict[i] = self.mean + self.row2deviation[row]
        elif self.prediction_mode == 'col_mean':
            for i, (row, col) in enumerate(X):
                y_predict[i] = self.mean + self.col2deviation[col]
        elif self.prediction_mode == 'row_col_mean':
            for i, (row, col) in enumerate(X):
                y_predict[i] = self.mean + self.row2deviation[row] + self.col2deviation[col]
        else:
            raise ValueError('Unexpected prediction mode: {}'.format(self.prediction_mode))

        return y_predict
