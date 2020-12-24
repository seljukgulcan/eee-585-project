import numpy as np

from mr.utils import compute_rmse

class FunkSVD:
    def __init__(self, row_count, col_count, factor_size, iteration_count, alpha, beta):
        """
        FunkSVD model
        :param row_count: Row count
        :param col_count: Column count
        :param factor_size: latent factor size
        :param iteration_count: Number of iterations
        :param alpha: Learning rate
        :param beta: Regularization parameter
        """
        self.row_count = row_count
        self.col_count = col_count
        self.factor_size = factor_size

        self.P = np.random.uniform(size=(row_count, factor_size))
        self.Q = np.random.uniform(size=(col_count, factor_size))

        self.iteration_count = iteration_count
        self.alpha = alpha
        self.beta = beta

        self._train_loss = None
        self._valid_loss = None

    def fit(self, X, y, validation_tuple=None):
        """
        X is Nx2,fac
        :param X: two dimensional np array where each line is [row_index, column_index]
        :param y: one dimensional np array
        :param validation_tuple: If (X_valid, y_valid) is given, the model calculates training error and validation
        error in each iteration and stores related information in _train_loss and _valid_loss attributes
        """

        is_compute_error = False
        if validation_tuple is not None:
            is_compute_error = True
            X_valid, y_valid = validation_tuple
            self._train_loss = []
            self._valid_loss = []

        if is_compute_error:
            y_predict = self.predict(X)
            self._train_loss.append(compute_rmse(y, y_predict))

            y_predict = self.predict(X_valid)
            self._valid_loss.append(compute_rmse(y_valid, y_predict))

        for i in range(self.iteration_count):
            N = X.shape[0]
            ind = np.arange(N)
            np.random.shuffle(ind)
            X = X[ind]
            y = y[ind]

            for (row, col), rating in zip(X, y):
                estimate = self.P[row].dot(self.Q[col])
                error = rating - estimate

                temp = np.copy(self.P[row])
                self.P[row] += self.alpha * (error * self.Q[col] - self.beta * self.P[row])
                self.Q[col] += self.alpha * (error * temp - self.beta * self.Q[col])

            if is_compute_error:
                y_predict = self.predict(X)
                self._train_loss.append(compute_rmse(y, y_predict))

                y_predict = self.predict(X_valid)
                self._valid_loss.append(compute_rmse(y_valid, y_predict))

        if is_compute_error:
            self._train_loss = np.array(self._train_loss)
            self._valid_loss = np.array(self._valid_loss)

    def predict(self, X):
        """
        :param X: Nx2
        :return: y_predict
        """
        y_predict = []
        for (row, col) in X:
            estimate = self.P[row].dot(self.Q[col])
            y_predict.append(estimate)

        return np.array(y_predict)
