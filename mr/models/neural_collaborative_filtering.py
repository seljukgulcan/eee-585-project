import numpy as np

from mr.utils import compute_rmse


class NCF:
    """
    TODO: fill
    """

    def __init__(self, row_count, col_count, factor_size, layers, iteration_count, lr=0.001):
        """
        TODO: fill
        :param factor_size:
        :param row_count:
        :param col_count:
        :param layers:
        :param iteration_count:
        :param lr:
        """
        # layers[0] should be 2 * factor_size
        # layers is a tuple of ints (3, 5) means 3 input nodes, 5 hidden nodes
        self.weight_lst = []
        self.bias_lst = []
        self.iteration_count = iteration_count
        self.lr = lr
        self.factor_size = factor_size
        self.layer_count = len(layers)

        self.P = np.random.uniform(size=(row_count, factor_size))
        self.Q = np.random.uniform(size=(col_count, factor_size))

        left_node_size = layers[0]
        for right_node_size in layers[1:]:
            self.weight_lst.append(np.random.randn(right_node_size, left_node_size))
            self.bias_lst.append(np.random.randn(right_node_size))
            left_node_size = right_node_size

        self.weight_lst.append(np.random.randn(1, left_node_size))
        self.bias_lst.append(np.random.randn(1))

        self._train_loss = None
        self._valid_loss = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

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

            # Shuffle the training samples
            ind = np.arange(X.shape[0])
            np.random.shuffle(ind)
            X = X[ind]
            y = y[ind]

            self._train_epoch(X, y)

            if is_compute_error:
                y_predict = self.predict(X)
                self._train_loss.append(compute_rmse(y, y_predict))

                y_predict = self.predict(X_valid)
                self._valid_loss.append(compute_rmse(y_valid, y_predict))

    def _train_epoch(self, X, y):

        a_lst = [None for _ in range(self.layer_count)]  # a_k = sigmoid(z_k) where z_k is the output, W_k * a_{k-1}

        for (row, col), y_true in zip(X, y):
            # Feed forward

            a = np.concatenate((self.P[row], self.Q[col]))
            a_lst[0] = a
            for i in range(self.layer_count - 1):
                W = self.weight_lst[i]
                b = self.bias_lst[i]
                z = W @ a + b
                a = self._sigmoid(z)
                a_lst[i + 1] = a

            y_predict = self.weight_lst[-1] @ a + self.bias_lst[-1]

            # Backpropagation
            e = y_predict - y_true
            next_e = e * self.weight_lst[-1]

            gradient = a_lst[-1] * e

            self.weight_lst[-1] -= self.lr * gradient
            self.bias_lst[-1] -= self.lr * e
            e = next_e

            for i in reversed(range(len(self.weight_lst) - 1)):
                next_e = e @ self.weight_lst[i]
                bias_gradient = e * a_lst[i + 1] * (1 - a_lst[i + 1])
                self.bias_lst[i] -= self.lr * bias_gradient.ravel()

                gradient = bias_gradient.T @ a_lst[i].reshape(1, -1)
                self.weight_lst[i] -= self.lr * gradient
                e = next_e

            self.P[row] -= self.lr * e.ravel()[:self.factor_size]
            self.Q[col] -= self.lr * e.ravel()[self.factor_size:]

    def predict(self, X):
        """
        :param X: Nx2
        :return: y_predict
        """
        # X is (N, 3) or (N, 2)

        # Get embeddings
        X = np.hstack((self.P[X[:, 0], :], self.Q[X[:, 1], :]))

        a = X.T
        # hidden layers
        for W, b in zip(self.weight_lst[:-1], self.bias_lst[:-1]):
            z = W @ a + b.reshape(-1, 1)
            a = self._sigmoid(z)

        y_predict = self.weight_lst[-1] @ a + self.bias_lst[-1]
        return y_predict.ravel()
