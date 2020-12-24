"""
Contains some helper functions to modify data
"""
import json
import numpy as np


def map_data_to_matrix(data, save=False, filename=None):
    """
    Downloaded data may have gaps between row/column indices
    This function maps row and columns to new indices inplace so that they row_count = max(row_indices) + 1 and
    col_count = max(col_indices + 1
    :param data: np array (Nx3), data is modified inplace
    :param save: whether save mapping info to a json file
    :param filename: filename of the mapping file, used if save is True
    :return: Data in COO matrix format: np.array (Nx3)
    """

    row2new_row = dict()
    col2new_col = dict()

    row_counter = 0
    col_counter = 0
    for i, nnz in enumerate(data):
        row, col, rating = nnz
        row = int(row)
        col = int(col)

        if row not in row2new_row:
            row2new_row[row] = row_counter
            row_counter += 1

        if col not in col2new_col:
            col2new_col[col] = col_counter
            col_counter += 1

        data[i][0] = row2new_row[row]
        data[i][1] = col2new_col[col]

    if save:
        mapping = dict()
        mapping['row'] = row2new_row
        mapping['col'] = col2new_col

        with open('{}_map.json'.format(filename), 'w') as file:
            json.dump(mapping, file)

    return data


def shuffle_split(M, ratio):
    """
    Shuffles the nonzero order of the matrix and split it into two
    :param M: Matrix (Nx3)
    :param ratio: split
    :return: 2 split matrices
    """

    np.random.shuffle(M)
    nnz_count = M.shape[0]

    split_count = round(nnz_count * ratio)

    return M[:split_count, :], M[split_count:, :]


def compute_rmse(y_true, y_predict):
    """
    Computes root mean squared error between true value and predicted value
    :param y_true:
    :param y_predict:
    :return: root mean squared error
    """
    return np.sqrt(((y_true - y_predict) ** 2).mean())


def sort_nonzero_order(data):
    """
    Sort COO matrix first by row index then column index
    :param data: Nx3 np array
    :return: sorted data
    """

    ind = np.lexsort((data[:, 1], data[:, 0]))
    return data[ind]
