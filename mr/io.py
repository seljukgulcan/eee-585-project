"""
Input-ouput utility functions
"""

import csv

import numpy as np


def read_movielens_input(filename):
    """
    Read movielens rating file as Nx3 np float array

    :param filename: Input filename
    :return: np array in (N, 3) shape
    """
    def iter_1d_nnz_file():
        with open(filename) as file:
            reader = csv.reader(file)
            next(reader)  # skip the header
            for row in reader:
                yield row[0]
                yield row[1]
                yield row[2]

    return np.fromiter(iter_1d_nnz_file(), float).reshape(-1, 3)
