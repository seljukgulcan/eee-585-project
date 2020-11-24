"""
Statistics related helper functions
"""

import math


def fit_normal_dist(lst):
    """
    Fits a normal distribution to given sequence
    :param lst: sequence of numbers
    :return: tuple of (mean, standard deviation)
    """
    count = len(lst)
    total = sum(lst)
    mean = total / count

    var = 0
    for no in lst:
        var += (no - mean) ** 2
    var /= count

    std = math.sqrt(var)

    return mean, std


def pdf_normal(x, mean=0.0, std=1.0):
    """
    Probability density function of normal distribution
    :param x: x
    :param mean: mean
    :param std: standard deviation
    :return: y = N(x, mean, std)
    """
    var = std ** 2
    return math.e ** (-1 * ((x - mean) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)
