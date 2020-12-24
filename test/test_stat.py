import pytest

import numpy as np

from mr.stat import *


def test_pdf_normal():

    x = np.array([0, 1, 2, 3])
    y = pdf_normal(x, mean=0, std=2)

    y_expected = [0.199471, 0.176033, 0.120985, 0.0647588]
    assert pytest.approx(y, rel=1e-3) == y_expected


def test_fit_normal_dist():
    y = [0, 0, 0, 2, 2, 2]

    mean, std = fit_normal_dist(y)
    assert mean == 1
    assert std == 1

    y = list(range(11))

    mean, std = fit_normal_dist(y)
    assert mean == 5
    assert pytest.approx(std ** 2) == 10
