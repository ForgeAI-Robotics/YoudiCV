from itertools import product
import warnings

import numpy as np
import pytest

from piper_handeye.depth_matrix_safety import downsample_depth_matrix


def reference(depth, factor):
    x = np.asarray(depth, dtype=float)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    n, h, w = x.shape
    blocks = x[:, :h//factor*factor, :w//factor*factor].reshape(
        n, h//factor, factor, w//factor, factor)
    valid = np.where((blocks > .08) & (blocks < 5), blocks, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.nanmedian(valid, axis=(2, 4))
    return out[0] if squeeze else out


def test_every_combination_of_validity_boundaries_matches_exactly():
    values = [-1., 0., .08, .08000001, .15, .8, 4.999999, 5., np.nan, np.inf]
    x = np.array(list(product(values, repeat=4))).reshape(-1, 2, 2)
    np.testing.assert_array_equal(downsample_depth_matrix(x, 2, reducer='median'), reference(x, 2))


@pytest.mark.parametrize('shape', [(7, 9), (3, 17, 23), (2, 1, 8), (2, 8, 1)])
@pytest.mark.parametrize('factor', [1, 2, 3])
def test_cropping_noncontiguous_input_and_fallback(shape, factor):
    x = np.random.default_rng(42).uniform(-1, 6, shape)[..., ::-1]
    x[..., ::3] = np.nan
    np.testing.assert_array_equal(downsample_depth_matrix(x, factor, reducer='median'), reference(x, factor))
