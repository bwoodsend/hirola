# -*- coding: utf-8 -*-
"""
"""
import functools
import warnings
from contextlib import contextmanager

import numpy as np

from hirola import exceptions


def random_ids(max, count, at_least_once=True, sort=False):
    if at_least_once:
        assert count >= max
        out = np.append(np.arange(max, dtype=np.intp),
                        np.random.randint(0, max, count - max, np.intp))
    else:
        out = np.random.randint(0, max, count, np.intp)
    if sort:
        out.sort()
    else:
        np.random.shuffle(out)
    return out


def ignore_almost_full_warnings(test):
    """Decorate a test to disable exceptions.AlmostFull warnings."""

    @functools.wraps(test)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=exceptions.AlmostFull)
            test(*args, **kwargs)

    return wrapped


@contextmanager
def warnings_as_errors():
    """A context manager which treats all warnings as errors."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        yield
