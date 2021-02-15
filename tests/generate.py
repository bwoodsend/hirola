# -*- coding: utf-8 -*-
"""
Test data generators for hash table keys.
"""

import numpy as np


def random(n, d=16) -> np.ndarray:
    """Random 32-bit integers utilising the full range."""
    bin = np.random.bytes(n * d)
    return np.frombuffer(bin, np.dtype(np.int32)).reshape((n, -1))


def id_like(n, d=2) -> np.ndarray:
    """Limited range integers. Roughly emulates mesh vertex indices or similar
    enumerations.
    """
    data = np.empty((n, d), dtype=np.int32)
    data[:, 0] = np.arange(n) * 2 // 3

    for i in range(1, d):
        data[:, i] = data[:, i - 1] + np.random.randint(-100, 100, n)

    return data.clip(min=0, max=data[-1, 0]).astype(np.int32)


def permutative(n) -> np.ndarray:
    """All possible pairs of integers from (0, 0) to (sqrt(n), sqrt(n))."""
    return np.c_[np.divmod(np.arange(n), int(np.sqrt(n)))].astype(np.int32)


def floating_32(n) -> np.ndarray:
    return floating_64(n).astype(np.float32)


def floating_64(n) -> np.ndarray:
    return np.random.uniform(-30, 30, (n, 3))


generators = [
    random,
    id_like,
    permutative,
    floating_32,
    floating_64,
]


def pysafe(x):
    """Convert an array into an array of elements which are hashable by Python's
     built in hash()."""
    void = np.void(x.size * x.itemsize // len(x))
    return np.frombuffer(x.tobytes(), void).astype(object)
