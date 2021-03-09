# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from hirola._hash_table import slug, ptr, choose_hash, vectorise_hash
from tests import generate as gen


def test_vectorised_hash():
    hash = slug.dll.hash
    keys = gen.random(10)
    key_size = keys.dtype.itemsize * keys.shape[1]
    hashes = np.array([hash(ptr(i), key_size) for i in keys])
    hashes_ = vectorise_hash(hash, key_size, keys)
    assert np.array_equal(hashes_, hashes)


def collisions(hashes: np.ndarray, N: int, cumulative=True):
    """Test effectiveness of a hash function for a given table size.

    Args:
        hashes:
        N:
        cumulative:

    Returns:
        Number of collisions at each hash value. Use ``.sum()`` on the output to
        get an overall score.

    """
    return (count(hashes, N) - 1).clip(min=0)


def count(hashes: np.ndarray, N: int):
    N = int(N)
    hashes = hashes % N
    counts = np.zeros(N, dtype=int)
    np.add.at(counts, hashes, 1)
    return counts


@pytest.mark.parametrize("generate", [gen.random, gen.permutative, gen.id_like])
@pytest.mark.parametrize("table_size", range(1245, 1255))
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_collisions(generate, table_size: int, dtype):
    np.random.seed(0)
    x = generate(1000).astype(dtype)
    key_size = x.dtype.itemsize * x[0].size
    hash = choose_hash(key_size)

    hashes = vectorise_hash(hash, key_size, x)
    c = collisions(hashes, table_size)

    # Ideally we want to drive this threshold as low as possible.
    assert c.sum() < 600, (key_size, c.sum())
