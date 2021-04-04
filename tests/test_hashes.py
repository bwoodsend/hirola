# -*- coding: utf-8 -*-
"""
Test the effectiveness of each hash function.

Hash functions should rarely give *hash collisions* (the same output for
different inputs). Count how often collisions occur.

This depends greatly on the data you give it. Floating point and text are
generally quite easy. Narrow ranged integers using large integer types are the
hardest.

All count thresholds in this file are completely arbitrary - they could be set
to anything. This file is more of a benchmark than a test but still aims to
assert that:

- No input types perform drastically worse than others.
- Any future improvements to a hash function or to collision handling for one
  data type doesn't degrade performance for any other.

If we do end up in a whack-a-mole situation where a change significantly
improves some cases but worsens others then we should probably consider making
the change configurable.

"""

import ctypes
import statistics

import numpy as np
import pytest

from hirola._hash_table import slug, ptr, choose_hash, vectorise_hash, HashTable
from tests import generate as gen


def test_vectorised_hash():
    hash = slug.dll.hash
    keys = gen.random(10)
    key_size = keys.dtype.itemsize * keys.shape[1]
    hashes = np.array([hash(ptr(i), key_size) for i in keys])
    hashes_ = vectorise_hash(hash, key_size, keys)
    assert np.array_equal(hashes_, hashes)


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
    c = counts_with(generate, table_size, dtype)
    collisions = (c - 1).clip(min=0)

    # Ideally we want to drive this threshold as low as possible.
    assert collisions.sum() < 600, ''


def counts_with(generate, table_size: int, dtype):
    np.random.seed(0)
    x = generate(1000).astype(dtype)
    key_size = x.dtype.itemsize * x[0].size
    hash = choose_hash(key_size)

    hashes = vectorise_hash(hash, key_size, x)
    return count(hashes, table_size)


per_test_collisions = {}

@pytest.mark.parametrize("generate", [gen.random, gen.permutative, gen.id_like])
@pytest.mark.parametrize("table_size", range(1245, 1255))
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_aggregate_collisions(generate, table_size: int, dtype):
    """Test empirically the number of extra steps taken due to hash collisions.
    Unlike test_collisions(), this test take into account clusters of collisions
    which are much worse than the same number of collisions but spaced out.
    """
    if not hasattr(slug.dll, "collisions"):
        # This test requires collision counting to be enabled at compile time.
        pytest.skip("Requires compiling with CC_FLAGS='-D COUNT_COLLISIONS'.")

    np.random.seed(0)
    x = generate(1000).astype(dtype)
    self = HashTable(table_size, (dtype, x[0].size))

    # Pulling a global variable out of a C library is disproportionately
    # awkward.
    # https://cslug.readthedocs.io/en/stable/globals.html#an-integer
    collisions_ptr = ctypes.cast(slug.dll.collisions,
                                 ctypes.POINTER(ctypes.c_size_t))
    old = collisions_ptr.contents.value
    self.add(x)
    collisions = collisions_ptr.contents.value - old

    assert collisions < 20000
    per_test_collisions[gen, table_size, dtype] = collisions


def test_average_collisions():
    """Generate a summary from test_aggregate_collisions() results."""
    # The pytest-order plugin guarantees that test_aggregate_collisions() will
    # be run first simply because it is written first in this file.
    if not per_test_collisions:
        pytest.skip("No data to work with.")
    assert statistics.mean(per_test_collisions.values()) < 2500
