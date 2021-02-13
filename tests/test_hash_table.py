# -*- coding: utf-8 -*-
"""
"""

import pytest
import numpy as np
from cslug import ptr

from hoatzin import HashTable, exceptions
from hoatzin._hash_table import slug

from tests import random_ids

DATA = np.arange(120, dtype=np.int8).data
DTYPES = [
    np.int16,
    np.float64,
    np.dtype([("vertex", np.float32, 5)]),
    np.dtype(np.bytes_) * 3,
    np.dtype(np.bytes_) * 15,
]


def test_modulo():
    for i in range(20, -20, -1):
        assert slug.dll.euclidean_modulo(i, 5) == i % 5


def test_hash():
    x = np.array([123, 4234, 213], dtype=np.uint32)
    assert slug.dll.hash(ptr(x), 12) == np.bitwise_xor.reduce(x * 0x0B070503)


def test_walk_through():
    data = np.array([100, 101, 100, 103, 104, 105, 103, 107], dtype=np.float32)
    self = HashTable(5, dtype=data.dtype)

    assert self.dtype == data.dtype
    assert np.all(self._hash_owners == -1)
    assert self.key_size == 4
    assert self.length == 0
    assert self.max == 5

    hash = slug.dll.hash(ptr(data), self.key_size)
    for i in range(2):
        assert slug.dll.HT_hash_for(self._raw._ptr, ptr(data)) \
               == hash % self.max
        assert self._add(data) == 0
        assert self.length == 1
        assert len(self) == 1
        assert np.array_equal(self.keys, [100])
        assert self._hash_owners[hash % self.max] == 0
        assert self._get(data) == 0

    assert self._add(data[1]) == 1
    assert self._add(data[2]) == 0
    assert self._add(data[3]) == 2
    assert self._add(data[4]) == 3
    assert self._add(data[5]) == 4
    assert self._add(data[6]) == 2
    assert self._add(data[7]) == -1

    assert self.add(data[:7]).tolist() == [0, 1, 0, 2, 3, 4, 2]
    assert self.get(data).tolist() == [0, 1, 0, 2, 3, 4, 2, -1]
    assert self[data].tolist() == [0, 1, 0, 2, 3, 4, 2, -1]

    with pytest.raises(exceptions.HashTableFullError,
                       match=r"element 107\.0 \(index 7\) "):
        self.add(data)

    with pytest.raises(exceptions.HashTableFullError,
                       match=r"element 107\.0 \(index \(1, 3\)\) "):
        self.add(data.reshape((2, 4)))


SHAPES = [(), (10,), (0, 10), (10, 0), (10, 10), (10, 10, 10)]


def test_dtype_normalisation_simple():
    self = HashTable(10, np.int16)
    assert isinstance(self.dtype, np.dtype)
    assert self._base_dtype == np.dtype(np.int16)
    assert self._dtype_shape == ()

    for shape in SHAPES:
        keys, shape_ = self._norm_input_keys(np.empty(shape, dtype=np.int16))
        assert shape_ == shape
    assert self._norm_input_keys(np.empty(10, dtype=np.int16))[1] == (10,)

    with pytest.raises(TypeError, match="Expecting int16 but got float64."):
        self.get(np.arange(10, dtype=np.float64))


def test_dtype_normalisation_multidimensional():
    self = HashTable(10, np.dtype(np.float32) * 3)
    assert self.key_size == 12
    assert self._base_dtype == np.dtype(np.float32)
    assert self._dtype_shape == (3,)

    with pytest.raises(TypeError):
        self._norm_input_keys(np.empty(10, np.int32))
    with pytest.raises(ValueError):
        self._norm_input_keys(np.empty(10, np.float32))
    with pytest.raises(ValueError):
        self._norm_input_keys(np.empty((10, 4), np.float32))

    assert self._norm_input_keys(np.empty(3, np.float32))[1] == ()

    for shape in SHAPES:
        _, shape_ = self._norm_input_keys(np.empty(shape + (3,), np.float32))
        assert shape_ == shape


def test_dtype_normalisation_records():
    dtype = np.dtype([("a", np.int16, 4), ("b", np.uint64)])
    self = HashTable(10, dtype)
    assert self._base_dtype == dtype
    assert self._dtype_shape == ()
    assert self.key_size == 16


def test_invalid_array():
    with pytest.raises(TypeError):
        HashTable(10, object)

    assert HashTable(0, int).max == 1
    assert HashTable(-10, int).max == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test(dtype, sort, batch):

    unique_ = np.unique(np.frombuffer(DATA, dtype=dtype))
    ids_ = random_ids(len(unique_), 100, at_least_once=True, sort=sort)

    values = unique_[ids_]

    self = HashTable(len(values), dtype)

    if batch:
        assert np.all(self.get(values) == -1)

    if batch:
        ids = self.add(values)
    else:
        ids = np.empty_like(ids_)
        for (i, value) in enumerate(values):
            value = np.array(value)
            ids[i] = self._add(value)

    assert np.all(self.keys[ids] == values)

    if sort:
        assert np.all(self.keys == unique_)
        assert np.all(ids == ids_)

    for (i, value) in enumerate(values):
        assert self._get(value) == ids[i]

    assert np.all(self.get(values) == ids)
    assert np.all(self.add(values) == ids)


def test_destroy():
    self = HashTable(10, float)
    self.add([.3, .5, .8])

    # Release self.keys so that it can be written to.
    keys = self.destroy()
    assert keys.flags.writeable
    assert np.shares_memory(keys, self.keys)

    # destroy() should be re-callable without complaint (although it's now
    # functionless).
    assert np.shares_memory(keys, self.destroy())

    # Now that self.keys has been made accessibly writeable, it is no longer
    # safe to use the table.
    with pytest.raises(exceptions.HashTableDestroyed, match=".*"):
        self.add(.8)
    with pytest.raises(exceptions.HashTableDestroyed):
        self.get(.5)
