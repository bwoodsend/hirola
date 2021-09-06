# -*- coding: utf-8 -*-
"""
"""

import os
import sys
import runpy

import pytest
import numpy as np
from cslug import ptr

from hirola import HashTable, exceptions
from hirola._hash_table import slug

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
    x = np.array([123, 4234, 213], dtype=np.int32)
    out = np.int32(0)
    old = np.seterr(over="ignore")
    for i in range(3):
        out ^= x[i] * np.int32(0x10001)
        out *= np.int32(0x0B070503)
    np.seterr(**old)
    assert slug.dll.hash(ptr(x), 12) == out


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
        assert slug.dll.HT_hash_for(self._raw._ptr, ptr(data), False) \
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
    assert self[data[:-1]].tolist() == [0, 1, 0, 2, 3, 4, 2]

    assert isinstance(self.add(data[0]), int)
    assert isinstance(self.get(data[0]), int)

    with pytest.raises(exceptions.HashTableFullError,
                       match=r".* add keys\[7\] = 107\.0 to .* and 107\.0 is"):
        self.add(data)

    with pytest.raises(exceptions.HashTableFullError,
                       match=r".* add keys\[1, 3\] = 107\.0 to .* and 107\.0 "):
        self.add(data.reshape((2, 4)))

    with pytest.raises(exceptions.HashTableFullError,
                       match=r".* add key = 107\.0 to .* and 107\.0 is"):
        self.add(data[7])


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


def test_string_like():
    self = HashTable(10, "U5")

    # Molly-coddle the types.
    assert self.add(np.array("bear", self.dtype)) == 0
    # Should convert from Python str without complaint.
    assert self.get("bear\x00") == 0
    # Should implicitly add trailing NULLs.
    assert self.get("bear") == 0
    # Should be case sensitive.
    assert self.get("Bear") == -1

    # NumPy implicitly truncates overlong strings. Accept this behaviour.
    assert self._check_dtype("tigers") == "tiger"
    assert self.add(["tigers"]) == 1
    assert self.keys[1] == "tiger"

    # Make absolutely darn certain that hirola's C code never comes into
    # contact with strings of random lengths.
    normed, shape = self._norm_input_keys(["cat", "dog", "hippopotamus"])
    assert normed.dtype == "U5"
    assert shape == (3,)
    assert normed.tolist() == ["cat", "dog", "hippo"]

    # NumPy implicitly converts non-strings to string. Accept this too although
    # is probably a bad idea for floats.
    for (i, key) in enumerate((1, 100, 10000000, .123, 1 / 9), start=len(self)):
        key_ = self._check_dtype(key)
        assert key_ == str(key)[:5]
        assert self.add(key) == i


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


def test_non_int_max():
    max = HashTable(3.5, int).max
    assert isinstance(max, int)
    assert max == 3


def test_getting():
    """Test HashTable().get() both with and without defaults and
    HashTable().__getitem__()."""
    self = HashTable(10, (str, 10))
    assert self.add(["duck", "goose", "chicken"]).tolist() == [0, 1, 2]

    # The default default of -1.
    assert self.get("pigeon") == -1
    assert self.get(["goose", "pigeon", "parrot"]).tolist() == [1, -1, -1]

    # User defined integer default.
    assert self.get("pigeon", default=10) == 10
    assert self.get(["pigeon", "goose", "parrot"], default=5).tolist() \
           == [5, 1, 5]

    # User defined random object default.
    default = object()
    assert self.get("toad", default=default) is default
    assert self.get(["chicken", "toad"], default=default).tolist() \
           == [2, default]
    assert self.get("toad", default=None) is None

    # Defaulting disabled. Currently a private option.
    with pytest.raises(KeyError, match=r"key = 'troll' is not"):
        self.get("troll", default=self._NO_DEFAULT)

    # __getitem__() disables the default.
    assert self["chicken"] == 2
    assert self[["chicken", "duck"]].tolist() == [2, 0]
    with pytest.raises(KeyError):
        self["toad"]


def test_blame_key_multidimensional():
    """Test that the custom KeyErrors work for non scalar keys. """

    # Create a hash table for float triplets.
    self = HashTable(10, dtype=(float, 3))
    keys = np.arange(24, dtype=float).reshape((-1, 3))
    # Add all but the last key.
    self.add(keys[:-1])

    # Try getting the last key. The resultant key errors should always point to
    # the correct one being missing.
    with pytest.raises(KeyError, match=r"key = array\(\[21., 22., 23.\]\) is"):
        self[keys[-1]]
    with pytest.raises(KeyError, match=r"keys\[7\] = array\(\[21"):
        self[keys]
    with pytest.raises(KeyError, match=r"keys\[3, 1\] = array\(\[21"):
        self[keys.reshape((4, 2, 3))]


def test_blame_key_structured():
    """Similar to test_blame_key_multidimensional() but for struct dtypes."""
    self = HashTable(10, dtype=[("name", str, 10), ("age", int)])
    keys = np.array([("bill", 10), ("bob", 12), ("ben", 13)], self.dtype)
    self.add(keys[:-1])

    with pytest.raises(KeyError, match=r"key = \('ben', 13\) is"):
        self[keys[-1]]
    with pytest.raises(KeyError, match=r"keys\[2\] = \('ben', 13\) is"):
        self[keys]


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


def test_resize():
    self = HashTable(5, int)
    self.add([4, 3, 2, 9])

    with pytest.raises(ValueError, match=".* size 3 is .* fit 4 keys"):
        self.resize(3)

    for new_size in [4, 10]:
        smaller = self.resize(new_size)
        assert smaller.length == self.length
        assert np.array_equal(smaller.keys, self.keys)
        assert smaller.max == new_size


def test_copy():
    self = HashTable(10, int)
    self.add(range(3, 8))
    copy = self.copy()
    assert copy._destroyed is False
    assert copy.keys.tolist() == self.keys.tolist()
    self.add(9)
    assert 9 in self.keys
    assert 9 not in copy.keys
    copy.add(0)

    keys = self.destroy()
    copy = self.copy(usable=False)
    assert copy._destroyed is True
    assert copy.keys.tolist() == self.keys.tolist()
    keys[0] = 5
    assert copy.keys[0] == 3

    copy = self.copy(usable=True)
    assert copy._destroyed is False
    assert copy.keys.tolist() == [5, 4, 6, 7, 9]


def test_in():
    """Test HashTable().contains() and ``x in table``."""
    self = HashTable(10, int)
    self.add([20, 5, 50, 3, 4])

    assert self.contains(50) is True
    assert self.contains(51) is False

    assert self.contains([20, 4, 10, 99, 12]).tolist() == \
        [True, True, False, False, False]
    assert self.contains([[3, 5], [2, 1]]).tolist() == \
        [[True, True], [False, False]]

    assert 3 in self
    assert not 9 in self

    with pytest.raises(ValueError):
        # Not allowed by Python.
        [1, 2] in self


def test_PyInstaller_hook():
    if getattr(sys, "frozen", False):
        pytest.skip("")

    from hirola import _PyInstaller_hook_dir
    hook_dir, = _PyInstaller_hook_dir()
    assert os.path.isdir(hook_dir)
    hook = os.path.join(hook_dir, "hook-hirola.py")
    assert os.path.isfile(hook)

    namespace = runpy.run_path(hook)
    assert len(namespace["datas"]) == 2
