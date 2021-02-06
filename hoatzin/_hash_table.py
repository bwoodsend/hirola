# -*- coding: utf-8 -*-
"""
"""

from numbers import Number
from typing import Union

import numpy as np
from cslug import CSlug, ptr, anchor

slug = CSlug(anchor("hash_table", "hash_table.h", "hash_table.c"))

dtype_types = Union[np.dtype, np.generic, type, str, list, tuple]


class HashTable(object):
    """The raw core behind a set or a dictionary's keys.

    A hash table resembles a dictionary where its keys are the :attr:`keys`
    array but the values are an just an enumeration. It's core API:

    * Use :meth:`add` to add new keys if they've not been already added.
    * The :attr:`keys` lists all keys that have been added in the order that
      they were added.
    * Use :meth:`get` to retrieve indices of keys in :attr:`keys`.

    """
    _keys: np.ndarray
    dtype: np.dtype
    """The data type for the table's keys.

    Use structured dtypes to indicate that several numbers make up a single key.

    - Individual float keys: :py:`HashTable(100, np.float64)`.
    - Triplets of floats keys: :py:`HashTable(100, (np.float64, 3))`.
    - Strings of up to 20 characters: :py:`HashTable(100, (str, 20))`.
    - Mixed types records data:
      :py:`HashTable(100, [("firstname", str, 20), ("lastname", str, 20), ("age", int)])`.

    """

    def __init__(self, max: Number, dtype: dtype_types):
        """

        Args:
            max:
                An upper bound for the number of keys which can fit in this
                table. Sets the :attr:`max` attribute.
            dtype:
                The data type for the table's keys. Sets the :attr:`dtype`
                attribute.

        """
        self.dtype = np.dtype(dtype)
        key_size = self.dtype.itemsize
        self._base_dtype, self._dtype_shape = self.dtype.base, self.dtype.shape
        if self._base_dtype == object:
            raise TypeError("Object arrays are not permitted.")

        if max <= 0:
            # Zero-sized tables get in the way of modulo.
            # Negative-sized tables obviously don't make sense.
            max = 1

        self._hash_owners = np.full(max, -1, np.intp)
        self._keys = np.empty(max, dtype=self.dtype)
        self._keys_readonly = np.frombuffer(self._keys, self.dtype)
        self._keys_readonly.flags.writeable = False

        self._raw = slug.dll.HashTable(max, key_size, ptr(self._hash_owners),
                                       ptr(self._keys))

    @property
    def max(self) -> int:
        """The maximum number of elements allowed in this table.

        Adding keys to exceed this maximum will trigger a
        :class:`~hoatzin.exceptions.HashTableFullError`. Nearly full tables
        will :meth:`add` and :meth:`get` much slower. For best performance,
        choose a maximum size which is 25-50% larger than you expect it to get.

        """
        return self._raw.max

    @property
    def key_size(self) -> int:
        """The number of bytes per key."""
        return self._raw.key_size

    @property
    def keys(self) -> np.ndarray:
        """The unique elements in the table.

        Unlike Python's builtin :class:`dict`, this :attr:`keys` is a
        :class:`property` rather than a method. Keys must be immutable hence
        this array is a readonly view.

        """
        return self._keys_readonly[:self.length]

    @property
    def length(self) -> int:
        """The number of elements currently in this table. Aliased via
        :py:`len(table)`."""
        return self._raw.length

    def __len__(self):
        return self.length

    def _add(self, key):
        return slug.dll.HT_add(self._raw._ptr, ptr(key))

    def _get(self, key):
        return slug.dll.HT_get(self._raw._ptr, ptr(key))

    def add(self, keys) -> np.ndarray:
        """Add **keys** to the table.

        Any key which is already in :attr:`keys` is not added again. Returns
        the index of each key in :attr:`keys` similarly to :meth:`get`.

        Raises:
            ValueError:
                If the :attr:`~numpy.ndarray.dtype` of **keys** doesn't match
                the :attr:`dtype` of this table.
            exceptions.HashTableFullError:
                If there is no space to place new keys.

        """
        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, np.intp)
        index = slug.dll.HT_adds(self._raw._ptr, ptr(keys), ptr(out), out.size)
        if index != -1:
            from hoatzin.exceptions import HashTableFullError
            if len(shape) != 1:
                index = np.unravel_index(index, shape)
            raise HashTableFullError(
                f"Failed to add element {keys[index]} (index {index}) to the "
                f"hash table because the table is full and {keys[index]} "
                f"isn't already in it.")
        return out

    def get(self, keys) -> np.ndarray:
        """Lookup indices of **keys** in :attr:`keys`.

        Arguments:
            keys:
                Elements to search for.
        Returns:
            The index/indices of **keys** in this table's :attr:`keys`. If a
            key is not there, returns :py:`-1` in its place.
        Raises:
            ValueError:
                If the :attr:`~numpy.ndarray.dtype` of **keys** doesn't match
                the :attr:`dtype` of this table.

        """
        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, np.intp)
        slug.dll.HT_gets(self._raw._ptr, ptr(keys), ptr(out), out.size)
        return out

    __getitem__ = get

    def _check_dtype(self, keys):
        if not np.issubdtype(keys.dtype, self._base_dtype):
            raise TypeError(
                "The dtype must match the dtype of the hash table. Expecting {}"
                " but got {}.".format(self._base_dtype, keys.dtype))

    def _norm_input_keys(self, keys):
        """Prepare input to be fed to C.

        * Convert to C contiguous array if not already.
        * Check/raise for wrong dtype.
        * For unlabelled records dtypes, right-strip :attr:`_dtype_shape`.

        """
        keys = np.asarray(keys, order="C")
        self._check_dtype(keys)
        if self._dtype_shape:
            split = keys.ndim - len(self._dtype_shape)
            if keys.shape[split:] != self._dtype_shape:
                raise ValueError(
                    "The given shape's suffix doesn't match that of the hash "
                    "table's dtype. Received {} which doesn't end with {}.".
                    format(self._dtype_shape, keys.shape))
            return keys, keys.shape[:split]
        return keys, keys.shape
