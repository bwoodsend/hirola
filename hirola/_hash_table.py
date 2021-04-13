# -*- coding: utf-8 -*-
"""
"""

import numbers
import ctypes

from numbers import Number
from typing import Union, Tuple

import numpy as np
from cslug import CSlug, ptr, anchor, Header

hashes_header = Header(*anchor("hashes.h", "hashes.c"),
                       includes=["<stddef.h>", "<stdint.h>"])
slug = CSlug(anchor("hash_table", "hash_table.h", "hash_table.c", "hashes.c"),
             headers=hashes_header)

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
    _NO_DEFAULT = object()

    def __init__(self, max: Number, dtype: dtype_types):
        """

        Args:
            max:
                An upper bound for the number of keys which can fit in this
                table. Sets the :attr:`max` attribute.
            dtype:
                The data type for the table's keys. Sets the :attr:`dtype`
                attribute.

        The **max** parameter is silently normalised to :class:`int` and clipped
        to a minimum of 1 if it is less than 1.

        """
        self._dtype = np.dtype(dtype)
        key_size = self.dtype.itemsize
        self._base_dtype, self._dtype_shape = self.dtype.base, self.dtype.shape
        if self._base_dtype == object:
            raise TypeError("Object arrays are not permitted.")
        if self._base_dtype.kind in "SUV":
            # String-like types are checked differently.
            self._check_dtype = self._check_str_dtype

        if max <= 0:
            # Zero-sized tables get in the way of modulo.
            # Negative-sized tables obviously don't make sense.
            max = 1
        max = int(max)

        self._hash_owners = np.full(max, -1, np.intp)
        self._keys = np.empty(max, dtype=self.dtype)
        self._keys_readonly = np.frombuffer(self._keys, self.dtype)
        self._keys_readonly.flags.writeable = False

        hash = choose_hash(key_size)
        self._destroyed = False
        self._raw = slug.dll.HashTable(max, key_size, ptr(self._hash_owners),
                                       ptr(self._keys_readonly),
                                       hash=ctypes.cast(hash, ctypes.c_void_p))

    @property
    def max(self) -> int:
        """The maximum number of elements allowed in this table.

        Adding keys to exceed this maximum will trigger a
        :class:`~hirola.exceptions.HashTableFullError`. Nearly full tables
        will :meth:`add` and :meth:`get` much slower. For best performance,
        choose a maximum size which is 25-50% larger than you expect it to get.

        """
        return self._raw.max

    @property
    def dtype(self) -> np.dtype:
        """The data type for the table's keys.

        Use a structured :class:`numpy.dtype` to indicate that several numbers
        make up a single key.

        Examples:

            - Individual float keys: :py:`HashTable(100, np.float64)`.
            - Triplets of floats keys: :py:`HashTable(100, (np.float64, 3))`.
            - Strings of up to 20 characters: :py:`HashTable(100, (str, 20))`.
            - Mixed types records data:
              :py:`HashTable(100, [("firstname", str, 20), ("lastname", str, 20), ("age", int)])`.

        """
        return self._dtype

    @property
    def key_size(self) -> int:
        """The number of bytes per key."""
        return self._raw.key_size

    @property
    def keys(self) -> np.ndarray:
        """The unique elements in the table.

        Unlike Python's builtin :class:`dict`, this :attr:`keys` is a
        :class:`property` rather than a method. Keys must be immutable hence
        this array is a readonly view. See :meth:`destroy` to release the
        writable version.

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
            exceptions.HashTableDestroyed:
                If the :meth:`destroy` method has been previously called.

        """
        self._check_destroyed()

        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, np.intp)
        index = slug.dll.HT_adds(self._raw._ptr, ptr(keys), ptr(out), out.size)
        if index != -1:
            from hirola.exceptions import HashTableFullError
            source, value = self._blame_key(index, keys, shape)
            raise HashTableFullError(
                f"Failed to add {source} = {value} to the "
                f"hash table because the table is full and {value} "
                f"isn't already in it.")
        return out if shape else out.item()

    def contains(self, keys) -> Union[bool, np.ndarray]:
        """Check if a key or keys are in the table.

        Args:
            keys:
                Elements to check for.
        Returns:
            Either true or false for each key in **keys**.

        This function is equivalent to but faster than
        :py:`table.get(keys) != -1`.
        To check only one key you may also use :py:`key in table`.

        """
        self._check_destroyed()

        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, bool)
        slug.dll.HT_contains(self._raw._ptr, ptr(keys), ptr(out), out.size)
        return out if shape else out.item()

    __contains__ = contains

    def get(self, keys, default=-1) -> np.ndarray:
        """Lookup indices of **keys** in :attr:`keys`.

        Arguments:
            keys:
                Elements to search for.
            default:
                Returned inplace of a missing key.
                May be any object.
        Returns:
            The index/indices of **keys** in this table's :attr:`keys`. If a
            key is not there, returns :py:`-1` in its place.
        Raises:
            ValueError:
                If the :attr:`~numpy.ndarray.dtype` of **keys** doesn't match
                the :attr:`dtype` of this table.
            exceptions.HashTableDestroyed:
                If the :meth:`destroy` method has been previously called.

        """
        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, np.intp)
        # This function forks out to several similar C functions depending on
        # how missing keys are to be handled.

        if default is self._NO_DEFAULT:
            # Default disabled - raise a key error if anything is missing.
            index = slug.dll.HT_gets_no_default(self._raw._ptr, ptr(keys),
                                                ptr(out), out.size)
            if index != -1:
                source, value = self._blame_key(index, keys, shape)
                raise KeyError(f"{source} = {value} is not in this table.")

        elif isinstance(default, numbers.Integral):
            if default == -1:
                # The default behaviour - use -1 to indicate missing keys.
                # This is already how the underlying C functions communicate
                # missing keys so nothing special needs to be done.
                slug.dll.HT_gets(self._raw._ptr, ptr(keys), ptr(out), out.size)
            else:
                # Not the default of -1 but still an integer default which can
                # be handled faster in C.
                slug.dll.HT_gets_default(self._raw._ptr, ptr(keys), ptr(out),
                                         out.size, default)

        else:
            # The slowest case: Return some non integer user defined default.
            slug.dll.HT_gets(self._raw._ptr, ptr(keys), ptr(out), out.size)
            out = np.where(out == -1, default, out)

        return out if shape else out.item()

    def _blame_key(self, index, keys, shape) -> Tuple[str, str]:
        """Get a key and its location from a ravelled index. Used to prettify
        key errors."""
        assert index >= 0
        if len(shape) == 0:
            if self._dtype_shape:
                return "key", repr(keys)
            return "key", repr(keys.item())
        if len(shape) == 1:
            return f"keys[{index}]", repr(keys[index])
        index = np.unravel_index(index, shape)
        return ("keys[" + ', '.join(map(str, index)) + "]"), repr(keys[index])

    def __getitem__(self, key):
        return self.get(key, default=self._NO_DEFAULT)

    def _check_dtype(self, keys):
        keys = np.asarray(keys, order="C")
        if keys.dtype != self._base_dtype:
            raise TypeError(
                "The dtype must match the dtype of the hash table. Expecting {}"
                " but got {}.".format(self._base_dtype, keys.dtype))
        return keys

    def _check_str_dtype(self, keys):
        return np.asarray(keys, dtype=self.dtype, order="C")

    def _norm_input_keys(self, keys):
        """Prepare input to be fed to C.

        * Convert to C contiguous array if not already.
        * Check/raise for wrong dtype.
        * For unlabelled records dtypes, right-strip :attr:`_dtype_shape`.

        """
        self._check_destroyed()
        keys = self._check_dtype(keys)

        if self._dtype_shape:
            split = keys.ndim - len(self._dtype_shape)
            if keys.shape[split:] != self._dtype_shape:
                raise ValueError(
                    "The given shape's suffix doesn't match that of the hash "
                    "table's dtype. Received {} which doesn't end with {}.".
                    format(self._dtype_shape, keys.shape))
            return keys, keys.shape[:split]
        return keys, keys.shape

    def destroy(self) -> np.ndarray:
        """Release a writeable version of :attr:`keys` and permanently disable
        this table.

        Returns:
            A writeable shallow copy of :attr:`keys`.

        Modifying :attr:`keys` would cause an internal meltdown and is
        therefore blocked by setting the writeable flag to false. However, if
        you no longer need this table then it is safe to do as you please with
        the :attr:`keys` array. This function grants you full access to
        :attr:`keys` but blocks you from adding to or getting from this table
        in the future. If you want both a writeable :attr:`keys` array and
        functional use of this table then use :py:`table.keys.copy()`.

        """
        # Destruction is just setting a flag.
        self._destroyed = True
        return self._keys[:self.length]

    def _check_destroyed(self):
        if self._destroyed:
            from hirola.exceptions import HashTableDestroyed
            raise HashTableDestroyed

    def resize(self, new_size) -> 'HashTable':
        """Copy the contents of this table into a new :class:`HashTable` of a
        different size.

        Args:
            new_size:
                The new value for :attr:`max`.
        Returns:
            A new hash table with attributes :attr:`keys` and :attr:`length`
            matching those from this table.
        Raises:
            ValueError:
                If requested size is too small (:py:`new_size < len(table)`).

        """
        self._check_destroyed()
        if new_size < self.length:
            raise ValueError(f"Requested size {new_size} is too small to fit "
                             f"{self.length} keys.")
        # This is only marginally (10-20%) faster than just creating an empty
        # table and running `new.add(self.keys)`.
        new = type(self)(new_size, self.dtype)
        slug.dll.HT_copy_keys(self._raw._ptr, new._raw._ptr)
        return new

    def copy(self, usable=True) -> 'HashTable':
        """Deep copy this table.

        Args:
            usable:
                If set to false and this table has called :meth:`destroy`, then
                the destroyed state is propagated to copies.
        Returns:
            Another :class:`HashTable` with the same size, dtype and content.

        """
        out = type(self)(self.max, self.dtype)
        if self._destroyed and usable:
            out.add(self.keys)
        else:
            out._raw.length = self._raw.length
            out._destroyed = self._destroyed
            out._hash_owners[:] = self._hash_owners
            out._keys[:out._raw.length] = self.keys

        return out


def choose_hash(key_size):
    if key_size < 4:
        hash = slug.dll.small_hash
    elif key_size % 4 == 0:
        hash = slug.dll.hash
    else:
        hash = slug.dll.hybrid_hash
    return hash


def vectorise_hash(hash, key_size, keys):
    """Apply a hash() function to an array of **keys**. Only used for testing.
    """
    keys = np.ascontiguousarray(keys)
    out = np.empty(keys.size * keys.dtype.itemsize // key_size, dtype=np.int32)
    slug.dll.vectorise_hash(ctypes.cast(hash, ctypes.c_void_p), ptr(keys),
                            ptr(out), key_size, out.size)
    return out
