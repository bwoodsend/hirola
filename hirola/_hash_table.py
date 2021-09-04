# -*- coding: utf-8 -*-
"""
"""

import numbers
import ctypes
import math

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

    A hash table resembles a dictionary where its keys are the `keys`
    array but the values are an just an enumeration. It's core API:

    * Use `add` to add new keys if they've not been already added.
    * The `keys` lists all keys that have been added in the order that
      they were added.
    * Use `get` to retrieve indices of keys in `keys`.

    """
    _keys: np.ndarray
    _NO_DEFAULT = object()

    def __init__(self, max: Number, dtype: dtype_types,
                 almost_full=(.9, "warn")):
        """

        Args:
            max:
                An upper bound for the number of keys which can fit in this
                table. Sets the `max` attribute.
            dtype:
                The data type for the table's keys. Sets the `dtype`
                attribute.
            almost_full:
                The handling of almost full hash tables. Sets the `almost_full`
                attribute.

        The **max** parameter is silently normalised to `int` and clipped
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
        self.almost_full = almost_full

    @property
    def max(self) -> int:
        """The maximum number of elements allowed in this table.

        Adding keys to exceed this maximum will trigger a
        :class:`~hirola.exceptions.HashTableFullError`. Nearly full tables
        will `add` and `get` much slower. For best performance,
        choose a maximum size which is 25-50% larger than you expect it to get.

        """
        return self._raw.max

    @property
    def dtype(self) -> np.dtype:
        """The data type for the table's keys.

        Use a structured `numpy.dtype` to indicate that several numbers
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

        Unlike Python's builtin `dict`, this `keys` is a
        `property` rather than a method. Keys must be immutable hence
        this array is a readonly view. See `destroy` to release the
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

        Any key which is already in `keys` is not added again. Returns
        the index of each key in `keys` similarly to `get`.

        Raises:
            ValueError:
                If the :attr:`~numpy.ndarray.dtype` of **keys** doesn't match
                the `dtype` of this table.
            exceptions.HashTableFullError:
                If there is no space to place new keys.
            exceptions.HashTableDestroyed:
                If the `destroy` method has been previously called.
            exceptions.AlmostFull:
                If the table becomes nearly full and is configured to raise an
                error (set by the `almost_full` attribute).
        Warns:
            exceptions.AlmostFull:
                If the table becomes nearly full and is configured to warn (set
                by the `almost_full` attribute).

        """
        self._check_destroyed()

        keys, shape = self._norm_input_keys(keys)
        out = np.empty(shape, np.intp)

        # This while loop will only iterate a second time if the "almost full"
        # threshold is enabled and crossed. It will only iterate more than twice
        # if `self.almost_full` is set to automatically upsize the table.
        index = -1
        while True:
            index = slug.dll.HT_adds(self._raw._ptr, ptr(keys), ptr(out),
                                     out.size, index + 1)

            # If everything worked. Return the indices.
            if index == out.size:
                return out if shape else out.item()

            # If the `almost_full` threshold has been crossed:
            if index < 0:
                # Convert to a real positive index.
                index = -1 - index
                # Issue a warning or raise an error or resize the table as per
                # the user's configuration.
                self._on_almost_full()
                continue

            # We're out of space. Raise an error.
            from hirola.exceptions import HashTableFullError
            source, value = self._blame_key(index, keys, shape)
            raise HashTableFullError(
                f"Failed to add {source} = {value} to the "
                f"hash table because the table is full and {value} "
                f"isn't already in it.")

    @property
    def almost_full(self):
        """The response to an almost full hash table. Hash tables become
        dramatically slower, the closer they get to being full. Hirola's default
        behaviour is to warn if this happens but can be configured to ignore the
        warning, raise an error or automatically make a new, larger table.

        This is an overloaded parameter.

        * :py:`almost_full = None`:
            Disable the *almost full* warning entirely.
        * :py:`almost_full = (0.8, "warn")`:
            Issue a `hirola.exceptions.AlmostFull` warning if the table reaches
            80% full.
        * :py:`almost_full = (0.7, "raise")`:
            Raise a `hirola.exceptions.AlmostFull` exception if the table
            reaches 80% full.
        * :py:`almost_full = (0.7, 2)`:
            Whenever the table reaches 70% full, double the table size.

        For reference, Python's `dict` grows 8-fold when two thirds full.
        To mimic this behaviour, set :py:`table.almost_full = (2 / 3, 8)`.

        """
        return self._almost_full

    @almost_full.setter
    def almost_full(self, x):
        # Asides from simply storing the user's value, this setter must also:
        # * Calculate the "panic table length" (self._raw.panic_at) at which the
        #   C code should notify Python that the table is almost full.
        # * Ensure that the user input is valid.

        if x is None:
            self._raw.panic_at = -1
            self._almost_full = None
            return

        try:
            ratio, scale_up = x
        except:
            raise TypeError(f"`almost_full` must be a 2-tuple of floats or"
                            f" None. Not `{repr(x)}`.") from None
        if not (0 < ratio <= 1):
            raise ValueError("The first parameter to almost_full must be "
                             ">0 and <=1.")
        if isinstance(scale_up, str):
            if scale_up not in ("raise", "warn"):
                raise ValueError("Valid near-full actions are 'raise' and "
                                 f"'warn'. Not '{scale_up}'.")
        elif isinstance(scale_up, numbers.Number):
            if int(self.max * scale_up) <= self.max:
                raise ValueError(
                    f"A scale_up resize factor of {scale_up} would lead to an "
                    f"infinite loop. Either increase scale_up or disable "
                    f"automatic resizing by setting hash_table.almost_full to "
                    f"None.")
        else:
            raise TypeError("The second parameter to almost_full must be "
                            "either a string or a float.")

        self._raw.panic_at = int(math.ceil(ratio * self.max))
        self._almost_full = x

    def _on_almost_full(self):
        """The callback to be invoked whenever the table becomes almost full."""
        assert self.almost_full is not None

        if not isinstance(self.almost_full[1], str):
            self.resize(self.max * self.almost_full[1], in_place=True)
            return

        from hirola.exceptions import AlmostFull

        message = f"HashTable() is {round(100 * len(self) / self.max)}% full." \
                  " A hash table becomes orders of magnitudes slower " \
                  "when nearly full. See help(HashTable.almost_full) for how " \
                  "to correct or silence this issue."

        if self.almost_full[1] == "raise":
            raise AlmostFull(message)
        import warnings
        warnings.warn(AlmostFull(message), stacklevel=3)

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
        """Lookup indices of **keys** in `keys`.

        Arguments:
            keys:
                Elements to search for.
            default:
                Returned inplace of a missing key.
                May be any object.
        Returns:
            The index/indices of **keys** in this table's `keys`. If a
            key is not there, returns :py:`-1` in its place.
        Raises:
            ValueError:
                If the :attr:`~numpy.ndarray.dtype` of **keys** doesn't match
                the `dtype` of this table.
            exceptions.HashTableDestroyed:
                If the `destroy` method has been previously called.

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
        * For unlabelled records dtypes, right-strip self._dtype_shape.

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
        """Release a writeable version of `keys` and permanently disable
        this table.

        Returns:
            A writeable shallow copy of `keys`.

        Modifying `keys` would cause an internal meltdown and is
        therefore blocked by setting the writeable flag to false. However, if
        you no longer need this table then it is safe to do as you please with
        the `keys` array. This function grants you full access to
        `keys` but blocks you from adding to or getting from this table
        in the future. If you want both a writeable `keys` array and
        functional use of this table then use :py:`table.keys.copy()`.

        """
        # Destruction is just setting a flag.
        self._destroyed = True
        return self._keys[:self.length]

    def _check_destroyed(self):
        if self._destroyed:
            from hirola.exceptions import HashTableDestroyed
            raise HashTableDestroyed

    def resize(self, new_size, in_place=False) -> 'HashTable':
        """Copy the contents of this table into a new `HashTable` of a
        different size.

        Args:
            new_size:
                The new value for `max`.
            in_place:
                If true resize this table. Otherwise make a modified copy.
        Returns:
            Either a new hash table with attributes `keys` and `length`
            matching those from this table or this table.
        Raises:
            ValueError:
                If requested size is too small (:py:`new_size < len(table)`).

        .. versionchanged:: 0.3.0
            Add the **in_place** option.

        """
        self._check_destroyed()
        if new_size < self.length:
            raise ValueError(f"Requested size {new_size} is too small to fit "
                             f"{self.length} keys.")
        new = type(self)(new_size, self.dtype, almost_full=self.almost_full)
        # This is only marginally (10-20%) faster than just creating an empty
        # table and running `new.add(self.keys)`.
        slug.dll.HT_copy_keys(self._raw._ptr, new._raw._ptr)

        if in_place:
            # In place resizing really just creates a resized copy then moves
            # all the resized attributes back to the original.
            self._raw = new._raw
            self._keys = new._keys
            self._hash_owners = new._hash_owners
            self._keys_readonly = new._keys_readonly
            self.almost_full = new.almost_full
            return self

        return new

    def copy(self, usable=True) -> 'HashTable':
        """Deep copy this table.

        Args:
            usable:
                If set to false and this table has called `destroy`, then
                the destroyed state is propagated to copies.
        Returns:
            Another `HashTable` with the same size, dtype and content.

        """
        out = type(self)(self.max, self.dtype, self.almost_full)
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
