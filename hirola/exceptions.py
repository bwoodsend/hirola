# -*- coding: utf-8 -*-
"""
=========================
:mod:`hirola.exceptions`
=========================

Custom :class:`Exception` subclasses used by :mod:`hirola`.

"""


class HirolaException(Exception):
    """Base exception for all errors :mod:`hirola`-related."""
    pass


class HashTableFullError(HirolaException):
    """Raised on attempting to add a new item to a table when the table is full.
    """
    pass


class HashTableDestroyed(HirolaException):
    """Raised on adding or getting from a :class:`hirola.HashTable` which has
    called :meth:`hirola.HashTable.destroy`."""

    def __str__(self):
        return "This table has been destroyed by HashTable.destroy() and can " \
               "no longer be used."
