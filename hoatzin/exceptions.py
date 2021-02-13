# -*- coding: utf-8 -*-
"""
=========================
:mod:`hoatzin.exceptions`
=========================

Custom :class:`Exception` subclasses used by :mod:`hoatzin`.

"""


class HoatzinException(Exception):
    """Base exception for all errors :mod:`hoatzin`-related."""
    pass


class HashTableFullError(HoatzinException):
    """Raised on attempting to add a new item to a table when the table is full.
    """
    pass


class HashTableDestroyed(HoatzinException):
    """Raised on adding or getting from a :class:`hoatzin.HashTable` which has
    called :meth:`hoatzin.HashTable.destroy`."""

    def __str__(self):
        return "This table has been destroyed by HashTable.destroy() and can " \
               "no longer be used."
