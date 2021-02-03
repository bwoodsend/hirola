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
