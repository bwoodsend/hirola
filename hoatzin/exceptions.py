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
