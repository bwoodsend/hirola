# -*- coding: utf-8 -*-
"""
"""
from ._version import __version__, __version_info__
from ._hash_table import HashTable
from . import exceptions


def _PyInstaller_hook_dir():
    import os
    return [os.path.dirname(__file__)]
