# -*- coding: utf-8 -*-
"""
Hook for PyInstaller.
"""

from hirola._hash_table import slug

datas = [(str(slug.path), "hirola"), (str(slug.types_map.json_path), "hirola")]
