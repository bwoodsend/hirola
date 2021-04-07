# -*- coding: utf-8 -*-
"""
Hook for PyInstaller.
"""

from hoatzin._hash_table import slug

datas = [(str(slug.path), "hoatzin"),
         (str(slug.types_map.json_path), "hoatzin")]
