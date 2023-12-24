# -*- mode: python ; coding: utf-8 -*-

import sys

from PyInstaller.utils.hooks import collect_entry_point

# Each pytest plugin needs to be explicitly collected.
datas, hiddenimports = collect_entry_point("pytest11")

# Collect modules imported by the test suite but not hirola itself.
hiddenimports += ["statistics"]

if sys.version_info < (3, 8):
  hiddenimports += ["py._path.local"]

a = Analysis(['frozen-pytest.py'], datas=datas, hiddenimports=hiddenimports)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='frozen-pytest')
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='frozen-pytest')
