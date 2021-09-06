# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_entry_point

# Each pytest plugin needs to be explicitly collected.
datas, hiddenimports = collect_entry_point("pytest11")

# Collect modules imported by the test suite but not hirola itself.
hiddenimports += ["statistics"]

a = Analysis(['frozen-pytest.py'], datas=datas, hiddenimports=hiddenimports)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='frozen-pytest')
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='frozen-pytest')
