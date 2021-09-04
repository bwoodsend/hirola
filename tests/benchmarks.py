# -*- coding: utf-8 -*-
"""Speed comparison of hirola vs numpy.keys vs Python dict/set.

This is highly platform/compiler dependent. Best performance is on Linux when
compiled with clang which is 1.5x faster than Linux with gcc, Windows with gcc
or FreeBSD with clang or gcc.
It also of course depends on exactly what code you compare.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__, "..", "..").resolve()))

import numpy as np
from sloth.simple import time_callable  # pip install sloth-speedtest

from hirola import HashTable
from tests.generate import pysafe, permutative


def pure_python(data):
    s = set(data)


def pure_python_get(data):
    # set(data) on its own is very efficient - only 2-3x slower than hirola.
    # However, getting anything out of the set, even a harmless
    #   for i in set(data):
    #        pass
    # is much slower (10-15x slower than hirola).
    s = set(data)
    list(zip(s, range(len(s))))


def pure_numpy(data):
    np.unique(data)


def hirola(data):
    self = HashTable(len(data) * 3 // 2, (data.dtype, data[0].shape))
    self.add(data)


if __name__ == '__main__' and "benchmark" in sys.argv:
    pure_python(pysafe(permutative(1000)))
    hirola(permutative(1000))

    print(
        time_callable(pure_python_get, 200, pysafe(permutative(100000))) /
        time_callable(hirola, 200, permutative(100000)))
