"""Speed comparison of hirola vs numpy.keys vs Python dict/set.

This is highly platform/compiler dependent. Best performance is on Linux when
compiled with clang which is 1.5x faster than Linux with gcc, Windows with gcc
or FreeBSD with clang or gcc.
It also of course depends on exactly what code you compare.

"""
import functools
import itertools
import sys
from pathlib import Path
import re
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__, "..", "..").resolve()))

import numpy as np
from colorama import init
from tabulate import tabulate, tabulate_formats
from humanize import metric
from sloth.simple import time_callable as timeit

from tests import generate

init()
cursor_up = lambda lines: '\x1b[{0}A'.format(lines)


def numpy_unique(data):
    np.unique(data)


def numpy_unique_indices(data):
    np.unique(data, return_inverse=True)


try:
    import pandas

    def pandas_categorical(data):
        pandas.Categorical(data).codes
except ImportError:
    pandas_categorical = None

try:
    import numpy_indexed

    def numpy_indexed_unique(data):
        numpy_indexed.unique(data)
except ImportError:
    numpy_indexed_unique = None


@functools.lru_cache
def hirola(size_multiplier, old=False):
    if old:
        try:
            from hirola_old import HashTable
        except ImportError:
            return None
    else:
        from hirola import HashTable

    def hirola(data):
        self = HashTable(
            len(data) * size_multiplier, (data.dtype, data[0].shape))
        self.add(data)

    hirola.__name__ += "_" + str(size_multiplier) + ("_old" if old else "")
    return hirola


METHODS = {
    "hirola x1.25": (hirola(1.25), None),
    "hirola x1.25 old": (hirola(1.25, True), None),
    "hirola x1.5": (hirola(1.5), None),
    "hirola x1.5 old": (hirola(1.5, True), None),
    "hirola x2.5": (hirola(2.5), None),
    "hirola x2.5 old": (hirola(2.5, True), None),
    "hirola x5": (hirola(5), None),
    "hirola x5 old": (hirola(5, True), None),
    "set()": (set, generate.pysafe),
    "numpy.unique()": (numpy_unique, None),
    "numpy.unique(return_indices=True)": (numpy_unique_indices, None),
    "numpy_indexed.unique()": (numpy_indexed_unique, None),
    "pandas.Categorical()": (pandas_categorical, generate.to_void),
}
METHODS = {i:j for (i, j) in METHODS.items() if j[0] is not None}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("data_type",
                        choices=[i.__name__ for i in generate.generators])
    parser.add_argument("size", type=int)
    parser.add_argument("--skip", action="append")
    parser.add_argument("--format", choices=tabulate_formats)
    options = parser.parse_args()
    if options.skip:
        skip_filter = re.compile("|".join(map(re.escape, options.skip))).search
    else:
        skip_filter = lambda x: False

    data = getattr(generate, options.data_type)(options.size)
    methods = {
        name: (method, normalize(data) if normalize else data)
        for (name, (method, normalize)) in METHODS.items()
        if not skip_filter(name)
    }

    means = dict.fromkeys(methods, 0)
    names = [i for i in means]

    for method, data in methods.values():
        method(data)

    def show(rewind=True):
        times = tabulate([[metric(i, "s") for i in means.values()]], names,
                         tablefmt=options.format)

        ratios = tabulate([[means[i] / means[j] for i in means] for j in means],
                          names, showindex=names, floatfmt=".3f",
                          tablefmt=options.format)

        out = f"{times}\n\n{ratios}\n"

        print(out, end="")
        if rewind:
            print(end=cursor_up(out.count("\n")))

    try:
        for i in itertools.count():
            for (name, (method, data)) in methods.items():
                time = timeit(method, max(10_000_000 // options.size, 3), data)
                means[name] = (i * means[name] + time) / (i + 1)

            show()

    except KeyboardInterrupt:
        print("\r", end="")
        show(rewind=False)
        print()
