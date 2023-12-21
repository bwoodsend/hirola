# -*- coding: utf-8 -*-
"""
Test data generators for hash table keys.
"""

import functools
from pathlib import Path
import gzip
from urllib.request import Request, urlopen

import numpy as np


def random(n, d=16) -> np.ndarray:
    """Random 32-bit integers utilising the full range."""
    if n * d > 200_000_000:
        raise DataLimitExeeded
    bin = np.random.bytes(n * d)
    return np.frombuffer(bin, np.dtype(np.int32)).reshape((n, -1))


def id_like(n, d=2) -> np.ndarray:
    """Limited range integers. Roughly emulates mesh vertex indices or similar
    enumerations.
    """
    data = np.empty((n, d), dtype=np.int32)
    data[:, 0] = np.arange(n) * 2 // 3

    for i in range(1, d):
        data[:, i] = data[:, i - 1] + np.random.randint(-100, 100, n)

    return data.clip(min=0, max=data[-1, 0]).astype(np.int32)


def permutative(n) -> np.ndarray:
    """All possible pairs of integers from (0, 0) to (sqrt(n), sqrt(n))."""
    return np.c_[np.divmod(np.arange(n), int(np.sqrt(n)))].astype(np.int32)


def floating_32(n) -> np.ndarray:
    return floating_64(n).astype(np.float32)


def floating_64(n) -> np.ndarray:
    return np.random.uniform(-30, 30, (n, 3))


@functools.lru_cache()
def _lots_of_words(binary):
    cache = Path.home() / ".cache" / "hirola-benchmark-wordlist.txt.gz"
    url = "https://github.com/kkrypt0nn/wordlists/raw/1ca65fea80381e2caf9031e02c0602da6b48e936/wordlists/passwords/bt4_passwords.txt"
    if not (cache.exists() and cache.stat().st_size == 5391945):
        with urlopen(Request(url,
                             headers={"Accept-Encoding": "gzip"})) as response:
            cache.parent.mkdir(exist_ok=True, parents=True)
            cache.write_bytes(response.read())

    words = np.empty(1_000_000, dtype="<S20" if binary else "<U20")
    with gzip.open(cache) as f:
        for i in range(len(words)):
            for line in f:
                word = line.strip()
                if len(word) <= 20:
                    if not binary:
                        word = word.decode(errors="replace")
                    words[i] = word
                    break
    np.random.shuffle(words)
    return words


def textual(n):
    if n > len(_lots_of_words(False)):
        raise DataLimitExeeded
    return _lots_of_words(False)[:n]


def utf8(n):
    if n > len(_lots_of_words(True)):
        raise DataLimitExeeded
    return _lots_of_words(True)[:n]


class DataLimitExeeded(Exception):
    pass


def random_16(n):
    return random(n, 16)


def random_128(n):
    return random(n, 128)


generators = [
    random_16,
    random_128,
    id_like,
    permutative,
    floating_32,
    floating_64,
    textual,
    utf8,
]


def pysafe(x):
    """Convert an array into an array of elements which are hashable by Python's
     built in hash()."""
    return to_void(x).astype(object)


def to_void(x):
    """Convert to a 1D array of raw bytes. Doing so can circumvent the
    1D input only limitation of some numpy/pandas methods."""
    void = np.void(x.size * x.itemsize // len(x))
    return np.frombuffer(x.tobytes(), void)
