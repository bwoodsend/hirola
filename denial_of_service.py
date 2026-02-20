import itertools
import time

import numpy as np
import hirola

MASK = (1 << 32) - 1
A = 0x10001  # magic constants come from hirola/hashes.c
B = 0x0B070503
A_inv = 0xFFFF0001  # pow(A, -1, 1 << 32)
B_inv = 0xF32F8DAB  # pow(B, -1, 1 << 32)


def hash32(seed, words):
    """Pure Python equivalent of hash() from hirola/hashes.c"""
    out = seed
    for w in words:
        out = ((out ^ ((w * A) & MASK)) * B) & MASK
    return out


def hybrid_hash(seed, words):
    """Pure Python equivalent of hybrid_hash() from hirola/hashes.c"""
    return hash32(seed, words[:-1]) ^ (words[-1] * B) & MASK


def fix_hash(current_hash, target_hash):
    """Calculate what int32 to append to a sequence to achieve a given hash"""
    pre = (target_hash & MASK) * B_inv & MASK
    return ((current_hash ^ pre) * A_inv) & MASK


def _combinations(sequence, n):
    """itertools.combinations_with_replacement() but allows huge sequences"""
    if n == 1:
        for i in sequence:
            yield (i,)
    elif n >= 2:
        for i in sequence:
            for _sequence in _combinations(sequence, n - 1):
                yield (i, *_sequence)


def _batched(iterable, chunk_size):
    while True:
        out = list(itertools.islice(iterable, chunk_size))
        if not out:
            break
        yield out


def generate_hash_collisions(seed, key_size, target_hash):
    """Generate all keys of a given size that produce a given hash"""
    # For key_size 𝜖 multiples of 4, treat keys as a sequence of uint32, iterate
    # through all combinations of all but the last uint32 then calculate the
    # last uint32 to be whatever achieves the intended hash.
    if key_size % 4 == 0:
        for prefix in _combinations(range(1 << 32), key_size // 4 - 1):
            key = (*prefix, fix_hash(hash32(seed, prefix), target_hash))
            yield key
    # For other key_size, iterate through all possible values of the remainder
    # bytes, calculate what the hash would need to be before applying the
    # remainder bytes then prepend any 4*n keys that would produce that
    # intermediate hash.
    else:
        remainder = key_size % 4
        _key_size = key_size - remainder
        for suffix in range(8 ** remainder):
            _target_hash = target_hash ^ (suffix * A_inv) & MASK
            for prefix in generate_hash_collisions(_key_size, _target_hash):
                yield (*prefix, suffix)


def main(seed, attacker_seed):
    a = hirola.HashTable(100, (np.uint32, 2), (0.8, 1.5), seed=seed)
    try:
        for keys in _batched(generate_hash_collisions(attacker_seed, 8, 0), 1000):
            t0 = time.time()
            a.add(np.array(keys, dtype=np.uint32))
            t1 = time.time()
            print(f"len(table)={len(a)}, table.max={a.max}, insertion-time={t1 - t0}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
        "DoS attack simulator: Set seed and attacker_seed to the same value to "
        "mimic a successful denial of service attack. Set to mismatching "
        "values to simulate DoS being thwarted.")
    parser.add_argument("seed", type=int, default=0, nargs="?")
    parser.add_argument("attacker_seed", type=int, default=0, nargs="?")
    options = parser.parse_args()
    main(options.seed, options.attacker_seed)
