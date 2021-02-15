// -*- coding: utf-8 -*-

#include <string.h>
#include <stdint.h>
#include "hashes.h"


/* Hash functions.

  Hash functions are the key to hash table organisation. Without a good one,
  table lookup devolves to brute force searching.

  Hashes must:

  - Accept arbitrary blob of data and return a int32_t integer.
  - Be deterministic. hash(x) == hash(x)
  - Be as chaotic as possible.hash(b"hello") should be completely different to
    hash(b"hellO").
  - Break up consecutive or small range values. hash(3) and hash(4) should be
    very different and large, utilising the full range of a int32_t.
  - Mustn't be more likely to yield some answers more than others. This means
    that if you use multiplication in a hash, it must be with a prime number
    and large enough induce overflow. Otherwise the output is limited to only
    multiples of the multiplying factor.

  Unlike Python dicts, hirola.HashTable() is typed giving us the advantage of
  knowing in advance what data types to expect and their sizes. Rather than
  having to invent a be-all-end-all super duper hash for absolutely anything
  such as Python's builtin hash() function, hirola hash functions can be
  tailored certain data types.

  These hashes are empirically designed to best satisfy tests/test_hashes.py.

 */

/* A big prime number that fits into a int32_t. */
const size_t NOISE = 0x0B070503;

int32_t hash(void * key, const size_t key_size) {
  /* Generic hash - assumes **key_size** is a multiple of sizeof(int32_t). */

  int32_t * key_ = (int32_t *) key;
  int32_t out = 0;
  for (size_t i = 0; i < key_size / sizeof(int32_t); i++) {
      out ^= key_[i] * 0x10001;
      out *= NOISE;
  }
  return out;
}


int32_t small_hash(void * key, const size_t key_size) {
  /* Hash for key_size <= sizeof(int32_t). */

  int32_t out = 0;
  memcpy(&out, key, key_size);
  return out * NOISE;
}


int32_t hybrid_hash(void * key, const size_t key_size) {
  /* A combination of hash() and small_hash() for when **key_size** is not a
     multiple of sizeof(int32_t). */

  size_t tail = key_size % sizeof(int32_t);
  return hash(key, key_size) ^ small_hash(key + (key_size - tail), tail);
}
