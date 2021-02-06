// -*- coding: utf-8 -*-

#include "hashes.h"


/* Hash functions.

  Hash functions are the key to hash table organisation. Without a good one,
  table lookup devolves to brute force searching.

  Hashes must:

  - Accept arbitrary blob of data and return a ptrdiff_t integer.
  - Be deterministic. hash(x) == hash(x)
  - Be as chaotic as possible.hash(b"hello") should be completely different to
    hash(b"hellO").
  - Break up consecutive or small range values. hash(3) and hash(4) should be
    very different and large, utilising the full range of a ptrdiff_t.
  - Mustn't be more likely to yield some answers more than others. This means
    that if you use multiplication in a hash, it must be with a prime number
    and large enough induce overflow. Otherwise the output is limited to only
    multiples of the multiplying factor.

  Unlike Python dicts, hoatzin.HashTable() is typed giving us the advantage of
  knowing in advance what data types to expect and their sizes. Rather than
  having to invent a be-all-end-all super duper hash for absolutely anything
  such as Python's builtin hash() function, hoatzin hash functions can be
  tailored certain data types.

 */


ptrdiff_t hash(void * key, const size_t key_size) {
  /* Generic hash - assumes **key_size** is a multiple of sizeof(int32_t). */

  int32_t * key_ = (int32_t *) key;
  ptrdiff_t out = 0;
  for (size_t i = 0; i < key_size / sizeof(int32_t); i++)
      out ^= key_[i] * 0x0B070503;
  return out;
}
