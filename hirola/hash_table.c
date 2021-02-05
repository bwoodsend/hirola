// -*- coding: utf-8 -*-

#include "hash_table.h"


ptrdiff_t euclidean_modulo(ptrdiff_t x, ptrdiff_t base) {
  /* Python style modulo.
     Unlike C modulo, the output is still >= 0 for negative input. */
  x = x % ((ptrdiff_t) base);
  if (x < 0) x += base;
  return x;
}


ptrdiff_t hash(void * key, const size_t key_size) {

  int32_t * key_ = (int32_t *) key;
  ptrdiff_t out = 0;
  for (size_t i = 0; i < key_size / sizeof(int32_t); i++)
      out ^= key_[i] * 0x0B070503;
  return out;
}


ptrdiff_t HT_hash_for(HashTable * self, void * key) {
  /* Either find **key**'s hash if **key** is in `self->keys` or
     choose an unused hash. */

  // Get an initial hash for `key`.
  ptrdiff_t _hash = euclidean_modulo(hash(key, self->key_size), self->max);

  // Search, starting from our initial hash:
  for (size_t j = 0; j < self->max; j++) {
    // Provided the hash function hash() is working well for the input
    // data, this loop should rarely require more than one iteration.

    // If _hash is unclaimed:
    if (self->hash_owners[_hash] == -1) {
      // Then **key** is not in this table and, if add()-ing, should be
      // registered under this hash.
      return _hash;
    }

    // This `_hash` has been claimed. Check if the owner of `_hash` is `key`:
    //      keys[hash_owners[_hash]] == key
    // Be careful here, memcmp() is really meant for sorting and it returns 0
    // if they are equal. 1 or -1 otherwise.
    if (!memcmp(self->keys + self->key_size * self->hash_owners[_hash],
                key, self->key_size)) {
        // **key** is already in the table. Return this hash which can be used
        // to locate the **key**.
        return _hash;
     }

    // Otherwise keep incrementing `_hash` until we either find a space or a
    // match.
    _hash = (_hash + 1) % self->max;
  }

  return -1;
}


ptrdiff_t HT_add(HashTable * self, void * key) {
  /* Add **key** to the table if it isn't already in it. Returns its location
     in either case. */

  // Check **key**'s hash in `self->hash_owners` (assuming this key is in the
  // table).
  ptrdiff_t _hash = HT_hash_for(self, key);

  // Safety check: If it's not there and there's not more space to put it in,
  // escape. This will be raised as an error in Python.
  if (_hash == -1) return -1;

  // Lookup the owner of the hash. This will be the location **key** if it's in
  // the table or -1 otherwise (no owner).
  ptrdiff_t index = self->hash_owners[_hash];

  // If **key** isn't there ...
  if (index == -1) {
    // ... claim the _hash. Record we're we about to put **key**.
    self->hash_owners[_hash] = index = self->length;

    // And append **key** to `self->keys`.
    // Note that memcpy()'s arguments are in an odd order:
    memcpy(self->keys + self->key_size * index,  // dest,
           key,  // source,
           self->key_size);  // number of bytes.

    self->length += 1;
  }

  return index;
}


ptrdiff_t HT_get(HashTable * self, void * key) {
  ptrdiff_t _hash = HT_hash_for(self, key);
  if (_hash == -1)
    return -1;
  return self->hash_owners[_hash];
}


/* The following are just the functions above but vectorized for fast bulk
   computation. */


ptrdiff_t HT_adds(HashTable * self, void * keys, ptrdiff_t * out, size_t len) {
  for (size_t i = 0; i < len; i++) {
    out[i] = HT_add(self, keys + (i * self->key_size));
    if (out[i] == -1)
      // Out of space - abort.
      return i;
  }
  return -1;
}


ptrdiff_t HT_gets(HashTable * self, void * keys, ptrdiff_t * out, size_t len) {
  for (size_t i = 0; i < len; i++) {
    out[i] = HT_get(self, keys + (i * self->key_size));
  }
  return 0;
}
