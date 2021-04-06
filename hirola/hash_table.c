// -*- coding: utf-8 -*-

#include "hash_table.h"


// Prototypes.
ptrdiff_t HT__claim(HashTable * self, void * key, ptrdiff_t _hash);


ptrdiff_t euclidean_modulo(ptrdiff_t x, ptrdiff_t base) {
  /* Python style modulo.
     Unlike C modulo, the output is still >= 0 for negative input. */
  x = x % ((ptrdiff_t) base);
  if (x < 0) x += base;
  return x;
}


#ifdef COUNT_COLLISIONS
// Used for debugging hash collisions.
size_t collisions = 0;
#endif


ptrdiff_t HT_hash_for(HashTable * self, void * key, bool its_not_there) {
  /* Either find **key**'s hash if **key** is in `self->keys` or
     choose an unused hash. */

  // Get an initial hash for `key`.
  ptrdiff_t __hash = self -> hash(key, self->key_size);

  // Search, starting from our initial hash:
  for (size_t j = 0; j < self->max; j++) {
    // Provided the hash function self->hash() is working well for the input
    // data, this loop should rarely require more than one iteration.
    ptrdiff_t _hash = euclidean_modulo(__hash, self->max);

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
    if (!its_not_there) {
      if (!memcmp(self->keys + self->key_size * self->hash_owners[_hash],
                  key, self->key_size)) {
          // **key** is already in the table. Return this hash which can be
          // used to locate the **key**.
          return _hash;
       }
     }

    #ifdef COUNT_COLLISIONS
    collisions += 1;
    #endif

    // Otherwise keep incrementing `__hash` until we either find a space or a
    // match.
    // Incrementing with a ridiculously big prime number instead of just adding
    // 1 helps to break up clusters of collisions (albeit inconsistently).
    __hash = euclidean_modulo(__hash + 118394396737867, self->max);
  }

  return -1;
}


ptrdiff_t HT_add(HashTable * self, void * key) {
  /* Add **key** to the table if it isn't already in it. Returns its location
     in either case. */

  // Check **key**'s hash in `self->hash_owners` (assuming this key is in the
  // table).
  ptrdiff_t _hash = HT_hash_for(self, key, false);

  // Safety check: If it's not there and there's not more space to put it in,
  // escape. This will be raised as an error in Python.
  if (_hash == -1) return -1;

  return HT__claim(self, key, _hash);
}


inline ptrdiff_t HT__claim(HashTable * self, void * key, ptrdiff_t _hash) {
  /* Write **key** to the table under the hash **_hash**. */

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


inline ptrdiff_t HT_add_new(HashTable * self, void * key) {
  /* Add **key** to the table without checking if it isn't already in it.
     Returns its location. Can lead to corruption if **key** is already
     present. */

  // This is identical to HT_add() but with the **its_not_there** argument to
  // HT_hash_for() set to true.
  ptrdiff_t _hash = HT_hash_for(self, key, true);
  return HT__claim(self, key, _hash);
}


ptrdiff_t HT_get(HashTable * self, void * key) {
  ptrdiff_t _hash = HT_hash_for(self, key, false);
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


ptrdiff_t HT_gets_no_default(HashTable * self, void * keys,
    ptrdiff_t * out, size_t len) {
  /* Like HT_gets() but returns an error code if a key is missing. */

  for (size_t i = 0; i < len; i++) {
    ptrdiff_t index = HT_get(self, keys + (i * self->key_size));
    if (index == -1)
      // This key is missing. Return its location which will be included in the
      // error message raised by Python.
      return i;
    out[i] = index;
  }
  // Return -1 to indicate that no keys were missing.
  return -1;
}


void HT_gets_default(HashTable * self, void * keys,
    ptrdiff_t * out, size_t len, size_t default_) {
  /* Like HT_gets() but allows a user defined default for missing keys. */

  for (size_t i = 0; i < len; i++) {
    ptrdiff_t index = HT_get(self, keys + (i * self->key_size));
    if (index == -1) index = default_;
    out[i] = index;
  }
}


void HT_contains(HashTable * self, void * keys, bool * out, size_t len) {
  /* Vectorised are ``keys in table``? For set().union()-like operations. */
  for (size_t i = 0; i < len; i++) {
    out[i] = HT_get(self, keys + (i * self->key_size)) != -1;
  }
}


void HT_copy_keys(HashTable * self, HashTable * other) {
  /* Vectorised copy contents from **self** into **other** without checking if
     each key is already there. */
  for (size_t i = 0; i < self->length; i++) {
    HT_add_new(other, self->keys + (i * self->key_size));
  }
}


void vectorise_hash(Hash hash, void * keys, int32_t * hashes, size_t key_size,
                    size_t length) {
  /* Apply a hash() function to an array of **keys**. Only used for testing. */
  for (size_t i = 0; i < length; i++)
    hashes[i] = hash(keys + i * key_size, key_size);
}
