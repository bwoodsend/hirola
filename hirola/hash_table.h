// -*- coding: utf-8 -*-
#ifndef hash_table_H
#define hash_table_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>


typedef int32_t (*Hash)(void * key, const size_t key_size);


typedef struct HashTable {
  const size_t max;
  const size_t key_size;
  ptrdiff_t * const hash_owners;
  void * const keys;
  size_t length;
  Hash hash;
  ptrdiff_t panic_at;
} HashTable;


#endif
