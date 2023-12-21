Benchmarks
==========

The relative performance of `hirola` against other methods for indexing or
deduplicating keys depends heavily on the type of keys being indexed. Hash table
based methods such as `hirola.HashTable` or Python's builtin `set` prefer high
entropy inputs whereas sorting based methods like `numpy.unique` or
``numpy_indexed`` prefer keys with patterns (where a pattern could be something
as benign as all keys being within a given numerical range).

Performance is also affected by the number of keys being indexed, the size of
the keys, and in `hirola`\ 's case, the amount of padding between the table's
maximum and actual capacities (:attr:`table.max <hirola.HashTable.max>` over
:attr:`table.length <hirola.HashTable.length>`).

Also worth bearing in mind when looking at these benchmarks is that very few of
these indexing methods are really functionally equivalent. `hirola`
deduplicates, preserving the order of first appearance and returning indices
which can be used to reproduce the original input whereas `numpy.unique` sorts
the keys without returning indices by default – if your use case requires order
of first appearance or those indices then setting `numpy.unique`\ 's
``return_indices=True`` flag and doing some awkward `numpy.argsort` would
greatly reduce performance whereas if you wanted your keys sorted then this is a
boon to `numpy.unique`. Similarly, since converting between a Python `set` and a
`numpy.ndarray` is remarkably expensive, using one when your data is in the
other format will most likely slow your code down in spite of what these
benchmarks say is the fastest method.

The following graphs show the per-key indexing time of various indexing methods
(low numbers mean better performance) when given key sets of a range of sizes.
Each graph represents a different kind of data (random binary, floats, integers,
strings, UTF-8 encoded strings) being indexed. Choose the graph that best
matches whatever it is that you are trying to index. Each trace on a graph
represents a different method for indexing keys, whose implementations are
summarised below:

* **hirola x{multiplier}** is ``hirola.HashTable().add()`` where the multiplier
  is the ratio of the table's :attr:`~hirola.HashTable.max` size over the number
  of unique keys in the dataset. The time to construct the table is **included**
  in the timings.

* **set()** is as you'd expect – it calls ``set()`` on the list of keys. Since
  NumPy types generally aren't hashable, where necessary, the input NumPy array
  is converted into Python native types (normally a list of bytes) beforehand.
  The conversion time is **excluded** from the benchmark times.

* **numpy.unique()**, **numpy.unique(return_indices=True)** and
  **numpy_indexed.unique()** are just function calls. Since they all only accept
  one dimensional inputs, any multi dimensional arrays are converted to single
  dimensional arrays of bytes. This conversion time is **excluded** from
  benchmarks.

* **pandas.Categorical()** is ``pandas.Categorical(keys).codes``.

.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

.. include:: _benchmarks.txt
