==================
Welcome to hirola!
==================

∘
`MIT license <https://github.com/bwoodsend/hirola/blob/master/LICENSE>`_
∘
`PyPI <https://pypi.org/project/hirola>`_
∘
`Documentation <https://hirola.readthedocs.io/>`_
∘
`Source code <https://github.com/bwoodsend/hirola>`_
∘
`Bug reports <https://github.com/bwoodsend/hirola/issues>`_
∘
`Support <https://github.com/bwoodsend/hirola/discussions>`_

A vectorized hash table written in C for `fast
<https://hirola.readthedocs.io/en/latest/benchmarks.html>`_ ``set``/``dict``
like operations on NumPy arrays.

Hirola provides fast indexing and de-duplication of keys.
It can be used as an extension of `numpy.unique()
<https://numpy.org/doc/stable/reference/generated/numpy.unique.html>`_ and a
very light (20-30KB download size) and much faster alternative to
`pandas.Categorical()
<https://pandas.pydata.org/docs/reference/api/pandas.Categorical.categories.html>`_.
Hirola obtains its speed in the same way that NumPy does – vectorising,
translating into C and imposing the following constraints:

* Keys must all be of the same predetermined type and size.
* The maximum size of a table must be chosen in advance and managed explicitly.
* To get any performance boost, operations should be done in bulk.
* Elements can not be removed.


Installation
------------

Install Hirola with pip:

.. code-block:: console

    pip install hirola


Quickstart
----------

``HashTable``
*************

A ``HashTable`` can be though of as a ``dict`` but with only an enumeration for
values.
To construct an empty hash table:

.. code-block:: python

    import numpy as np
    import hirola

    table = hirola.HashTable(
        20,  # <--- Maximum size for the table - up to 20 keys.
        "U10",  # <--- NumPy dtype - strings of up to 10 characters.
    )

Keys may be added individually...

.. code-block:: python

    >>> table.add("cat")
    0

... But it's much more efficient to add in bulk.
The return value is an enumeration of when each key was first added.
Duplicate keys are not re-added.

.. code-block:: python

    >>> table.add(["dog", "cat", "moose", "gruffalo"])
    array([1, 0, 2, 3])


Multidimensional inputs give multidimensional outputs of matching shapes.

.. code-block:: python

    >>> table.add([["rabbit", "cat"],
    ...            ["gruffalo", "moose"],
    ...            ["werewolf", "gremlin"]])
    array([[4, 0],
           [3, 2],
           [5, 6]])

Inspect all keys added so far via the ``keys`` attribute.
(Note that, unlike ``dict.keys()``, it's a property instead of a method.)

.. code-block:: python

    >>> table.keys
    array(['cat', 'dog', 'moose', 'gruffalo', 'rabbit', 'werewolf', 'gremlin'],
          dtype='<U10')

Key indices can be retrieved with ``table.get(key)`` or just ``table[key]``.
Again, retrieval is NumPy vectorised and is much faster if given large arrays of
inputs rather than one at a time.

.. code-block:: python

    >>> table.get("dog")
    1
    >>> table[["moose", "gruffalo"]]
    array([2, 3])

Like the Python dict,
using ``table[key]`` raises a ``KeyError`` if keys are missing
but using ``table.get(key)`` returns a configurable default.
Unlike Python's dict, the default default is ``-1``.

.. code-block:: python

    >>> table["tortoise"]
    KeyError: "key = 'tortoise' is not in this table."
    >>> table.get("tortoise")
    -1
    >>> table.get("tortoise", default=99)
    99
    >>> table.get(["cat", "bear", "tortoise"], default=[100, 101, 102])
    array([  0, 101, 102])


Choosing a ``max`` size
.......................

Unlike Python's ``set`` and ``dict``, ``Hirola`` does not manage its size
automatically (although `it can be reconfigured to <automatic-resize>`_).
To prevent wasted resizing (which is what Python does under the hood),
you have full control of and responsibility for how much space the table uses.
Obviously the table has to be large enough to fit all the keys in it.
Additionally, when a hash table gets to close to full it becomes much slower.
Depending on how much you favour speed over memory you should add 20-50% extra
headroom.
If you intend to a lot of looking up of the same small set of values then it can
continue to run faster if you increase ``max`` to 2-3x its minimal size.


Structured key data types
.........................

To indicate that an array axis should be considered as a single key,
use NumPy's structured dtypes.
In the following example, the data type ``(points.dtype, 3)``
indicates that a 3D point - a triplet of floats -
should be considered as one object.
See ``help(hirola.HashTable.dtype)`` for more information of specifying dtypes.
Only the last axis or last axes may be thought of as single keys.
For other setups, first convert with ``numpy.transpose()``.

.. code-block:: python

    import numpy as np
    import hirola

    # Create a cloud of 3D points with duplicates. This is 3000 points in total,
    # with up to 1000 unique points.
    points = np.random.uniform(-30, 30, (1000, 3))[np.random.choice(1000, 3000)]

    # Create an empty hash table.
    # In practice, you generally don't know how many unique elements there are
    # so we'll pretend we don't either an assume the worst case of all 3000 are
    # unique. We'll also give 25% padding for speed.
    table = hirola.HashTable(len(points) * 1.25, (points.dtype, 3))

    # Add all points to the table.
    ids = table.add(points)

Duplicate-free contents can be accessed from ``table.keys``:

.. code-block:: python

    >>> table.keys  # <--- These are `points` but with no duplicates.
    array([[  3.47736554, -15.17112511,  -9.51454466],
           [ -6.46948046,  23.64504329, -16.25743105],
           [-27.02527253, -16.1967225 , -10.11544157],
           ...,
           [  3.75972597,   1.24130412,  -8.14337206],
           [-13.62256791,  11.76551455, -13.31312988],
           [  0.19851678,   4.06221179, -22.69006592]])
    >>> table.keys.shape
    (954, 3)

Each point's location in ``table.keys`` is returned by ``table.add()``,
like ``numpy.unique(..., return_args=True)``.

.. code-block:: python

    >>> ids  # <--- These are the indices in `table.keys` of each point in `points`.
    array([  0,   1,   2, ..., 290, 242, 669])
    >>> np.array_equal(table.keys[ids], points)
    True

Lookup the indices of points without adding them using ``table.get()``.


.. _automatic-resize:

Handling of nearly full hash tables
...................................

``HashTable``\ s become very slow when almost full.
As of v0.3.0, an efficiency warning will notify you if a table exceeds 90% full.
This warning can be reconfigured into an error, silenced or set to resize the
table automatically to make more room.
These are demonstrated in the example constructors below:

.. code-block:: python

    # The default: Issue a warning when the table is 90% full.
    hirola.HashTable(..., almost_full=(0.9, "warn"))

    # Disable all "almost full" behaviours.
    hirola.HashTable(..., almost_full=None)

    # To consider a table exceeding 80% full as an error use:
    hirola.HashTable(..., almost_full=(0.8, "raise"))

    # To automatically triple in size whenever the table exceeds 80% full use:
    hirola.HashTable(..., almost_full=(0.8, 3.0))

Resizing tables is slow (it's only marginally optimized beyond creating a new
bigger table and ``.add()``\ -ing the existing keys) which is why it's not
enabled by default. It should be avoided unless you really have no idea how big
your table will need to be and favour the memory savings of not overestimating
over raw speed.


Recipes
*******

A ``HashTable`` can be used to replicate a `dict <as-a-dict>`_,
`set <as-a-set>`_ or a `collections.Counter <as-a-collections.Counter>`_.
These examples below might turn into their own proper classes in the future but
so far I've never come across a real use case where they would actually fit.


.. _as-a-dict:

Using a ``HashTable`` as a ``dict``
...................................

A ``dict`` can be imitated using a ``HashTable()`` with a second array for
values.
The output of ``HashTable.add()``  and ``HashTable.get()`` should be used as
indices of ``values``:

.. code-block:: python

    import numpy as np
    import hirola

    # The `keys` - will be populated with names of countries.
    countries = hirola.HashTable(40, (str, 20))
    # The `values` - will be populated with the names of each country's capital city.
    capitals = np.empty(countries.max, (str, 20))

Add or set items using the pattern ``values[table.add(key)] = value``:

.. code-block:: python

    capitals[countries.add("Algeria")] = "Al Jaza'ir"

Or in bulk:

.. code-block:: python

    new_keys = ["Angola", "Botswana", "Burkina Faso"]
    new_values = ["Luanda", "Gaborone", "Ouagadougou"]
    capitals[countries.add(new_keys)] = new_values

Like Python dicts, the syntax to overwrite values is exactly the same as to
write them.

Retrieve values with ``values[table[key]]``:

.. code-block:: python

    >>> capitals[countries["Botswana"]]
    'Gaborone'
    >>> capitals[countries["Botswana", "Algeria"]]
    array(['Gaborone', "Al Jaza'ir"], dtype='<U20')

View all keys and values with ``table.keys`` and ``values[:len(table)]``.
A ``HashTable`` remembers the order keys were first added so this dict is
automatically a sorted dict.

.. code-block:: python

    # keys
    >>> countries.keys
    array(['Algeria', 'Angola', 'Botswana', 'Burkina Faso'], dtype='<U20')
    # values
    >>> capitals[:len(countries)]
    array(["Al Jaza'ir", 'Luanda', 'Gaborone', 'Ouagadougou'], dtype='<U20')

Depending on the usage scenario,
it may or may not make sense to want an equivalent to  ``dict.items()``.
If you do want an equivalent,
use ``numpy.rec.fromarrays([table.keys, values[:len(table)]])``,
possibly adding a ``names=`` option:

.. code-block:: python

    >>> np.rec.fromarrays([countries.keys, capitals[:len(countries)]],
    ...                   names="countries,capitals")
    rec.array([('Algeria', "Al Jaza'ir"), ('Angola', 'Luanda'),
               ('Botswana', 'Gaborone'), ('Burkina Faso', 'Ouagadougou')],
              dtype=[('countries', '<U20'), ('capitals', '<U20')])

If the keys and values have the same dtype then ``numpy.c_`` works too.

.. code-block:: python

    >>> np.c_[countries.keys, capitals[:len(countries)]]
    array([['Algeria', "Al Jaza'ir"],
           ['Angola', 'Luanda'],
           ['Botswana', 'Gaborone'],
           ['Burkina Faso', 'Ouagadougou']], dtype='<U20')


.. _as-a-set:

Using a ``HashTable`` as a ``set``
..................................

To get set-like capabilities from a ``HashTable``,
leverage the ``contains()`` method.
For these examples we will experiment with integer multiples of 3 and 7.

.. code-block:: python

    import numpy as np

    of_3s = np.arange(0, 100, 3)
    of_7s = np.arange(0, 100, 7)

We'll only require one array to be converted into a hash table.
The other can remain as an array.
If both are hash tables, simply use one table's ``keys`` attribute as the array.

.. code-block:: python

    import hirola

    table_of_3s = hirola.HashTable(len(of_3s) * 1.25, of_3s.dtype)
    table_of_3s.add(of_3s)

Use ``table.contains()`` as a vectorised version of ``in``.

.. code-block:: python

    >>> table_of_3s.contains(of_7s)
    array([ True, False, False,  True, False, False,  True, False, False,
            True, False, False,  True, False, False])

From the above, the common set operations can be derived:

*   ``set.intersection()`` - Values in the array and in the set:

.. code-block:: python

        >>> of_7s[table_of_3s.contains(of_7s)]
        array([ 0, 21, 42, 63, 84])

*   Set subtraction - Values in the array which are not in the set:

.. code-block:: python

        >>> of_7s[~table_of_3s.contains(of_7s)]
        array([ 7, 14, 28, 35, 49, 56, 70, 77, 91, 98])

*   ``set.union()`` - Values in either the table or in the tested array (with no
    duplicates):

.. code-block:: python

        >>> np.concatenate([table_of_3s.keys, of_7s[~table_of_3s.contains(of_7s)]], axis=0)
        array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
               51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99,
                7, 14, 28, 35, 49, 56, 70, 77, 91, 98])


.. _`as-a-collections.Counter`:

Using a ``HashTable`` as a ``collections.Counter``
..................................................

For this example,
let's give ourselves something a bit more substantial to work on.
Counting word frequencies in Shakespeare's Hamlet play is the
trendy example for ``collections.Counter`` and it's what we'll use too.

.. code-block:: python

    from urllib.request import urlopen
    import re
    import numpy as np

    hamlet = urlopen("https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt").read()
    words = np.array(re.findall(rb"([\w']+)", hamlet))

A counter is just a ``dict`` with integer values and a ``dict`` is just a hash
table with a separate array for values.

.. code-block:: python

    import hirola

    word_table = hirola.HashTable(len(words), words.dtype)
    counts = np.zeros(word_table.max, dtype=int)

The only new functionality that is not defined in `using a hash table as a dict
<as-a-dict>`_ is the ability to count keys as they are added.
To count new elements use the rather odd line
``np.add(counts, table.add(keys), 1)``.

.. code-block:: python

    np.add.at(counts, word_table.add(words), 1)

This line does what you might expect ``counts[word_table.add(words)] += 1`` to
do but, due to the way NumPy works,
the latter form fails to increment each count more than once if ``words``
contains duplicates.

Use NumPy's indirect sorting functions to get most or least common keys.

.. code-block:: python

    # Get the most common word.
    >>> word_table.keys[counts[:len(word_table)].argmax()]
    b'the'

    # Get the top 10 most common words. Note that these are unsorted.
    >>> word_table.keys[counts[:len(word_table)].argpartition(-10)[-10:]]
    array([b'it', b'and', b'my', b'of', b'in', b'a', b'to', b'the', b'I',
           b'you'], dtype='|S14')

    # Get all words in ascending order of commonness.
    >>> word_table.keys[counts[:len(word_table)].argsort()]
    array([b'END', b'whereat', b"griev'd", ..., b'to', b'and', b'the'],
          dtype='|S14')


A Security Note
---------------

Unlike the builtin ``hash()`` used internally by Python's ``set`` and ``dict``,
``hirola`` does not randomise a hash seed on startup
making an online server running ``hirola`` more vulnerable to denial of service
attacks.
In such an attack, the attacker clogs up your server by sending it requests that
he/she knows will cause hash collisions and therefore slow it down.
Whereas a Python hash table's size is always predictably the next power of 8
above ``len(table) * 3 / 2``, a ``hirola.HashTable()`` may be any size meaning
that you can make an attack considerably more difficult by adding a little
randomness to the sizes of your hash tables.
But if your writing an online server
which performs dictionary lookup based on user input
and your user-base doesn't like you much
or you have some very spiteful below-the-belt competitors
then I recommend that you don't use this library.
