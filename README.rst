=======
Hirola
=======

..  from urllib.parse import quote
    "https://img.shields.io/badge/"
    quote("Python- {}-blue.svg".format(" | ".join(["3.6", "3.7", "3.8", "3.9"])))

.. image::
    https://img.shields.io/badge/
    Python-%203.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg

NumPy vectorized hash table written in C for fast (roughly 10x faster) ``set``/``dict``
like operations.

* Free software: `MIT license <https://github.com/bwoodsend/hirola/blob/master/LICENSE>`_
* Documentation: `<https://hirola.readthedocs.io/>`_
* Source code: `<https://github.com/bwoodsend/hirola/>`_
* Releases: `<https://pypi.org/project/hirola/>`_

A ``hirola.HashTable`` is to ``dict`` what ``numpy.array`` is to ``list``.
By imposing some constraints, vectorising, and translating into C, the speed
can be improved dramatically.
These constraints are:

* Keys must all be of the same predetermined type and size.
* The maximum size of a table must be chosen in advance.
* To get any performance boost, operations should be done in bulk.
* Elements can not be removed.

If any of the above are not satisfied for your use case then don't use
hirola.

.. highlight:: python


Installation
------------

Install Hirola with pip:

.. code-block:: console

    pip install hirola


Quickstart
----------

``HashTable``
*************

At the rawest and dirtiest level lives the ``HashTable`` class.
A ``HashTable`` can be though of as a ``dict`` but with only an enumeration for
values.
To construct an empty hash table::

    import numpy as np
    from hirola import HashTable

    table = HashTable(
        20,  # <--- Maximum size for the table - up to 20 keys.
        "U10",  # <--- NumPy dtype - strings of up to 10 characters.
    )

Keys may be added individually... ::

    >>> table.add("cat")
    0

... But it's much more efficient to add in bulk.
The return value is an enumeration of when each key was first added.
Duplicate keys are not re-added. ::

    >>> table.add(["dog", "cat", "moose", "gruffalo"])
    array([1, 0, 2, 3])


Multidimensional inputs give multidimensional outputs of matching shapes. ::

    >>> table.add([["rabbit", "cat"],
    ...            ["gruffalo", "moose"],
    ...            ["werewolf", "gremlin"]])
    array([[4, 0],
           [3, 2],
           [5, 6]])

Inspect all keys added so far via the ``keys`` attribute.
(Note that, unlike ``dict.keys()``, it's a property instead of a method.) ::

    >>> table.keys
    array(['cat', 'dog', 'moose', 'gruffalo', 'rabbit', 'werewolf', 'gremlin'],
          dtype='<U10')

Key indices can be retrieved with ``table.get(key)`` or just ``table[key]``.
Again, retrieval is NumPy vectorised and is much faster if given large arrays of
inputs rather than one at a time. ::

    >>> table.get("dog")
    1
    >>> table[["moose", "gruffalo"]]
    array([2, 3])

Like the Python dict,
using ``table[key]`` raises a ``KeyError`` if keys are missing
but using ``table.get(key)`` returns a configurable default.
Unlike Python's dict, the default is ``-1``. ::

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
automatically.
You have full control of and responsibility for how much space the table uses.
This is to prevent wasted resizing (which is what Python does under the hood).
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
See ``help(HashTable.dtype)`` for more information of specifying dtypes.
Only the last axis or last axes may be thought of as single keys.
For other setups, first convert with ``numpy.transpose()``.

.. code-block:: python

    import numpy as np
    from hirola import HashTable

    # Create a cloud of 3D points with duplicates. This is 3000 points in total,
    # with up to 1000 unique points.
    points = np.random.uniform(-30, 30, (1000, 3))[np.random.choice(1000, 3000)]

    # Create an empty hash table.
    # In practice, you generally don't know how many unique elements there are
    # so we'll pretend we don't either an assume the worst case of all 3000 are
    # unique. We'll also give 25% padding for speed.
    table = HashTable(len(points) * 1.25, (points.dtype, 3))

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
similarly to ``numpy.unique(..., return_args=True)``.

.. code-block:: python

    >>> ids  # <--- These are the indices in `table.keys` of each point in `points`.
    array([  0,   1,   2, ..., 290, 242, 669])
    >>> np.array_equal(table.keys[ids], points)
    True

Lookup the indices of points without adding them using ``table.get()``.


A Minor Security Implication
----------------------------

Unlike the builtin ``hash()`` used internally by Python's ``set`` and ``dict``,
``hirola`` does not randomise a hash seed on startup
making an online server running ``hirola`` more vulnerable to denial of service
attacks.
In such an attack, the attacker clogs up your server by sending it requests that
he/she knows will cause hash collisions and therefore slow it down.
You can make this considerably more difficult by adding a little
randomness to the sizes of your hash tables.
But if your writing an online server
which performs dictionary lookup based on user input
and your user-base doesn't like you much
or you have some very spiteful below-the-belt competitors
then I recommend that you don't use this library.

Credits
-------

This package was initially created with Cookiecutter_ and a fork of the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
