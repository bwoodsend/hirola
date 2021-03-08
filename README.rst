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



Credits
-------

This package was initially created with Cookiecutter_ and a fork of the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
