.. py:currentmodule:: hirola

==================
:class:`HashTable`
==================

.. autoclass:: HashTable
    :exclude-members: DEFAULT_SEED

    .. attribute:: DEFAULT_SEED

        The default value for `seed` if none is provided.

        This can be frozen with the ``HIROLA_HASH_SEED`` environment variable.
        Otherwise it's set to a randomised value on startup.
