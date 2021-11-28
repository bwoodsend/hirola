.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated!

You can contribute in many ways:


Report Bugs
~~~~~~~~~~~

Report bugs by creating an `issue on Github
<https://github.com/bwoodsend/hirola/issues>`_.


Getting Help
~~~~~~~~~~~~

You can get help directly from the package authors by asking questions on
`Github discussions <https://github.com/bwoodsend/hirola/discussions/>`_.


Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with *bug* and *help
wanted* is open to whoever wants to implement it. If it hasn't been reported
already, please `report a bug <Report Bugs>`_ before trying to fix it. Indicate
on the bug report that you have a fix or at least an idea to fix it with.


Adding or Requesting New Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New features, whether you want to add it yourself or simply request it should be
proposed as `discussions on Github <https://github.com/bwoodsend/hirola/discussions/>`_
first. However, be warned that I am very picky about what should and should not
be added to ``hirola``. Features will be rejected if
they don't satisfy at least most of these criteria.

#. You are both willing and able to do at least some of the work.

#. I'm a big believer of the Linux *small packages that do one thing well*
   philosophy. This feature should not introduce new overhead or dependencies,
   including optional dependencies (which are even worse in my mind). Such a
   feature should instead be its own package which depends on both ```hirola`` and these other dependencies.

#. You can justify the general need for this feature - not just that you
   specifically want it for your given use case.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

#. There are no nonfunctional or stylistic changes to existing code. I don't
   give a wet-slap about PEP8 - I care very much about the signal to noise ratio
   of ``git log -S "new code"``, ``git log -- filename.py`` and ``git diff``.

#. The pull request should work for all supported Python versions and
   PyInstaller. Run::

        pip install -e .[test]

   to install test requirements. And::

        pytest

   to test everything. See also the readme in the ``tests`` folder.

#. Running the full test suite should provide 100% coverage.

#. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a Google or NumPy style
   docstring, add it to the docs stubs (see ``docs/source/stubs``) if it isn't
   already collected.

#. Python source code should be formatted by `yapf
   <https://github.com/google/yapf>`_. Install with::

        pip install -r requirements-dev.txt

   and run from the root of this repo with::

        yapf -rip .

   Aim to do this with every commit. I don't want to see any *run yapf* commits.
   I have no interest in PEP8 compliance so there is no need to appease flake8.

#. Similarly, C source code should be formatted by clang-format (typically
   installable with clang or standalone via your system package manager) using::

        clang-format -i **.h **.c

    (Note that this must be ran using a Unix shell such as bash - not the
    generic Windows command prompt.)

#. I'm fussy about git history. Follow `these guidelines
   <https://chris.beams.io/posts/git-commit/>`_ for writing good commit messages
   but include a ``.`` at the end of the 1st line. Use ``git rebase -i`` to
   clear up any mistakes you make. Alternatively, if your understanding of git
   is weak, then limit your pull request to one change only (so that the history
   can be squashed automatically) and make as much mess as you like.

