.. highlight:: shell

==================
Contributing Guide
==================

Contributions are welcome, and they are greatly appreciated!


Quick Links
---------------------------

* `Report bugs`_
* `Request features`_
* `Ask for help`_


Guide to Submitting Changes
---------------------------

Before you start making changes, please open either a bug report or a feature
request.
That way, I can tell you if what you are about to do is likely to be accepted
before you invest your time in it.

This guide assumes that you have at least a basic knowledge of git, working
with command line tools and that you have a GitHub account.
If that is not the case then attempting these changes yourself is generally not
a good idea.
Please do not try to make changes, no matter how trivial, via GitHub's web UI -
it looks tantilisingly simple but invariably ends in carnage...

If you get stuck anywhere throughout the following then please either push your
current changes, submit a pull request but mark it as a draft or just `ask for
help`_!
As a rough rule of thumb, if you've gotten nowhere in the last 15 minutes, it's
time to ask for help.


Setting up
..........

The following steps will set you up with an editable installation of hirola and
its test dependencies.

1. Either create and activate a virtual environment or run::

    pip uninstall hirola

   so that your original installation is safely out of the way.

2. Fork_ this repo.

3. Clone and open a terminal inside your local clone. ::

    git clone https://github.com/your-username/Hirola.git
    cd Hirola

4. Install locally by running::

    pip install -e .[test]

   Note, if you normally use ``pip3`` instead of ``pip`` then do so here.
   The above command tells pip to install your local clone (the ``.``)
   in editable mode (the ``-e`` flag)
   and additionally install Hirola's test requirements (the ``[test]``).
   The first time you run this may take a while if your internet connection is
   poor.

5. Compile the C code. (Similarly replace ``python`` with ``python3`` if that's
   what you normally do.) ::

    pip install toml
    python setup.py -q build

   If you don't have gcc installed then this will issue a
   ``cslug.exceptions.NoGccError()`` with instructions on how to install gcc.
   Follow those then retry the above command.

6. Do a run of the test suite so that you know your starting point works. ::

    pytest

   All non-skipped tests should pass and you should get a nice green message
   saying *Required test coverage of 100% reached*.
   If that is not the case then
   `please report it <https://github.com/bwoodsend/Hirola/issues/new>`_.

7. Create and switch to a new branch to make your changes in. ::

    git switch -c add-foo-feature

You now have a basic functional development environment. Go ahead and hack away!


Codebase Roadmap
................

The code base is hopefully small enough that you don't need a map but here's
a brief one irregardless.

* The ``hirola`` folder is what you ultimately get when you
  ``pip install hirola``.
  This is where you will add your features or fix your bugs.
* The documentation source code lives in ``docs/source`` but it also pulls in
  the contents of the top level ``README.rst`` from the root of this repository.
  Additionally the stub files located in ``docs/source/reference`` import the
  docstrings from hirola's classes and functions to form the API reference.
* The tests live in the ``tests`` directory.
* Continuous integration jobs are defined in ``.github/workflows``.

Pretty much everything outside of those directories is just project boilerplate
and configuration.

Inside the ``hirola`` directory, the main files are:

* ``hash_table.h`` which defines the ``HashTable()`` class (or rather structure)
  used throughout the C code to hold the various components of a hash table.
  Note that this ``HashTable()`` structure is not the ``hirola.HashTable()``
  class you typically interact with.
  It's just a handy C bucket which ultimately becomes the ``_raw`` attribute of
  ``hirola.HashTable()``.
* ``hash_table.c`` is where all the work happens. Any functions in there whose
  names start with ``HT_`` are intended to be ``HashTable`` methods but be aware
  that this is purely just a convention.
* ``_hash_table.py`` provides a user friendly wrapper class
  (``hirola.HashTable()``) around the C functions, adding type checking, input
  normalisation, output checking and whatever else is needed to keep C happy.

The ``slug = CSlug(...)``  line near the top of ``_hash_table.py`` is the glue
between the C and Python code.
Any function defined in either of the C source files becomes an attribute of
``slug.dll`` (e.g. ``void foo()`` becomes ``slug.dll.foo()`` in Python).
For the most part, types are automatic but see cslug's
`arrays <https://cslug.readthedocs.io/en/latest/arrays-and-buffers/arrays-and-buffers.html>`_
and `NumPy arrays <https://cslug.readthedocs.io/en/latest/arrays-and-buffers/numpy.html>`_
tutorials for passing arrays in and out of the C code.


Commands Cheatsheet
...................

A list of terminal commands you are likely to need for day to day development.
Unless specified otherwise, these should all be ran from the repository's root.



Compile or recompile the C code
+++++++++++++++++++++++++++++++

Whenever you modify C code, you'll have to trigger a recompile for the changes
to take effect.

* From terminal (requires restarting any open Python consoles):
  ``python setup.py -q build``
* From Python (no need to restart anything):
  ``from hirola._hash_table import slug; slug.make()``
* With hash collision metrics enabled (only needed if you're writing hash functions):
    * Unix:  ``CC_FLAGS='-D COUNT_COLLISIONS' python setup.py -q build``
    * Windows:
      ``set CC_FLAGS="-D COUNT_COLLISIONS" && python setup.py -q build``
* With the clang compiler instead of gcc (requires installing clang):
    * Unix: ``CC=clang python setup.py -q build``
    * Windows: ``set CC=clang && python setup.py -q build``
* Clean (remove all generated files): ``git clean -Xdf hirola/``


Test
++++

To run the test suite, use pytest_:

* Run everything: ``pytest``
* Run everything including the tests normally skipped: Recompile with
  ``COUNT_COLLISIONS`` enabled (see above) then run ``pytest`` as usual.
* Run everything but stop on the first failure: ``pytest -x``
* Run a single test file (ignore the *FAIL Required test coverage of
  100% not reached* error it issues): ``pytest tests/test_hash_table.py``
* Run a single test function by name: ``pytest -k test_automatic_resize``
* Run a single test function by its full path:
  ``pytest tests/test_hash_table.py::test_automatic_resize``

New tests can be added by defining functions whose names starts with ``test_``
in python files whose name also starts with ``test_`` inside the ``tests``
folder.
Tests should be ordered so that low level tests happen before high level tests
so that the first test to fail (as given by ``pytest -x``) indicates exactly
where the break is rather than indicating that a more complex function is
broken as a side effect of the lower level function's being broken.
The per-file running order is determined by ``pytestmark = pytest.mark.order()``
calls at the top of each file and within each file, tests are ordered simply by
their line numbers.


Run coverage
++++++++++++

Coverage tells us which lines of code were never ran when running the test
suite.
The test suite automatically collects coverage statistics.

* Do a full run of the test suite: ``pytest``
* Generate an HTML report: ``coverage html``
* View said report:
    * Linux: ``xdg-open htmlcov/index.html``
    * macOS or FreeBSD: ``open htmlcov/index.html``
    * Windows: ``start htmlcov/index.html``


Run automatic code formatter
++++++++++++++++++++++++++++

* Install with: ``pip install -r requirements-dev.txt``
* Run on all Python files: ``yapf -rip .``
* Run on one file: ``yapf -i hirola/_hash_table.py``


Build documentation
+++++++++++++++++++

All documentation commands should be ran inside the ``docs`` folder.

* Install docs requirements: ``pip install -r requirements.txt``
* Build: ``make html``
* View:
    * Linux: ``xdg-open build/html/index.html``
    * macOS or FreeBSD: ``open build/html/index.html``
    * Windows: ``start build/html/index.html``


Trigger continuous integration
++++++++++++++++++++++++++++++

Continuous integration allows us to quickly test all platforms and Python
versions.
First push your changes to GitHub then either:

* Trigger from the web UI:

  1. Go to your fork's GitHub page.
  2. Select the **Actions** tab.
  3. Say yes if it prompts you to enable actions.
  4. On the left hand side, select **Test**.
  5. Press **Run Workflow**.
  6. Select the branch you are working on from the drop down menu.
  7. Press the green **Run Workflow** button.
  8. Wait a few seconds then refresh the page.
     Your new job should appear below.

* Trigger using `GitHub's CLI`_::

    gh workflow run --ref=your-branch-name test.yml

  Then see it running by checking the **Actions** tab on your fork.


Benchmarking
............

There is a really crude benchmark script which compares the speed of hirola
against Python's ``set()``.
The number it emits is how many times faster hirola is (i.e. big number is
better).
For historical reasons, it is invoked via::

    python tests/benchmarks.py benchmark

I am in the process of replacing this script.

Please note that the binaries on PyPI are compiled with clang instead of gcc.
Clang produces binaries which are about 20% faster so unless you also compile
with clang, it is not meaningful to compare to a hirola downloaded from PyPI.



Before Submission
.................

Before you submit a pull request, here is a checklist of things that I am likely
to moan about if your changes don't meet the criteria below.

#. Make sure that there are no nonfunctional or stylistic changes to existing
   code. I don't give a wet-slap about PEP8 -
   I care very much about the signal to noise ratio of
   ``git log -S "new code"``, ``git log -- filename.py`` and ``git diff`` as
   well as the ability to merge, cherry-pick and rebase without merge conflicts.

#. The test suite passes with 100% coverage (see `Test`_ and `Run coverage`_).
   If your adding code then this means that you will also have to add tests
   to keep coverage happy.

#. If adding functionality, the docs need to be updated.
   Add `Google style docstrings`_ to new classes or functions,
   ensure that they appear somewhere in the API reference section of the docs
   and, unless the feature is quite niche, find a good place to add it to the
   `README.rst`_.

#. There is a clear distinction between public and private API.
   Anything that is intended to be used by end users should be documented.
   Any functions that aren't intended for use should have an underscore
   prefixed name or be defined in an underscore prefixed submodule to serve as
   a signal both to users and IDE code completions not to use them.
   Hidden, undocumented or barely documented functionality leads to nightmares
   where users
   don't know what they can safely use without fear of their code breaking after
   upgrading hirola and hirola developers can't change anything for fear of
   breaking someone else's downstream project.

#. Python source code should be formatted by yapf (see
   `Run automatic code formatter`_).

#. Proper grammar is used for anything textual.
   This means capital letters, full stops and no skipping the joining words -
   this goes for comments, documentation, docstrings and commit messages.

That's you, go ahead and submit...

If you wish, append *By your name / username / email / URL / some other
piece of information you wish to be identified by* to a commit message and I
will add it to the credits section of the README.


.. _`Report bugs`: https://github.com/bwoodsend/Hirola/issues/new?&template=bug-report.yml
.. _`Request features`: https://github.com/bwoodsend/Hirola/issues/new?&template=feature-request.yml
.. _`Ask for help`: https://github.com/bwoodsend/Hirola/discussions
.. _Fork: https://github.com/bwoodsend/Hirola/fork
.. _`GitHub's CLI`: https://github.com/cli/cli#github-cli
.. _`Google style docstrings`: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
.. _`README.rst`: https://github.com/bwoodsend/Hirola#readme
.. _pytest: https://docs.pytest.org/en/6.2.x/
