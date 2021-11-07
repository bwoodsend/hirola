# Welcome to hirola's test suite!

This guide explains how to run the test suite.

## Test requirements

Test requirements are lists under `extras_require` in `setup.py` and can be
installed by running (in the root of this repository):

```shell
pip install .[test]
```

You may use the `-e` parameter.

## Run the tests

The test-suite is a `pytest` one. Use (in the root of this repository):

```shell
pytest tests
```

Or, if your working directory is not the root of this repo:

```shell
pytest /full/or/relative/path/to/tests
```

## Testing with PyInstaller

hirola aims to be out-the-box compatible with
[PyInstaller](https://github.com/pyinstaller/pyinstaller/). To enforce this we
run the test-suite inside a frozen PyInstaller application. Should it turn out
that PyInstaller does need special help when compiling hirola
then that *help* goes in a *hook* file (see
`hirola/__pyinstaller/hook-hirola.py`).

### Setup

To test requires a PyInstaller that is recent enough to allow libraries to
provide their own hooks and to use the `collect_entry_point()` function:

``` shell
pip install "PyInstaller>=4.3"
```

### Build

Next `cd` inside the `./tests/PyInstaller_/` directory and run:

```shell
python -m PyInstaller --clean --noconfirm frozen-pytest.spec
```

The `--noconfirm` is optional. And `--clean` is only necessary if you have
changed the hirola source code.

This creates an executable at `./dist/frozen-pytest/frozen-pytest` (or
`dist\frozen-pytest\frozen-pytest.exe` on Windows).

### Test

This executable is just `pytest.main()` with hirola
installed in it. The tests themselves are excluded. Run it as you would run
`pytest`. Assuming the working directory is still `./tests/PyInstaller/` use:

```shell
dist/frozen-pytest/frozen-pytest ../
```

Avoid running it with the root of this repo as the current working directory.
Otherwise you may unintentionally use the original source hirola
rather than the bundled one.

### Rebuilding

I've just changed something, if/when do I need to recompile? That depends on
what you changed.

If you've modified either the source code or the hooks for hirola
then you must re-run:

```shell
python -m PyInstaller --clean frozen-pytest.spec
```

before re-testing for the changes to propagate.

If you modify either `frozen-pytest.py` or `frozen-pytest.spec` in the
`tests/PyInstaller_` folder then:

```shell script
python -m PyInstaller frozen-pytest.spec
```

is sufficient. However, you should only alter these for experimenting.
Ultimately, changes necessary to make it work should go in hooks (see
`hirola/__pyinstaller/`).

And if you modify the tests code then you don't have to do anything. The tests
are run from source and are not part of the compiled application.

If in doubt, there is no harm in rebuilding or using `--clean` unnecessarily -
it's just a bit tedious...
