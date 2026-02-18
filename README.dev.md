# `mibiremo` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

```shell
# Create a virtual environment, e.g. with
python -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip setuptools

# (from the project root directory)
# install mibiremo as an editable package
python -m pip install --no-cache-dir --editable .
# install development dependencies
python -m pip install --no-cache-dir --editable .[dev]
# install documentation dependencies only
python -m pip install --no-cache-dir --editable .[docs]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Quick Start

For a quick overview of available development commands, run:

```shell
make help
```

This will show shortcuts to common development tasks like testing, linting, and building documentation.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `mibiremo`, but then builds the necessary virtual environments itself by simply running:

```shell
tox
```

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.## Running linters locally

For linting and sorting imports we will use [ruff](https://beta.ruff.rs/docs/). Running the linters requires an
activated virtual environment with the development tools installed.

```shell
# linter
ruff check .

# linter with automatic fixing
ruff check . --fix
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf).

## Testing docs locally

To build the documentation locally, first make sure `mkdocs` and its dependencies are installed:
```shell
python -m pip install .[doc]
```

Then you can build the documentation and serve it locally with
```shell
mkdocs serve
```

This will return a URL (e.g. `http://127.0.0.1:8000/mibiremo/`) where the docs site can be viewed.

Note that this will only create the "non-versioned" documentation, which should be fine for testing changes to the docs.
The versioned documentation is created using the python utility called [mike](https://github.com/jimporter/mike?tab=readme-ov-file#mike) and its corresponding [mkdocs integration](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/).
In general it should not be necessary to test this, but if necessary, use [the mike documentation](https://github.com/jimporter/mike?tab=readme-ov-file#viewing-your-docs) to inspect locally.

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

## Making a release

Before you make a new release:
1. Verify that the information in [`CITATION.cff`](CITATION.cff) is correct.
1. Make sure the [version has been updated](#versioning).
1. Run the unit tests with `pytest -v`

Now, make a [release on GitHub](https://github.com/MiBiPret/mibiremo/releases/new). The `publish.yml` workflow will publish the software on PyPI. GitHub-Zenodo integration will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.


## Getting a new `SONAR_TOKEN`
If the Sonar pipeline does not run for 60 days, the `SONAR_TOKEN` expires.
You can see the original announcement [in this thread](https://community.sonarsource.com/t/removing-inactive-tokens-after-60-days/142451/6).

The expiry happens silently, so any github action workflows using Sonar will appear to suddenly fail with cryptic errors such as `"ERROR Failed to query JRE metadata: . Please check the property sonar.token or the environment variable SONAR_TOKEN. "`. If you see this, it means you need to recreate the `SONAR_TOKEN` for this repository. You will need some admin privileges to carry out the following steps:

1. Go to 'My Account' on the sonarcloud.io site after logging in (or try following this URL: <https://sonarcloud.io/account>)
1. Click on the 'Security' tab (<https://sonarcloud.io/account/security>)
1. Enter a token name in the field under "Generate Tokens" and click "Generate Token"
1. Copy the text for the generated token (you will not be able to see this text again)
1. Go to the "Actions secrets and variables" section of the `mibiremo` repository on GitHub (<https://github.com/MiBiPreT/mibiremo/settings/secrets/actions>)
1. Edit the existing `SONAR_TOKEN` (or create it, if it does not exist). Paste in the token text you copied from sonarcloud earlier
1. Click "Update Secret"

This should result in the newly created `SONAR_TOKEN` being used in any new github workflow runs. You may also choose to rerun some previously failed workflows (these should now succeed).
