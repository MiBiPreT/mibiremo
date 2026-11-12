# Documentation for `mibiremo` python package

MiBiReMo (Microbiome Bioremediation Reactive-transport Module) is an open-source Python package for the simulation and design of soil and groundwater bioremediation systems. It is part of the MiBiPreT (Micro-Bioremediation Prediction Tool) suite, developed within the [MIBIREM](https://www.mibirem.eu/) bioremediation toolbox. See the [introduction](introduction.md) for an overview.

## Installation

### Installation of stable release from PyPI

Use `pip` to install the most recent stable release of `mibiremo` from PyPI as follows:

```console
pip install mibiremo
```

### Installation of most recent development version

To install mibiremo from the GitHub repository directly, do:

```console
git clone git@github.com:MiBiPreT/mibiremo.git
cd mibiremo
python -m pip install .
```

Note that this is the (possibly unstable) development version from the `main` branch. If you want a stable release, use the PyPI installation method instead.

### MODFLOW 6

Field-scale simulations require the MODFLOW 6 executable and shared library. After installing mibiremo, install them into the active Python environment with:

```console
get-modflow :python --subset mf6,libmf6
```

## Documentation

- [Introduction](introduction.md): overview of the package and of the MIBIREM project.
- [Batch model](batch_model.md), [Column model](column_model.md), and [Field model](field_model.md): model description, workflow, and conventions.
- [Examples](examples.md): description of the examples, which are also available as tutorial notebooks.
- [API reference](reference.md): classes and functions of the package.

## Contributing

If you want to contribute to the development of mibiremo, have a look at the [development](development.md) page.

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
