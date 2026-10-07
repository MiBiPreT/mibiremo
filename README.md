
[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/MiBiPreT/mibiremo)
[![github license badge](https://img.shields.io/github/license/MiBiPreT/mibiremo)](https://github.com/MiBiPreT/mibiremo) 
[![RSD](https://img.shields.io/badge/rsd-mibiremo-00a3e3.svg)](https://www.research-software.nl/software/mibiremo) 
[![workflow pypi badge](https://img.shields.io/pypi/v/mibiremo.svg?colorB=blue)](https://pypi.python.org/project/mibiremo/) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15180602.svg)](https://doi.org/10.5281/zenodo.15180602)
[![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/10401/badge)](https://bestpractices.coreinfrastructure.org/projects/10401) 
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) 
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=MiBiPreT_mibiremo&metric=alert_status)](https://sonarcloud.io/dashboard?id=MiBiPreT_mibiremo) 
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=MiBiPreT_mibiremo&metric=coverage)](https://sonarcloud.io/dashboard?id=MiBiPreT_mibiremo)
<!-- [![Documentation Status](https://readthedocs.org/projects/mibiremobadge/?version=latest)](https://mibiremo.readthedocs.io/en/latest/?badge=latest) -->
[![build](https://github.com/MiBiPreT/mibiremo/actions/workflows/build.yml/badge.svg)](https://github.com/MiBiPreT/mibiremo/actions/workflows/build.yml)
[![cffconvert](https://github.com/MiBiPreT/mibiremo/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/MiBiPreT/mibiremo/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/MiBiPreT/mibiremo/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/MiBiPreT/mibiremo/actions/workflows/sonarcloud.yml)
[![link-check](https://github.com/MiBiPreT/mibiremo/actions/workflows/link-check.yml/badge.svg)](https://github.com/MiBiPreT/mibiremo/actions/workflows/link-check.yml)


# `mibiremo`

MiBiReMo (MiBiPreT Remediation Module) is a Python package for designing and testing bioremediation installations. It provides field-scale models of injection and extraction wells with MODFLOW 6 (groundwater flow, tracer transport, and reactive transport coupled to PHREEQC through [mf6rtm](https://github.com/p-ortega/mf6rtm)), a Python interface to the PhreeqcRM geochemical reaction module, and a 1D advection–dispersion solver for reactive transport in porous media. MiBiReMo is part of the MiBiPreT (Micro-Bioremediation Prediction Tool) tools, developed within the [MIBIREM](https://www.mibirem.eu/) toolbox for bioremediation.

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

## Examples
Examples are available in the [`examples`](examples/) directory. 

## Documentation

The project's full documentation is available [here](https://mibipret.github.io/mibiremo/).

## Contributing

If you want to contribute to the development of mibiremo,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
