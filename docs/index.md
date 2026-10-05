# Documentation for `mibiremo` python package

MiBiPreT Remediation Module, a Python package for designing and testing bioremediation installations. Part of the MiBiPreT (Micro-Bioremediation Prediction Tool) tools, developed within the MIBIREM toolbox for bioremediation.


## Installation

To install mibiremo from GitHub repository, do:

```console
git clone git@github.com:MiBiPreT/mibiremo.git
cd mibiremo
python -m pip install .
```

Field-scale simulations require the MODFLOW 6 executable and shared library. Install them into the active Python environment with:

```console
get-modflow :python --subset mf6,libmf6
```
