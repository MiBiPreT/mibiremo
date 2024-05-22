# MiBiReMo

MiBiReMo (Microbiome Bioremediation Reaction Module) is a Python interface to the PhreeqcRM library. The package is designed to be coupled with transport models to simulate reactive transport in porous media, with applications in environmental and geochemical engineering. Developed as part of the [MIBIREM](https://www.mibirem.eu/) toolbox for Bioremediation.


## Installation
To locally install the package, clone the repository and (optionally) create a virtual environment, activate it, and install the required packages using pip:
```sh
cd path/to/project_folder
python -m venv mibiremo
source mibiremo/bin/activate
pip install -r requirements.txt
```
(to activate the environment on Windows, use the command `mibiremo\Scripts\activate`).


To install the package itself, run the following command in the root directory of the package:

```sh
cd path/to/mibiremo_source
pip install .
```

## Examples

Currently, two examples are available in the `examples` directory. 
The first example `ex1-titration.py` demonstrates the use of the package to simulate a simple titration in a batch, where a solution is equilibrated with a mineral phase (calcite) and the pH is adjusted by adding HCl.
The second example `ex2-BTEX_dissolution.py` demonstrates the use of the package to simulate the kinetic dissolution of benzene and ethylbenzene in a batch reactor.

To run the examples, navigate to the `examples` directory and run the desired example script:
```sh
cd path/to/project_folder
source mibiremo/bin/activate
cd path/to/mibiremo_source/examples
python ex1-titration.py
```
