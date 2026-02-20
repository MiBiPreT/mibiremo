# MiBiReMo - Examples

Currently, three examples are available as both Python scripts and Jupyter notebooks.
To run the examples as scripts, navigate to the `examples` directory and run the desired example:
```sh
cd examples
python ex1-titration.py
```

Interactive Jupyter notebooks are also available in the `examples` folder, and can be viewed in the [Notebooks](notebooks/ex1-titration.ipynb) section.

## Example 1 - Titration
The first example `ex1-titration.py` demonstrates the use of the package to simulate a simple titration in a batch, where a solution is equilibrated with a mineral phase (calcite) and the pH is adjusted by adding HCl.

## Example 2 - Benzene and Ethylbenzene kinetic dissolution
The second example `ex2-BTEX_dissolution.py` demonstrates the use of the package to simulate the kinetic dissolution of benzene and ethylbenzene in a batch reactor.

## Example 3 - Reactive transport - BTEX dissolution and transport
The third example `ex3-BTEX_dissolution_transport_coupling.py` demonstrates the use of the package to simulate the reactive transport of benzene and ethylbenzene in a 1D domain.
The problem is described in the following scheme:

![Model diagram](img/ex3-BTEX_dissolution_transport.png)

The model consists of a 1D domain with a length of 50 m. The domain is initially filled with clean groundwater with a spot of benzene and ethylbenzene pure phases present at the left side of the domain extending for 0.5 m. We assumed that the contaminant pure phase in the source zone is immobile (it only dissolves).
The groundwater flows from left to right with a velocity of 1 m/d, and dissolved benzene and ethylbenzene are transported towards the right end of the domain. The dissolution process is modelled both kinetically and equilibrium-based. 
The model is run for 100 days, and the concentration of benzene and ethylbenzene is monitored at the right side of the domain.

Three simulation runs are performed:
1. Equilibrium dissolution of benzene and ethylbenzene simulated with PHREEQC (standalone).
2. Equilibrium dissolution of benzene and ethylbenzene simulated by MiBiReMo (1D transport solver coupled with PhreeqcRM).
3. Kinetic dissolution of benzene and ethylbenzene simulated by MiBiReMo.

The transport equition (advection and dispersion) in MiBiReMo is solved using a Semi-Lagrangian scheme with operator splitting, where the advection is solved using the method of characteristics with cubic spline interpolation, and the dispersion is solved using the Saul'yev finite differences technique. 

Simulation results are shown in the following figure:

![Model results](img/ex3_BTEX_dissolution_and_transport_results.png)

The results obtaines show a pattern similar to the experimental results obtained by Geller and Hunt (1993) [[1]](#1).
In their experiment they injected an equimolar mixture of benzene and toluene in the center of a column which was subsequently eluted with water. The results show that benzene is eluted first, followed by toluene because of the different solubilities.
The following figure shows the experimental results obtained by Geller and Hunt (1993):

![Geller and Hunt (1993) results](img/Geller_Hunt_1993.png)



## References

<a id="1">[1]</a> Geller, J. T., and J. R. Hunt (1993), Mass transfer from nonaqueous phase organic liquids in water-saturated porous media, Water Resour. Res., 29(4), 833–845, doi:[10.1029/92WR02581](https://doi.org/10.1029/92WR02581). 
