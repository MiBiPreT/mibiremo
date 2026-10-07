# Introduction

## General

MiBiReMo (MiBiPreT Remediation Module) is an open-source Python package for the simulation and design of groundwater bioremediation systems. It allows to set up field-scale models, focusing on biological flushing technique, to simulate groundwater flow, solute transport, and reactive transport. Flow and transport are simulated using MODFLOW 6, while geochemical and biodegradation reactions are computed using PHREEQC, and coupled to MODFLOW through [mf6rtm](https://github.com/p-ortega/mf6rtm)). The package also includes a one-dimensional advection–dispersion solver for the simulation of laboratory-scale column experiments. MiBiReMo is part of the MiBiPreT (Micro-Bioremediation Prediction Tool) tools, developed within the [MIBIREM](https://www.mibirem.eu/) toolbox for bioremediation.

## MIBIREM

[MIBIREM - Innovative technological toolbox for bioremediation](https://www.mibirem.eu/) is a EU funded consortium project by 12 international partners all over Europe working together to develop an *Innovative technological toolbox for bioremediation*. The project will develop molecular methods for the monitoring, isolation, cultivation and subsequent deposition of whole microbiomes. The toolbox will also include the methodology for the improvement of specific microbiome functions, including evolution and enrichment. The performance of selected microbiomes will be tested under real field conditions. The `mibiremo` package is part of this toolbox.
