## MiBiReMo - Example - BTEX dissolution
#
#
# Last revision: 02/07/2024


import time
import matplotlib.pyplot as plt
import numpy as np
import mibiremo

########### SETTINGS ##############
db = "../mibiremo/database/mibirem.dat"  # .dat database path
ncells = 1  # number of model cells
nthreads = 4  # multithread calculation. -1 for all CPUs
pqifile = "pqi/ex2_BTEX_dissolution.pqi"  # Name of the phreeqc input file

unit_sol = 2  # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1  # 0: mol/L cell; 1: mol/L water; 2: mol/L rock

n = 1.0  # Porosity
S = 1.0  # Saturation

simDuration = 7.0  # Simulation duration (days)
nsteps = 100  # Number of steps
##################################


# Initialize Phreeqc
phr = mibiremo.PhreeqcRM()
phr.create(nxyz=ncells, n_threads=nthreads)

# Load database
status = phr.RM_LoadDatabase(db)

### Set properties/parameters
phr.RM_SetComponentH2O(0)  # Don't include H2O in the component list
phr.RM_SetRebalanceFraction(0.5)  # Rebalance the load of each thread

### Set units
phr.RM_SetUnitsSolution(unit_sol)
phr.RM_SetUnitsPPassemblage(units)
phr.RM_SetUnitsExchange(units)
phr.RM_SetUnitsSurface(units)
phr.RM_SetUnitsGasPhase(units)
phr.RM_SetUnitsSSassemblage(units)
phr.RM_SetUnitsKinetics(units)

### Porosity and saturation
phr.RM_SetPorosity(n * np.ones(ncells))
phr.RM_SetSaturation(S * np.ones(ncells))

### Create error log files
phr.RM_SetFilePrefix("btex")
phr.RM_OpenFiles()

### Multicomponent diffusion transport calculation settings
phr.RM_SetSpeciesSaveOn(1)


# RUN initial solution
status = phr.RM_RunFile(1, 1, 1, pqifile)
# Print some of the reaction module information and log to file
th = phr.RM_GetThreadCount()
str1 = f"Number of threads:  {th}\n"
phr.RM_OutputMessage(str1)


# Transfer solutions and reactants from the InitialPhreeqc instance to the reaction-module workers
# Column index for the initial conditions ic1.shape ->  (ncells, 7)
# (1) SOLUTIONS, (2) EQUILIBRIUM_PHASES, (3) EXCHANGE, (4) SURFACE, (5) GAS_PHASE, (6) SOLID_SOLUTIONS, and (7) KINETICS
ncl = phr.RM_GetGridCellCount()

ic1 = -1 * np.ones(ncells * 7, dtype=np.int32)
for i in np.arange(ncells):
    ic1[i] = 1  # Solution 1
    ic1[i + ncells] = -1  # Equilibrium phases 1
    ic1[i + 2 * ncells] = -1  # Exchange none
    ic1[i + 3 * ncells] = -1  # Surface none
    ic1[i + 4 * ncells] = -1  # Gas phase none
    ic1[i + 5 * ncells] = -1  # Solid solutions none
    ic1[i + 6 * ncells] = 1  # Kinetics 1

ic2 = -1 * np.ones(ncells * 7, dtype=np.int32)
f1 = np.ones(ncells * 7, dtype=np.float64)


status = phr.RM_InitialPhreeqc2Module(ic1, ic2, f1)

# Get number and name of components and species
ncomps = phr.RM_FindComponents()
nspecies = phr.RM_GetSpeciesCount()

# Get the components
components = np.zeros(ncomps, dtype="U20")
for i in range(ncomps):
    status = phr.RM_GetComponent(i, components, 20)

# Get the species
species = np.zeros(nspecies, dtype="U20")
for i in range(nspecies):
    status = phr.RM_GetSpeciesName(i, species, 20)


# Retrieve initial solution
# Make an initial step to get the initial solution
phr.RM_SetTime(0.0)
phr.RM_SetTimeStep(0.1)  # 0.1 second
status = phr.RM_RunCells()

# Initialize component concentration Cc vector
Cc = np.zeros(ncells * ncomps, dtype=np.float64)
status = phr.RM_GetConcentrations(Cc)

# Initialize species concentration Cs matrix
Cs = np.zeros(ncells * nspecies, dtype=np.float64)
status = phr.RM_GetSpeciesConcentrations(Cs)


# GET SELECTED OUTPUT
nselout = phr.RM_GetSelectedOutputCount()
ncolsel = phr.RM_GetSelectedOutputColumnCount()
selout = np.zeros(ncells * ncolsel, dtype=np.float64)
status = phr.RM_GetSelectedOutput(selout)

# Get headers of the selected output
selout_h = np.zeros(ncolsel, dtype="U100")
for i in range(ncolsel):
    status = phr.RM_GetSelectedOutputHeading(i, selout_h, 100)


###### MAIN LOOP ######
# Time step
dt = simDuration / nsteps * 24 * 3600.0  # s

tvect = np.zeros(nsteps)
Cheadings = ["Benz", "Benznapl", "Ethyl", "Ethylnapl"]
Csim = np.zeros((nsteps, len(Cheadings)))

# Map Cheadings to components
Cmap = np.zeros(len(Cheadings), dtype=np.int32)
for i in range(len(Cheadings)):
    Cmap[i] = np.where(components == Cheadings[i])[0][0]

# Store initial concentrations
Csim[0, :] = Cc[Cmap]

tic = time.time()
for i in range(1, nsteps):
    # Update time
    tvect[i] = i * dt

    # Run cells
    phr.RM_SetTime(tvect[i])  # s
    phr.RM_SetTimeStep(dt)  # s
    status = phr.RM_RunCells()

    # Get concentrations
    status = phr.RM_GetConcentrations(Cc)
    Csim[i, :] = Cc[Cmap]

elapsed = time.time() - tic


###### PLOTS ######
plt.figure(figsize=(10, 6))
plt.plot(tvect / 86400.0, Csim[:, [1, 3]])
plt.xlabel("Elapsed time (days)")
plt.ylabel("NAPL Concentration (mol/L)")
plt.legend([Cheadings[1], Cheadings[3]])
plt.title("BTEX dissolution")
plt.twinx()  # Plot on a second y-axis
CmgL = Csim[:, [0, 2]].copy()
CmgL[:, 0] = CmgL[:, 0] * 78.11 * 1000  # Convert mol/L to mg/L (Benzene)
CmgL[:, 1] = CmgL[:, 1] * 106.17 * 1000  # Convert mol/L to mg/L (Ethylbenzene)
plt.plot(tvect / 86400.0, CmgL, "--")
plt.ylabel("Dissolved concentration (mg/L)")
plt.legend([Cheadings[0], Cheadings[2]])
plt.show()
