## MiBiReMo - Example - Calculate calcite titration curve
# CaCO3(s) + 2HCl(aq) → CaCl2(aq) + CO2(g) + H2O(l)
# Chemical reactions assumed at equilibrium
#
# Last revision: 21/05/2024

# %%
import mibiremo
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
########### SETTINGS ##############
db = '../mibiremo/database/phreeqc.dat'             # .dat database path
ncells = 1000                                       # number of model cells
nthreads = 4                                        # multithread calculation. -1 for all CPUs
pqifile = 'pqi/Calcite_titration.pqi'               # Name of the phreeqc input file
HCl_range = [0.0, 4.0]                              # mol/L

unit_sol = 2                                        # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1                                           # 0: mol/L cell; 1: mol/L water; 2: mol/L rock

n = 1.0                                             # Porosity
S = 1.0                                             # Saturation
##################################

# %%
# Initialize Phreeqc
phr = mibiremo.PhreeqcRM() 
phr.create(nxyz=ncells, n_threads=nthreads)

# Load database
status = phr.RM_LoadDatabase(db)

# %%
### Set properties/parameters
phr.RM_SetComponentH2O(0)           # Don't include H2O in the component list
phr.RM_SetRebalanceFraction(0.5)    # Rebalance the load of each thread

### Set units
phr.RM_SetUnitsSolution(unit_sol)   
phr.RM_SetUnitsPPassemblage(units)
phr.RM_SetUnitsExchange(units)
phr.RM_SetUnitsSurface(units)
phr.RM_SetUnitsGasPhase(units)
phr.RM_SetUnitsSSassemblage(units)
phr.RM_SetUnitsKinetics(units)

### Porosity and saturation
phr.RM_SetPorosity(n*np.ones(ncells))
phr.RM_SetSaturation(S*np.ones(ncells))

### Create error log files
phr.RM_SetFilePrefix('titr')
phr.RM_OpenFiles()

### Multicomponent diffusion transport calculation settings
phr.RM_SetSpeciesSaveOn(1)

# %%
# RUN 
status = phr.RM_RunFile(1,1,1,pqifile)
print('Run: ' + mibiremo.IRM_RESULT(status)[1])
# Print some of the reaction module information and log to file
th = phr.RM_GetThreadCount()
str1 = 'Number of threads:  {}\n'.format(th)
phr.RM_OutputMessage(str1)
print(str1)

# %%
# Transfer solutions and reactants from the InitialPhreeqc instance to the reaction-module workers
# Column index for the initial conditions ic1.shape ->  (ncells, 7) 
#(1) SOLUTIONS, (2) EQUILIBRIUM_PHASES, (3) EXCHANGE, (4) SURFACE, (5) GAS_PHASE, (6) SOLID_SOLUTIONS, and (7) KINETICS

ncl = phr.RM_GetGridCellCount()

ic1 = -1*np.ones(ncells*7,dtype=np.int32)
for i in np.arange(ncells):
    ic1[i] = 1             # Solution 1
    ic1[i+ncells] = 1      # Equilibrium phases 1
    ic1[i+2*ncells] = -1   # Exchange none
    ic1[i+3*ncells] = -1   # Surface none
    ic1[i+4*ncells] = -1   # Gas phase none
    ic1[i+5*ncells] = -1   # Solid solutions none
    ic1[i+6*ncells] = -1   # Kinetics none

ic2 = -1*np.ones(ncells*7,dtype=np.int32)
f1 = np.ones(ncells*7,dtype=np.float64)

# %%
status = phr.RM_InitialPhreeqc2Module(ic1,ic2,f1)

# Get number and name of components and species
ncomps = phr.RM_FindComponents()
nspecies = phr.RM_GetSpeciesCount()

# Get the components
components = np.zeros(ncomps,dtype='U20')
for i in range(ncomps):
    status = phr.RM_GetComponent(i, components,20)

# Get the species
species = np.zeros(nspecies,dtype='U20')
for i in range(nspecies):
    status = phr.RM_GetSpeciesName(i, species,20)


# %%
# Retrieve initial solution
# Make an initial step (zero lenght) to be sure that 
# everything is at equilibrium
phr.RM_SetTime(0.0)
phr.RM_SetTimeStep(0.0)
status = phr.RM_RunCells()

# %%
# Initialize component concentration Cc vector
Cc = np.zeros(ncells*ncomps,dtype=np.float64)
status = phr.RM_GetConcentrations(Cc)

# Initialize species concentration Cs matrix
Cs = np.zeros(ncells*nspecies,dtype=np.float64)
status = phr.RM_GetSpeciesConcentrations(Cs)

# Get species diffusion coefficients
# D = np.zeros(nspecies,dtype=np.float64)
# status = phr.RM_GetSpeciesD25(D)

# %%
# GET SELECTED OUTPUT
nselout = phr.RM_GetSelectedOutputCount()
ncolsel = phr.RM_GetSelectedOutputColumnCount()
selout = np.zeros(ncells*ncolsel, dtype=np.float64) 
status = phr.RM_GetSelectedOutput(selout)

# Get headers of the selected output
selout_h = np.zeros(ncolsel,dtype='U100')
for i in range(ncolsel):
    status = phr.RM_GetSelectedOutputHeading(i,selout_h,100)
    

# %%
# SET SPECIES CONCENTRATION
HCl = np.linspace(HCl_range[0],HCl_range[1],ncells) #mol/L

# Find column of Cl- and H+ species
indx_Cl = np.where(species == 'Cl-')[0][0]
indx_H = np.where(species == 'H+')[0][0]

# Species matrix
Cs_r = Cs.reshape(nspecies,ncells).T

# %%
# ADD HCl AND RUN
Cs_r[:,indx_Cl] = Cs_r[:,indx_Cl] + HCl     # Cl-
Cs_r[:,indx_H] = Cs_r[:,indx_H] + HCl       # H+
Cs0 = Cs_r.T.reshape(ncells*nspecies)
Cs1 = np.copy(Cs0)
status = phr.RM_SpeciesConcentrations2Module(Cs1)
phr.RM_SetTime(1.0)
phr.RM_SetTimeStep(1.0)

t = time.time()
phr.RM_RunCells()
elapsed = time.time() - t
print('Elapsed time {:.2f} (s)'.format(elapsed))

# %%
# GET RESULTS - SELECTED OUTPUT
s_pd = phr.pdSelectedOutput()  # Returns a pandas Data Frame

# %%
# PLOTS
plt.figure(figsize = (10,6))
plt.plot(HCl, s_pd.pH)
plt.xlabel('molH$^+$ added')
plt.ylabel('pH')
plt.show()

