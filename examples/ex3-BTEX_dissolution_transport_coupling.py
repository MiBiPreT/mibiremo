## MiBiReMo - Example - BTEX dissolution and 1D transport coupling
#
# Coupling - Sequential Non-Iterative Approach (SNIA)
# Step 1) Advective and dispersive transport: custom 1D solver based on Semi-Lagrangian method
# Step 2) Reaction: PhreeqcRM
#
# Simulations are compared to PHREEQC results with TRANSPORT keyword
#
# Two types of simulations are performed and compared:
# 1) Equilibrium dissolution - NAPL phase are redefined as equilibrium phases
# 2) Kinetic dissolution - NAPL kinetic dissolution is considered
#
# Author: M.M.
# Last revision: 03/07/2024


import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mibiremo

#############  SETTINGS  ###############
db = "../mibiremo/database/mibirem.dat"                             # .dat database path
pqieq =   "pqi/ex3_BTEX_dissolution_and_transport_coupling_eq.pqi"  # Name of the phreeqc input file - equilibrium
pqikin =  "pqi/ex3_BTEX_dissolution_and_transport_coupling_kin.pqi" # Name of the phreeqc input file - kinetics
selfile = "pqi/ex3_BTEX_dissolution_and_transport.sel"              # Name of the selected output file (PHREEQC results)

###########  MODEL PARAMETERS  #########
ncells = 1000           # number of model cells
dt = 0.1                # Coupling time step (days)
L = 100.0               # Length of the domain (m)
nthreads = 6            # Multithread calculation. -1 for all CPUs
alpha = 0.05            # Dispersivity m2
v = 1.0                 # Groundwater velocity m/d
De = 0.0                # Molecular diffusion m2/s
simDuration = 100.0     # Simulation duration (days)
unit_sol = 2            # 1: mg/L; 2: mol/L; 3: kg/kgs
units = 1               # 0: mol/L cell; 1: mol/L water; 2: mol/L rock
n = 1.0                 # Porosity
S = 1.0                 # Saturation

probe = 49.75           # Probe location (m) for results extraction (for comparison with PHREEQC results)

D = alpha*v     # Dispersion coefficient m2/d


def phr_initialize():
    """Utility function for common PhreeqcRM initialization."""
    phr = mibiremo.PhreeqcRM()
    phr.create(nxyz=ncells, n_threads=nthreads)
    phr.RM_LoadDatabase(db)


    # Set properties/parameters
    phr.RM_SetComponentH2O(0)           # Don't include H2O in the component list
    phr.RM_SetRebalanceFraction(0.5)    # Rebalance the load of each thread

    # Set units
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
    phr.RM_SetRepresentativeVolume(1.0*np.ones(ncells))


    # Run initial calculations
    phr.RM_SetFilePrefix("btex")
    phr.RM_OpenFiles()

    return phr


#########################################
# 1) EQUILIBRIUM DISSOLUTION
#########################################

phr = phr_initialize()
status = phr.RM_RunFile(1,1,1,pqieq)

# Transfer solutions and reactants from the InitialPhreeqc instance to the reaction-module workers
# Column index for the initial conditions ic1.shape ->  (ncells, 7)
#(1) SOLUTIONS, (2) EQUILIBRIUM_PHASES, (3) EXCHANGE, (4) SURFACE, (5) GAS_PHASE, (6) SOLID_SOLUTIONS, and (7) KINETICS
ncl = phr.RM_GetGridCellCount()

ic1 = -1*np.ones(ncells*7,dtype=np.int32)

# Assign definitions to cells
for i in np.arange(ncells):
    ic1[i] = 2             # Solution
    ic1[i+ncells] = -1     # Equilibrium phases
    ic1[i+2*ncells] = -1   # Exchange
    ic1[i+3*ncells] = -1   # Surface
    ic1[i+4*ncells] = -1   # Gas phase
    ic1[i+5*ncells] = -1   # Solid solutions
    ic1[i+6*ncells] = -1   # Kinetics

# Contaminated spot 0.5 m
spot = int(0.5*ncells/L)
ic1[0:spot] = 1               # Solution
ic1[ncells:ncells+spot] = 1   # Equilibrium phases

ic2 = -1*np.ones(ncells*7,dtype=np.int32)
f1 = np.ones(ncells*7,dtype=np.float64)

status = phr.RM_InitialPhreeqc2Module(ic1,ic2,f1)

# Get number and name of components and species
ncomps = phr.RM_FindComponents()
nspecies = phr.RM_GetSpeciesCount()

# Get the components
components = np.zeros(ncomps,dtype="U20")
for i in range(ncomps):
    status = phr.RM_GetComponent(i, components,20)

# Retrieve initial solution
phr.RM_SetTime(0.0)
phr.RM_SetTimeStep(0.1) # Small initial step to get the initial solution
status = phr.RM_RunCells()

# Initialize component concentration Cc vector
Cc = np.zeros(ncells*ncomps,dtype=np.float64)
status = phr.RM_GetConcentrations(Cc)

# Prepare simulation
nsteps = int(simDuration/dt)
tvect = np.zeros(nsteps)
Cheadings = ["Benz", "Ethyl"] # Transported species
Csim = np.zeros((nsteps, len(Cheadings)))

# Map Cheadings to components
Cmap = np.zeros(len(Cheadings),dtype=np.int32)
for i in range(len(Cheadings)):
    Cmap[i] = np.where(components == Cheadings[i])[0][0]


# Store initial concentrations
x = np.linspace(0,L,ncells)     # Space vector
loc = np.argmin(np.abs(x-probe)) # Probe location
Cmat = Cc.reshape((ncomps,ncells)).T
Csim[0,:] = Cmat[loc,Cmap]

C_L = 0.0   # Concentrations at the left boundary


###### MAIN LOOP ######
tic = time.time()
for i in range(1,nsteps):
    # Update time
    tvect[i] = i*dt

    # 1) Reactive step
    phr.RM_SetTime(tvect[i]*3600*24)  # s
    phr.RM_SetTimeStep(dt*3600*24)    # s
    status = phr.RM_RunCells()
    status = phr.RM_GetConcentrations(Cc)
    Cmat = Cc.reshape((ncomps,ncells)).T

    # 2) Transport step
    for j in range(len(Cheadings)):
        s =  mibiremo.SemiLagSolver(x, Cmat[:,Cmap[j]], v, D, dt)
        Ctmp = s.transport(C_L)
        # Store results
        Csim[i,j] = Ctmp[loc]
        # Update concentration
        Cmat[:,Cmap[j]] = Ctmp

    # Set back the concentrations to PhreeqcRM
    Cc = Cmat.T.flatten()
    status = phr.RM_SetConcentrations(Cc)


elapsed = time.time() - tic



#########################################
# 2) KINETIC DISSOLUTION
#########################################
# Run initial calculations
phr = phr_initialize()
status = phr.RM_RunFile(1,1,1,pqikin)

# Reassign initial conditions and run the initial module
for i in np.arange(ncells):
    ic1[i] = 2             # Solution
    ic1[i+ncells] = -1     # Equilibrium phases
    ic1[i+2*ncells] = -1   # Exchange
    ic1[i+3*ncells] = -1   # Surface
    ic1[i+4*ncells] = -1   # Gas phase
    ic1[i+5*ncells] = -1   # Solid solutions
    ic1[i+6*ncells] = -1   # Kinetics
ic1[0:spot] = 1                     # Contaminant spot
ic1[6*ncells:6*ncells+spot] = 1     # Kinetics
status = phr.RM_InitialPhreeqc2Module(ic1,ic2,f1)
ncomps = phr.RM_FindComponents()
components = np.zeros(ncomps,dtype="U20")
for i in range(ncomps):
    status = phr.RM_GetComponent(i, components,20)
phr.RM_SetTime(0.0)
phr.RM_SetTimeStep(0.1)
status = phr.RM_RunCells()
Cc = np.zeros(ncells*ncomps,dtype=np.float64)
status = phr.RM_GetConcentrations(Cc)
Cmap = np.zeros(len(Cheadings),dtype=np.int32)
for i in range(len(Cheadings)):
    Cmap[i] = np.where(components == Cheadings[i])[0][0]
Csim_kin = np.zeros((nsteps, len(Cheadings)))
Cmat = Cc.reshape((ncomps,ncells)).T
Csim_kin[0,:] = Cmat[loc,Cmap]

tic = time.time()
for i in range(1,nsteps):
    # Update time
    tvect[i] = i*dt

    # 1) Reactive step
    phr.RM_SetTime(0.0)             # s
    phr.RM_SetTimeStep(dt*3600*24)  # s
    status = phr.RM_RunCells()
    status = phr.RM_GetConcentrations(Cc)
    Cmat = Cc.reshape((ncomps,ncells)).T

    # 2) Transport step
    for j in range(len(Cheadings)):
        s =  mibiremo.SemiLagSolver(x, Cmat[:,Cmap[j]], v, D, dt)
        Ctmp = s.transport(C_L)
        # Store results
        Csim_kin[i,j] = Ctmp[loc]
        # Update concentration
        Cmat[:,Cmap[j]] = Ctmp

    # Set back the concentrations to PhreeqcRM
    Cc = Cmat.T.flatten()
    status = phr.RM_SetConcentrations(Cc)


elapsed = time.time() - tic


#########################################
# PLOT RESULTS
#########################################

# Import .sel file
df = pd.read_csv(selfile, sep="\t")
df = df.iloc[:, 0:-1]
df.columns = df.columns.str.replace(" ", "")
transp = df[df["state"].str.contains("transp")]
fig = plt.figure(figsize=(10, 4))
# PHREEQC results
plt.plot(transp["time"]/3600/24, transp["Benz"]*1e3*78.114, label="Benzene - PHREEQC - equilibrium", 
         linestyle="None", marker="o", markersize=8, markerfacecolor="none")
plt.plot(transp["time"]/3600/24, transp["Ethyl"]*1e3*106.17, label="Ethylbenzene - PHREEQC - equil.",  
         linestyle="None", marker="^", markersize=8, markerfacecolor="none")

# Equilibrium dissolution
plt.plot(tvect[:], Csim[:,0]*78.114*1000, label="Benzene - MIBIREMO - equilibrium")
plt.plot(tvect[:], Csim[:,1]*106.17*1000, label="Ethylbenzene - MIBIREMO - equil.")

# Kinetic dissolution
plt.plot(tvect, Csim_kin[:,0]*78.114*1000, label="Benzene - MIBIREMO - kinetics", linestyle="--")
plt.plot(tvect, Csim_kin[:,1]*106.17*1000, label="Ethylbenzene - MIBIREMO - kinetics", linestyle="--")

plt.xlabel("Time (days)")
plt.ylabel("Aqueous concentration (mg/L)")
plt.legend()
plt.title("BTEX dissolution and transport at 50 m distance from the source - equilibrium vs kinetics", fontsize=10)
plt.show()

