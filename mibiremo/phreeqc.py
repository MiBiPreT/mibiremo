"""Python-PhreeqcRM interface.
==========================

PhreeqcRM documentation:
https://usgs-coupled.github.io/phreeqcrm/namespacephreeqcrm.html

Last revision: 03/07/2024
"""


import ctypes
import os
import numpy as np
import pandas as pd
from .irmresult import IRM_RESULT


class PhreeqcRM:

    def __init__(self):
        """Initialize PhreeqcRM instance."""
        self.dllpath = None
        self.nxyz = 1
        self.n_threads = 1
        self.libc = None
        self.id = None

    def create(self, dllpath = "", nxyz = 1, n_threads = 1) -> None:
        """Create a PhreeqcRM instance
        Args:
            dllpath: Path to the PhreeqcRM library. If left empty, the provided libraries are used.
            nxyz: Number of grid cells. Default: 1.
            n_threads: Number of threads. Default: 1.
        """
        if dllpath == "":
            # If no path is provided, use the default path, based on operating system
            if os.name == "nt":
                dllpath = os.path.join(os.path.dirname(__file__), "lib", "PhreeqcRM.dll")
            elif os.name == "posix":
                dllpath = os.path.join(os.path.dirname(__file__), "lib", "PhreeqcRM.so")
            else:
                msg = "Operating system not supported"
                raise Exception(msg)
        self.dllpath = dllpath

        if n_threads == -1:
            n_threads = os.cpu_count()

        self.n_threads = n_threads
        self.nxyz = nxyz
        self.libc = ctypes.CDLL(dllpath)
        self.id = self.libc.RM_Create(nxyz,n_threads)

    def pdSelectedOutput(self):
        """Returns a Pandas data frame for Selected Output."""
        # Get selected ouput headings
        ncolsel = self.RM_GetSelectedOutputColumnCount()
        selout_h = np.zeros(ncolsel,dtype="U100")
        for i in range(ncolsel):
            self.RM_GetSelectedOutputHeading(i,selout_h,100)
        so = np.zeros(ncolsel*self.nxyz).reshape(self.nxyz, ncolsel)
        self.RM_GetSelectedOutput(so)
        return pd.DataFrame(so.reshape(ncolsel,self.nxyz).T, columns=selout_h)


    ### PhreeqcRM functions

    def RM_Abort(self,result,err_str):
        return IRM_RESULT(self.libc.RM_Abort(self.id,result,err_str))

    def RM_CloseFiles(self):
        return self.libc.RM_CloseFiles(self)

    def RM_Concentrations2Utility(self,c,n,tc,p_atm):
        return self.libc.RM_Concentrations2Utility(self.id,c,n,tc,p_atm)

    def RM_CreateMapping(self,grid2chem):
        return self.libc.RM_CreateMapping(self.id,grid2chem.ctypes)

    def RM_DecodeError(self,e):
        return self.libc.RM_DecodeError(self.id,e)

    def RM_Destroy(self):
        return self.libc.RM_Destroy(self.id)

    def RM_DumpModule(self,dump_on,append):
        return self.libc.RM_DumpModule(self.id,dump_on,append)

    def RM_ErrorMessage(self,errstr):
        return self.libc.RM_ErrorMessage(self.id,errstr)

    def RM_FindComponents(self):
        return self.libc.RM_FindComponents(self.id)

    def RM_GetBackwardMapping(self,n,list,size):
        return self.libc.RM_GetBackwardMapping(self.id,n,list,size)

    def RM_GetChemistryCellCount(self):
        return self.libc.RM_GetChemistryCellCount(self.id)

    def RM_GetComponent(self,num,chem_name,length):
        String = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetComponent(self.id,num,String,length)
        chem_name[num] = String.value.decode()
        return status

    def RM_GetConcentrations(self,c):
        return self.libc.RM_GetConcentrations(self.id,c.ctypes)

    def RM_GetDensity(self,density):
        return self.libc.RM_GetDensity(self.id,density.ctypes)

    def RM_GetEndCell(self,ec):
        return self.libc.RM_GetEndCell(self.id,ec)

    def RM_GetEquilibriumPhaseCount(self):
        return self.libc.RM_GetEquilibriumPhaseCount(self.id)
    def RM_GetEquilibriumPhaseName(self,num,name,l1):
        return self.libc.RM_GetEquilibriumPhaseName(self.id,num,name,l1)
    def RM_GetErrorString(self,errstr,length):
        return self.libc.RM_GetErrorString(self.id,errstr,length)
    def RM_GetErrorStringLength(self):
        return self.libc.RM_GetErrorStringLength(self.id)
    def RM_GetExchangeName(self,num,name,l1):
        return self.libc.RM_GetExchangeName(self.id,num,name,l1)

    def RM_GetExchangeSpeciesCount(self):
        return self.libc.RM_GetExchangeSpeciesCount(self.id)

    def RM_GetExchangeSpeciesName(self,num,name,l1):
        return self.libc.RM_GetExchangeSpeciesName(self.id,num,name,l1)

    def RM_GetFilePrefix(self,prefix,length):
        return self.libc.RM_GetFilePrefix(self.id,prefix.encode(),length)
    def RM_GetGasComponentsCount(self):
        return self.libc.RM_GetGasComponentsCount(self.id)

    def RM_GetGasComponentsName(self,nun,name,l1):
        return self.libc.RM_GetGasComponentsName(self.id,nun,name,l1)

    def RM_GetGfw(self,gfw):
        return self.libc.RM_GetGfw(self.id,gfw.ctypes)

    def RM_GetGridCellCount(self):
        return self.libc.RM_GetGridCellCount(self.id)

    def RM_GetIPhreeqcId(self,i):
        return self.libc.RM_GetIPhreeqcId(self.id,i)

    def RM_GetKineticReactionsCount(self):
        return self.libc.RM_GetKineticReactionsCount(self.id)

    def RM_GetKineticReactionsName(self,num,name,l1):
        return self.libc.RM_GetKineticReactionsName(self.id,num,name,l1)

    def RM_GetMpiMyself(self):
        return self.libc.RM_GetMpiMyself(self.id)

    def RM_GetMpiTasks(self):
        return self.libc.RM_GetMpiTasks(self.id)

    def RM_GetNthSelectedOutputUserNumber(self,n):
        return self.libc.RM_GetNthSelectedOutputUserNumber(self.id,n)

    def RM_GetSaturation(self,sat_calc):
        return self.libc.RM_GetSaturation(self.id,sat_calc)

    def RM_GetSelectedOutput(self,so):
        return self.libc.RM_GetSelectedOutput(self.id,so.ctypes)
    def RM_GetNthSelectedOutputColumnCount(self):
        return self.libc.RM_GetNthSelectedOutputColumnCount(self.id)

    def RM_GetSelectedOutputCount(self):
        return self.libc.RM_GetSelectedOutputCount(self.id)

    def RM_GetSelectedOutputHeading(self,col,headings,length):
        String = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetSelectedOutputHeading(self.id,col,String,length)
        headings[col] = String.value.decode()
        return status

    def RM_GetSelectedOutputColumnCount(self):
        return self.libc.RM_GetSelectedOutputColumnCount(self.id)

    def RM_GetSelectedOutputRowCount(self):
        return self.libc.RM_GetSelectedOutputRowCount(self.id)

    def RM_GetSICount(self):
        return self.libc.RM_GetSICount(self.id)

    def RM_GetSIName(self,num,name,l1):
        return self.libc.RM_GetSIName(self.id,num,name,l1)

    def RM_GetSolidSolutionComponentsCount(self):
        return self.libc.RM_GetSolidSolutionComponentsCount(self.id)

    def RM_GetSolidSolutionComponentsName(self,num,name,l1):
        return self.libc.RM_GetSolidSolutionComponentsName(self.id,num,name,l1)

    def RM_GetSolidSolutionName(self,num,name,l1):
        return self.libc.RM_GetSolidSolutionName(self.id,num,name,l1)

    def RM_GetSolutionVolume(self,vol):
        return self.libc.RM_GetSolutionVolume(self.id,vol.ctypes)

    def RM_GetSpeciesConcentrations(self,species_conc):
        return self.libc.RM_GetSpeciesConcentrations(self.id,species_conc.ctypes)

    def RM_GetSpeciesCount(self):
        return self.libc.RM_GetSpeciesCount(self.id)

    def RM_GetSpeciesD25(self,diffc):
        return self.libc.RM_GetSpeciesD25(self.id,diffc.ctypes)

    def RM_GetSpeciesLog10Gammas(self,species_log10gammas):
        return self.libc.RM_GetSpeciesLog10Gammas(self.id,species_log10gammas)

    def RM_GetSpeciesName(self,num,chem_name,length):
        String = ctypes.create_string_buffer(length)
        status = self.libc.RM_GetSpeciesName(self.id,num,String,length)
        chem_name[num] = String.value.decode()
        return status

    def RM_GetSpeciesSaveOn(self):
        return self.libc.RM_GetSpeciesSaveOn(self.id)
    def RM_GetSpeciesZ(self,Z):
        return self.libc.RM_GetSpeciesZ(self.id,Z)
    def RM_GetStartCell(self,sc):
        return self.libc.RM_GetStartCell(self.id,sc)
    def RM_GetSurfaceName(self,num,name,l1):
        return self.libc.RM_GetSurfaceName(self.id,num,name,l1)
    def RM_GetSurfaceType(self,num,name,l1):
        return self.libc.RM_GetSurfaceType(self.id,num,name,l1)
    def RM_GetThreadCount(self):
        return self.libc.RM_GetThreadCount(self.id)
    def RM_GetTime(self):
        self.libc.RM_GetTime.restype = ctypes.c_double
        return self.libc.RM_GetTime(self.id)
    def RM_GetTimeConversion(self):
        self.libc.RM_GetTimeConversion.restype = ctypes.c_double
        return self.libc.RM_GetTimeConversion(self.id)
    def RM_GetTimeStep(self):
        self.libc.RM_GetTimeStep.restype = ctypes.c_double
        return self.libc.RM_GetTimeStep(self.id)
    def RM_InitialPhreeqc2Module(self,ic1,ic2,f1):
        return self.libc.RM_InitialPhreeqc2Module(self.id,ic1.ctypes,ic2.ctypes,f1.ctypes)

    def RM_InitialPhreeqc2Concentrations(self,c,n_boundary,boundary_solution1,boundary_solution2,fraction1):
        return self.libc.RM_InitialPhreeqc2Concentrations(self.id,c.ctypes,n_boundary,boundary_solution1.ctypes,
                                                          boundary_solution2.ctypes,fraction1.ctypes)

    def RM_InitialPhreeqc2SpeciesConcentrations(self,species_c,n_boundary,boundary_solution1,boundary_solution2,
                                                fraction1):
        return self.libc.RM_InitialPhreeqc2SpeciesConcentrations(self.id,species_c.ctypes,n_boundary.ctypes,
                                                                 boundary_solution1.ctypes,boundary_solution2.ctypes,
                                                                 fraction1.ctypes)

    def RM_InitialPhreeqcCell2Module(self,n,module_numbers,dim_module_numbers):
        return self.libc.RM_InitialPhreeqcCell2Module(self.id,n,module_numbers,dim_module_numbers)

    def RM_LoadDatabase(self,db_name):
        return self.libc.RM_LoadDatabase(self.id,db_name.encode())
    def RM_LogMessage(self,str):
        return self.libc.RM_LogMessage(self.id,str.encode())
    def RM_MpiWorker(self):
        return self.libc.RM_MpiWorker(self.id)
    def RM_MpiWorkerBreak(self):
        return self.libc.RM_MpiWorkerBreak(self.id)
    def RM_OpenFiles(self):
        return self.libc.RM_OpenFiles(self.id)
    def RM_OutputMessage(self,str):
        return self.libc.RM_OutputMessage(self.id,str.encode())
    def RM_RunCells(self):
        return self.libc.RM_RunCells(self.id)
    def RM_RunFile(self,workers,initial_phreeqc,utility,chem_name):
        return self.libc.RM_RunFile(self.id,workers,initial_phreeqc,utility,chem_name.encode())
    def RM_RunString(self,workers,initial_phreeqc,utility,input_string):
        return self.libc.RM_RunString(self.id,workers,initial_phreeqc,utility,input_string.encode())
    def RM_ScreenMessage(self,str):
        return self.libc.RM_ScreenMessage(self.id,str.encode())
    def RM_SetComponentH2O(self,tf):
        return self.libc.RM_SetComponentH2O(self.id,tf)
    def RM_SetConcentrations(self,c):
        return self.libc.RM_SetConcentrations(self.id,c.ctypes)
    def RM_SetCurrentSelectedOutputUserNumber(self,n_user):
        return self.libc.RM_SetCurrentSelectedOutputUserNumber(self.id,n_user)
    def RM_SetDensity(self,density):
        return self.libc.RM_SetDensity (self.id, density.ctypes)
    def RM_SetDumpFileName(self,dump_name):
        return self.libc.RM_SetDumpFileName(self.id,dump_name)
    def RM_SetErrorHandlerMode(self,mode):
        return self.libc.RM_SetErrorHandlerMode(self.id,mode)
    def RM_SetFilePrefix(self,prefix):
        return self.libc.RM_SetFilePrefix(self.id,prefix.encode())
    def RM_SetMpiWorkerCallbackCookie(self,cookie):
        return self.libc.RM_SetMpiWorkerCallbackCookie(self.id,cookie)
    def RM_SetPartitionUZSolids(self,tf):
        return self.libc.RM_SetPartitionUZSolids(self.id,tf)
    def RM_SetPorosity(self, por):
        return self.libc.RM_SetPorosity(self.id, por.ctypes)
    def RM_SetPressure(self, p):
        return self.libc.RM_SetPressure(self.id,p.ctypes)
    def RM_SetPrintChemistryMask(self, cell_mask):
        return self.libc.RM_SetPrintChemistryMask(self.id, cell_mask.ctypes)
    def RM_SetPrintChemistryOn(self,workers,initial_phreeqc,utility):
        return self.libc.RM_SetPrintChemistryOn(self.id,workers,initial_phreeqc,utility)
    def RM_SetRebalanceByCell(self,method):
        return self.libc.RM_SetRebalanceByCell(self.id,method)
    def RM_SetRebalanceFraction(self,f):
        return self.libc.RM_SetRebalanceFraction(self.id,ctypes.c_double(f))
    def RM_SetRepresentativeVolume(self,rv):
        return self.libc.RM_SetRepresentativeVolume(self.id,rv.ctypes)
    def RM_SetSaturation(self,sat):
        return self.libc.RM_SetSaturation(self.id,sat.ctypes)
    def RM_SetScreenOn(self,tf):
        return self.libc.RM_SetScreenOn(self.id,tf)
    def RM_SetSelectedOutputOn(self,selected_output):
        return self.libc.RM_SetSelectedOutputOn(self.id,selected_output)
    def RM_SetSpeciesSaveOn(self,save_on):
        return self.libc.RM_SetSpeciesSaveOn(self.id,save_on)
    def RM_SetTemperature(self,t):
        return self.libc.RM_SetTemperature(self.id,t.ctypes)
    def RM_SetTime(self,time):
        return self.libc.RM_SetTime(self.id,ctypes.c_double(time))
    def RM_SetTimeConversion(self,conv_factor):
        return self.libc.RM_SetTimeConversion(self.id,ctypes.c_double(conv_factor))
    def RM_SetTimeStep(self,time_step):
        return self.libc.RM_SetTimeStep(self.id,ctypes.c_double(time_step))

    def RM_SetUnitsExchange(self,option):
        return self.libc.RM_SetUnitsExchange(self.id,option)

    def RM_SetUnitsGasPhase(self,option) -> None:
        self.libc.RM_SetUnitsGasPhase(self.id,option)

    def RM_SetUnitsKinetics(self,option) -> None:
        self.libc.RM_SetUnitsKinetics(self.id,option)

    def RM_SetUnitsPPassemblage(self,option):
        return self.libc.RM_SetUnitsPPassemblage(self.id,option)

    def RM_SetUnitsSolution(self,option):
        return self.libc.RM_SetUnitsSolution(self.id,option)

    def RM_SetUnitsSSassemblage(self,option):
        return self.libc.RM_SetUnitsSSassemblage(self.id,option)
    def RM_SetUnitsSurface(self,option):
        return self.libc.RM_SetUnitsSurface(self.id,option)

    def RM_SpeciesConcentrations2Module(self,species_conc):
        return self.libc.RM_SpeciesConcentrations2Module(self.id,species_conc.ctypes)

    def RM_UseSolutionDensityVolume(self,tf):
        return self.libc.RM_UseSolutionDensityVolume(self.id,tf)

    def RM_WarningMessage(self,warn_str):
        return self.libc.RM_WarningMessage(self.id,warn_str)
    def RM_GetComponentCount(self):
        return self.libc.RM_GetComponentCount(self.id)

