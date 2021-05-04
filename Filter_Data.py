# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 2020


@Author: The Author

____________________________________________________
Plots to produce:
1. LCC of equipment for each scenario for all the individuals
2, CO2 of equipment for each scenario for all the individuals

3. CO2 vs LCC scatter plot.

4. CO2 vs chiller type
5. CO2 vs CHP type,
6. LCC vs chiller type
7. CO2 vs CHP type

8. Traces of building types across all the runs
____________________________________________________

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



# Constants
Num_Sites = 4


LCC_Var = Num_Sites+11
CO2_Var = Num_Sites+14
WalkScore_Var = Num_Sites+15
GFA_Var = Num_Sites+16
FAR_Var = Num_Sites+17

CHPType_Var = Num_Sites+1
ChillerType_Var = Num_Sites+2

PercentArea_0 = Num_Sites+19
PercentArea_1 = Num_Sites+27


Max_FAR = 20
Max_Site_GFA = 647497/Max_FAR # m2


bldg_types = ['Res','Off','Ret','Sup','Rest','Edu','Med','Lod','Ind']



def DF_Filter(filename, experiment=None, verbose=1):
    
    file = np.loadtxt(filename, dtype='float', delimiter=',')
    
    inputDF = pd.DataFrame(file)
    
    error_tol = 1.15
    

    if verbose: print('+++++ processing %s +++++\n'%(filename))
    
    if verbose: print('Count duplicates:')
    condition1 = inputDF.duplicated()==True
    if verbose: print(inputDF[condition1][GFA_Var].count())
    
    
    if verbose: print('Count under the min GFA:') # Count non-trivial neighborhoods
    condition2 = inputDF[GFA_Var] <= 1/error_tol#<=647497/10
    if verbose: print(inputDF[condition2][GFA_Var].count())
    
    
    if verbose: print('Count over the max GFA:')
    condition3 = inputDF[GFA_Var] >= Max_Site_GFA*Max_FAR*error_tol
    if verbose: print(inputDF[condition3][GFA_Var].count())
    
    
    if verbose: print('Count over the max Site GFA:')
    condition4 = inputDF[GFA_Var]/inputDF[FAR_Var] >= Max_Site_GFA*error_tol
    if verbose: print(inputDF[condition4][GFA_Var].count())
    
    
    if verbose: print('Normalizing the LCC and CO2 Obj Fxns')
    inputDF[LCC_Var] /= inputDF[GFA_Var] # Normalizing LCC ($/m2)
    inputDF[CO2_Var] /= inputDF[GFA_Var] # Normalizing CO2 (Tonnes/m2)
    if verbose: print('Converting percent areas from decimal into percentage')
    for i in range(PercentArea_0,PercentArea_1+1): # Converting percent areas to integer %
        inputDF[i] = inputDF[i] * 100
    
    
    # Filter the data based on the 50-th percentiles of the objective function values
    if experiment == 'WorstHalfLCC':
        if verbose: print('Permit only the worst 50% of individuals in terms of LCC:')
        LCC_percentile = np.nanpercentile(inputDF[LCC_Var], 50)
        if verbose: print("Lowest LCC considered in training data:%.2f"%LCC_percentile)
        condition8 = (inputDF[LCC_Var] >= LCC_percentile)
        
        
    elif experiment == 'BestHalfLCC':
        if verbose: print('Permit only the best 50% of individuals in terms of LCC:')
        LCC_percentile = np.nanpercentile(inputDF[LCC_Var], 50)
        if verbose: print("Highest LCC considered in training data:%.2f"%LCC_percentile)
        condition8 = (inputDF[LCC_Var] <= LCC_percentile)
        
    elif experiment == 'WorstHalfCO2':
        if verbose: print('Permit only the worst 50% of individuals in terms of CO2:')
        CO2_percentile = np.nanpercentile(inputDF[CO2_Var], 50)
        if verbose: print("Lowest CO2 considered in training data: %.2f"%CO2_percentile)
        condition8 = (inputDF[CO2_Var] >= CO2_percentile)
        
    elif experiment == 'WorstHalfWalkScore':
        if verbose: print('Permit only the worst 50% of individuals in terms of WalkScore:')
        WalkScore_percentile = np.nanpercentile(inputDF[WalkScore_Var], 50)
        if verbose: print("Highest Walkscore considered in training data:%.2f"%WalkScore_percentile)
        condition8 = (inputDF[WalkScore_Var] <= WalkScore_percentile)
    
    elif experiment == 'WorstHalfAll':
        if verbose: print('Permit only the worst 50% of individuals in terms of all objective functions:')
        LCC_percentile = np.nanpercentile(inputDF[LCC_Var], 50)
        CO2_percentile = np.nanpercentile(inputDF[CO2_Var], 50)
        WalkScore_percentile = np.nanpercentile(inputDF[WalkScore_Var], 50)
        condition8 = (inputDF[LCC_Var] >= LCC_percentile) & (inputDF[CO2_Var] >= CO2_percentile) & (inputDF[WalkScore_Var] <= WalkScore_percentile)
        
    elif experiment == 'BestHalfAll':
        if verbose: print('Permit only the best 50% of individuals in terms of all objective functions:')
        LCC_percentile = np.nanpercentile(inputDF[LCC_Var], 50)
        CO2_percentile = np.nanpercentile(inputDF[CO2_Var], 50)
        WalkScore_percentile = np.nanpercentile(inputDF[WalkScore_Var], 50)
        condition8 = (inputDF[LCC_Var] <= LCC_percentile) & (inputDF[CO2_Var] <= CO2_percentile) & (inputDF[WalkScore_Var] >= WalkScore_percentile)
    
    elif experiment == 'BestHalfAny':
        print('Permit only the best 50% of individuals in terms of each objective functions:')
        LCC_percentile = np.nanpercentile(inputDF[LCC_Var], 50)
        CO2_percentile = np.nanpercentile(inputDF[CO2_Var], 50)
        WalkScore_percentile = np.nanpercentile(inputDF[WalkScore_Var], 50)
        condition8 = (inputDF[LCC_Var] <= LCC_percentile) | (inputDF[CO2_Var] <= CO2_percentile) | (inputDF[WalkScore_Var] >= WalkScore_percentile)
    
    
    
    
    
    # Filtering the inadmissible results
    Filtered = ~(condition1 | condition2 | condition3 | condition4)
    if experiment != 'FullData':
        inputDF = inputDF[Filtered & condition8]
    else:
        inputDF = inputDF[Filtered]
    
    
    ## Print the lowest values of individual objective functions in the sample popultation when applying combined filters
    if experiment == 'WorstHalfAll' or experiment == 'BestHalfAll' or experiment == 'FullData' or experiment == 'BestHalfAny':
        if verbose: print("Lowest LCC considered in training data:%.2f"%np.min(inputDF[LCC_Var]))
        if verbose: print("Lowest CO2 considered in training data: %.2f"%np.min(inputDF[CO2_Var]))
        if verbose: print("Highest Walkscore considered in training data:%.2f"%np.max(inputDF[WalkScore_Var]))    
        
    
    if verbose: print('Count of valid answers: %d out of %d'%(len(inputDF), len(file)))
    
    
    if verbose: print(inputDF[[LCC_Var, CO2_Var, WalkScore_Var]].describe())
    
    inputDF.reset_index(inplace=True, drop=True)
    

    
    
    return inputDF