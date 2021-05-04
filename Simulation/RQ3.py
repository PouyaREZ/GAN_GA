from __future__ import division
from queue import Queue

# from ctypes import *
## NOTE
# libadd = cdll.LoadLibrary('\usr\lib\x86_64-linux-gnu\libgfortran.so.3')

#-------------------------------------------------------------------------------
# Name:        IntegratedInfrastructureLocationOptimization.py
# Purpose:     To model the location-based impacts of infrastructure systems and
#              building loads.
#
# Author:      Rob Best
# Modified and Updated by The Author
# Created:     04/04/2016
# Updated:     1/20/2021
# Copyright:   (c) Rob Best 2016
#-------------------------------------------------------------------------------

''' General approach:
    -Model the potential building sites as a matrix with costs between them
     encoded in a separate matrix
    -Costs are a combination of pure distance and the cost to build the pipe
     which can be piece-wise linear based on the amount of maximum flow required
     to capture the dimensioning of the pipes
    -Need to investigate if there should also be some penalty for pumping or
     loss aside from purely the distance
    -Distances need to consider additional run beyond the apparent termination
     node to account for the distribution from street center to building
    -Links that are infeasible should be represented with an infinite cost to
     ensure that they are never built
    -Run an optimization that is an assignment or allocation problem, or perhaps
     a Minimum Spanning Tree (TBD as code is written)
'''
# import pdb

import json
import math as mp
import numpy as np
import timeit
import copy
import random as rp
# import sys
import mutPolynomialBoundedInt
import statsmodels.api as sm
import pyDOE as pd

from deap import base
from deap import creator
from deap import tools
import fitness_with_constraints
# from functools import reduce

from scoop import futures

# from pulp import *
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, LpContinuous, lpSum, LpStatusOptimal, GUROBI_CMD#, value

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

# Start a timer so that the optimization is timed and there is a flag upon completion
# Start = timeit.default_timer() # OFFLINED # MODIFIED

'''-----------------------------------------------------------------------------------------------'''
# Universal Constants #
'''-----------------------------------------------------------------------------------------------'''
Btu_to_J = 1055.00                  # Conversion of Btu to J
kWh_to_J = 3600000.00               # Conversion of kWh to J
MWh_to_J = 3600000000.00            # Conversion of MWh to J
# tons_to_J_hr = 12660670.23144        # Conversion of J to tons cooling
kWh_to_Btu = 3412.14                   # Conversion of kWh to Btu
tons_to_MMBtu_hr = 0.012            # Conversion of tons to MMBtu/hr
tons_to_Btu_hr = 12000              # Conversion of tons to Btu/hr
tons_to_kW = 3.5168525              # Conversion of tons to kWh
meters_to_ft = 3.28084              # Conversion of meters to feet
Metric_to_Imperial_Flow = 15850.3   # Conversion of m3/s to gal/min
m_to_in = 39.3701                   # Conversion of in to m
MT_to_lbs = 2204.62                 # Conversion of metric tonnes to lbs

LHV_Natural_Gas = 47100000.00       # Lower heating value of natural gas (J/kg) ## updated from engineeringtoolbox (Apr 2019)#54000000.00       # Lower heating value of natural gas (J/kg)
Specific_Heat_Water = 4216.00       # Specific heat of water at STP (J/kg-C) ## updated from engineeringtoolbox (Apr 2019) #4179.00       # Specific heat of water at STP (J/kg-C)
Density_Water = 999.9               # Density of liquid water (kg/m^3)


Standard_Thermal_Efficiency = 0.8   # Assumed efficiency of heating in a separate system (dimensionless)
Standard_Grid_Efficiency = 0.4      # Assumed grid efficiency of electrical generation and distribution (dimensionless)

Const_Carbon_per_Mil_Dollar = 243   # tonnes CO2 equivalent per million 2007 USD from http://www.eiolca.net/cgi-bin/dft/display.pl?hybrid=no&value=6325700065&newmatrix=US388EPAEEIO2007&second_level_sector=233240&first_level_sector=Utilities+Buildings+And+Infrastructure&key=55196567699&incdemand=1&demandmult=1&selectvect=gwp&top=10

Discount_Rate = 0.035                # Discounted rate for future cash flows # from https://www.whitehouse.gov/wp-content/uploads/2018/12/M-19-05.pdf #0.03                # Discounted rate for future cash flows
Project_Life = 30                   # Years to consider future cash flows for the project
Current_Year = 2019

USD_2008_to_2019 = 1.19             # From http://www.in2013dollars.com/2008-dollars-in-2019
USD_2007_to_2019 = 1.23             # From http://www.in2013dollars.com/us/inflation/2007

Soil_Thermal_Conductivity = 1.6     # Soil thermal conductivity (W/m-K)

'''-----------------------------------------------------------------------------------------------'''
# Optimization Parameters #
'''-----------------------------------------------------------------------------------------------'''
Population_Size = 128
Mutation_Probability = 0.05
Crossover_Probability = 0.75
Eta = 2.5
Number_Generations = 512

# Num_Chillers = 16#2                           # Must be updated if new chillers are added ## NOTE: changed from 16 to 2
Num_Res_Solar = 3                           # Must be updated if new residential solar types are added
Num_Comm_Solar = 3                          # Must be updated if new commercial solar types are added
# Max_Buildings_per_Type = 5
Building_Min = 0
Supply_Min = 1
Max_GFA = 647497#24281200                          # square meters
Max_FAR = 20 # TO NOTE ++ 5 in my SF case study
Total_Site_GFA = 647497/Max_FAR #1214060                    # square meters
Max_Average_Height = 150                 # m
Max_Solar = 60.0   ## 0 in my SF case study # TO NOTE                         # Percentage of available solar roof area: from http://www.nrel.gov/docs/fy14osti/60593.pdf

## TO NOTE
Min_Res = 0.0                               # Percentage of total GFA
Max_Res = 100.0                             # Percentage of total GFA
Min_Off = 0.0                               # Percentage of total GFA
Max_Off = 100.0                             # Percentage of total GFA
Min_Ret = 0.0                               # Percentage of total GFA
Max_Ret = 100.0                             # Percentage of total GFA
Min_Sup = 0.0                               # Percentage of total GFA
Max_Sup = 100.0                             # Percentage of total GFA
Min_Rest = 0.0                              # Percentage of total GFA
Max_Rest = 100.0                            # Percentage of total GFA
Min_Edu = 0.0                               # Percentage of total GFA
Max_Edu = 100.0                             # Percentage of total GFA
Min_Med = 0.0                               # Percentage of total GFA
Max_Med = 100.0                             # Percentage of total GFA
Min_Lod = 0.0                               # Percentage of total GFA
Max_Lod = 100.0                             # Percentage of total GFA
Min_Ind = 0.0                               # Percentage of total GFA
Max_Ind = 100.0                             # Percentage of total GFA

Num_Test_Points = 100                       # unitless
Num_LHS_Points = 100                        # unitless

Num_Iterations = 2                         # unitless
Num_Heat_Test_Points = 100                  # unitless

'''-----------------------------------------------------------------------------------------------'''
# Walkability Index Parameters #
'''-----------------------------------------------------------------------------------------------'''
# Note that these are from the Rakha and Reinhart paper (http://static1.squarespace.com/static/53d65c30e4b0d86829f32e6f/t/53f3c509e4b06927b947aba6/1408484616999/SB12_TS04b_3_Rakha.pdf)
# Note that Coffee, Banks, Books, and Entertainment are not currently used but included for completeness
Grocery_Weights = [3]
Restaurants_Weights = [0.75,0.45,0.25,0.25,0.225,0.225,0.225,0.225,0.2,0.2]
Shopping_Weights = [0.5,0.45,0.4,0.35,0.3]
Coffee_Weights = [1.25,0.75]
Banks_Weights = [1]
Parks_Weights = [1]
Schools_Weights = [1]
Books_Weights = [1]
Entertainment_Weights = [1]
Num_Countable_Grocery = len(Grocery_Weights)
Num_Countable_Restaurants = len(Restaurants_Weights)
Num_Countable_Shopping = len(Shopping_Weights)
Num_Countable_Coffee = len(Coffee_Weights)
Num_Countable_Banks = len(Banks_Weights)
Num_Countable_Parks = len(Parks_Weights)
Num_Countable_Schools = len(Schools_Weights)
Num_Countable_Books = len(Books_Weights)
Num_Countable_Entertainment = len(Entertainment_Weights) 

Walk_Weight_1_m = -1.1733                   # Straight line function slope for the weight of amenities between 0.25 and 1 miles
Walk_Weight_1_b = 1.29325                   # Straight line function intercept for the weight of amenities between 0.25 and 1 miles
Walk_Weight_2_m = -0.12                     # Straight line function slope for the weight of amenities between 1 and 2 miles
Walk_Weight_2_b = 0.24                      # Straight line function intercept for the weight of amenities between 1 and 2 miles

'''-----------------------------------------------------------------------------------------------'''
# Municipal Lighting Parameters #
'''-----------------------------------------------------------------------------------------------'''
Curfew_Modifier = 0.50                      # Fraction
Light_Spacing = 48.8                        # m
Lights_Per_Side = 2                         # Number
Light_Power = .190                          # kW
E_W_Street_Spacing = 2                      # Number of buildings between E-W running streets
N_S_Street_Spacing = 12                     # Number of buildings between N-S running streets
Width_to_Length_Ratio = 1/8


'''-----------------------------------------------------------------------------------------------'''
# CHP Parameters #
'''-----------------------------------------------------------------------------------------------'''
Gas_Line_Pressure = 55.0 
ft3_to_Btu_Nat_Gas = 1037			# https://www.eia.gov/tools/faqs/faq.php?id=45&t=8
kg_to_Btu_H2 = 115119				# https://www.nrel.gov/docs/gen/fy08/43061.pdf
ton_to_Btu_Biomass = 5.74*10**6		# https://www.fpl.fs.fed.us/documnts/techline/fuel-value-calculator.pdf                   # psi
Natural_Gas_Cost = 7.05/ft3_to_Btu_Nat_Gas 	# $/Btu 2017 avg industrial gas price from https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_SCA_a.htm # Old value: 0.000006
Hydrogen_Cost = 15/kg_to_Btu_H2  			# $/Btu   # retail price from https://cafcp.org/sites/default/files/CAFCR.pdf # Old price 0.000044
Biomass_Cost = 30/ton_to_Btu_Biomass		# $/Btu		# https://www.eia.gov/biofuels/biomass/?year=2018&month=12#table_data # Old price 0.000002 

'''-----------------------------------------------------------------------------------------------'''
# Solar Parameters #
'''-----------------------------------------------------------------------------------------------'''
Tilt = 20                                   # degrees
Azimuth = 180                               # degrees

'''-----------------------------------------------------------------------------------------------'''
# Chiller Parameters #
'''-----------------------------------------------------------------------------------------------'''
Chilled_Water_Supply_Temperature = 44.0 # deg F # 6.67     # deg C
Number_Iterations = 1                       # dimensionless
Heat_Source_Temperature = 100 # deg F #37.78             # deg C         ### WHY? WHERE DOES THIS COME FROM?
Cooling_Tower_Approach = 5.0 # deg F # 1.11               # Celsius

'''-----------------------------------------------------------------------------------------------'''
# DHC Parameters # # TO NOTE
'''-----------------------------------------------------------------------------------------------'''
Trench_Cost = 0.0                           # $/m (IGNORED FOR NOW--NEED TO CHECK) # TO NOTE
Max_Supply_Temp_Heating = 95.0              # deg C
Min_Supply_Temp_Heating = 50.0              # deg C
Summer_Thermal_Reset_Hour = 2880            # Assume: Summer is May, June, July, August, September
Winter_Thermal_Reset_Hour = 5808            # Assume: Winter is January, February, March, April, October, November, December
Min_Summer_Heating_Reset_del_T = 0          # deg C
Min_Winter_Cooling_Reset_del_T = 0          # deg C
Max_Summer_Heating_Reset_del_T = 10         # deg C
Max_Winter_Cooling_Reset_del_T = 3          # deg C
Heating_Loop_Max_delta_T = 30.0             # deg C
Heating_Loop_Min_delta_T = 20.0             # deg C 
Min_Secondary_Temp_Heating = 20.0           # deg C (Based on idea of minimum temperature water on secondary heating side is 68 F = 20 C due to internal setpoint; could be lower if mixing with Outside Air, assumes perfect heat transfer)
Max_Supply_Temp_Cooling = 8.0               # deg C
Min_Supply_Temp_Cooling = 1.0               # deg C
Cooling_Loop_Max_delta_T = 10.0             # deg C
Cooling_Loop_Min_delta_T = 3.0              # deg C
Max_Secondary_Cooling_T = 14.4              # deg C (From IEA District Heating and Cooling Connection Handbook--http://www.districtenergy.org/assets/CDEA/Best-Practice/IEA-District-Heating-and-Cooling-Connection-Handbook.pdf)
Heat_Exchanger_Approach = 2.0               # deg C (Assumption)
Depth_of_Pipes = 1.03515                    # m
Distance_Betweeen_Pipes = 0.1703            # m
Loss_Number_Iterations = 1                  # dimensionless
Min_Flow_Required = 0.1                     # Fraction of flow required in every hour by the DHC systems at each node
Pump_Efficiency = 0.85                      # unitless

'''-----------------------------------------------------------------------------------------------'''
# Secondary System Parameters #
'''-----------------------------------------------------------------------------------------------'''
Fan_Coil_Approach = 2.0                     # deg C (Assumption)
Max_Secondary_Temp_Cooling = 14.4           # deg C (From IEA District Heating and Cooling Connection Handbook)
Min_Secondary_Temp_Heating = 33.0           # deg C
Heating_Setpoint = 21                       # deg C
Cooling_Setpoint = 24                       # deg C
Pressure_Differential = 100                 # kPa
Diversifier_Peak = 0.8                      # unitless, based on assumptions in practice ## What's the resource?

'''-----------------------------------------------------------------------------------------------'''
# Import Helper Functions #
'''-----------------------------------------------------------------------------------------------'''
import RemoveHeaders as RH
import ElectricChillers as EC
import CHPEngines as CHP
import AbsorptionChillers as AC
# import SolarPanels as SP
import PairDistance as PD




# Defining cmp from older versions of Python ## ADDED
def cmp(a, b):
    return (a > b) - (a < b) 



'''-----------------------------------------------------------------------------------------------'''
# Initialize Data for Site, Buildings, CHP Engines #
'''-----------------------------------------------------------------------------------------------'''

# Create inputs for site parameters
Site_Info = np.genfromtxt('Site_Info.csv', 'float', delimiter=',')
Site_Info = RH.RemoveHeaders(Site_Info)
Latitude = Site_Info[0,0]
Longitude = Site_Info[0,1]
Altitude = Site_Info[0,2]
UTC = Site_Info[0,3]

# Create input array for the electricity pricing
Grid_Parameters = np.genfromtxt('Grid_Parameters.csv', 'float', delimiter=',')
Grid_Parameters = RH.RemoveHeaders(Grid_Parameters)

Buy_Price = Grid_Parameters[:,0]
Sell_Price = Grid_Parameters[:,1]
Grid_Emissions = Grid_Parameters[:,2]
Demand_Charge = Grid_Parameters[:,3]

# Create input arrays for electricity, heating, and cooling
Electricity_Input = np.genfromtxt('Hourly_Electricity_Toy.csv', 'float', delimiter=',')
Electricity_Input = RH.RemoveHeaders(Electricity_Input)
Heating_Input = np.genfromtxt('Hourly_Heating_Toy.csv', 'float', delimiter=',')
Heating_Input = RH.RemoveHeaders(Heating_Input)
Cooling_Input = np.genfromtxt('Hourly_Cooling_Toy.csv', 'float', delimiter=',')
Cooling_Input = RH.RemoveHeaders(Cooling_Input)

# Create input arrays for weather
Weather = np.genfromtxt('Hourly_Weather_CA_SAN_FRANCISCO.csv', 'float', delimiter=',')
Weather = RH.RemoveHeaders(Weather)
Hourly_Temperature = Weather[:,0]
Hourly_Wet_Bulb = Weather[:,1]
Hourly_DNI = Weather[:,2]
Hourly_DHI = Weather[:,3]
Hourly_GHI = Weather[:,4]
Hourly_Albedo = Weather[:,5]
Hourly_Wind_Speed = Weather[:,6]
Hourly_Ground_Temperature = Weather[:,7]
Hourly_Mains_Water_Temperature = Weather[:,8]

# Find the max and min hourly ground temperature
Min_Ground_Temperature = np.min(Hourly_Ground_Temperature)
Max_Ground_Temperature = np.max(Hourly_Ground_Temperature)

# Create input array for building, CHP, and Chiller metadata
Building_Info = np.genfromtxt('Building_Info_Toy.csv', 'float', delimiter=',')
Building_Info = RH.RemoveHeaders(Building_Info)

CHP_Info = np.genfromtxt('CHP_Info_Toy.csv', 'float', delimiter=',')
CHP_Info = RH.RemoveHeaders(CHP_Info)

Chiller_Info = np.genfromtxt('Chiller_Info_Toy.csv', 'float', delimiter=',')
Chiller_Info = RH.RemoveHeaders(Chiller_Info)

# Define an index for the number of buildings in the simulation
Num_Buildings = len(Electricity_Input[1])

# Define an index for the number of CHP units in the simulation
Num_Engines = len(CHP_Info[:,0])

# Define and index for the number of Chillers in the simulation ## ADDED
Num_Chillers = len(Chiller_Info[:,0])

# Set up the dictionary that will hold the demand information
Init_Demand_Types = {}

# Populate the dictionary with arrays of Electricity, Heating, and Cooling in that order
for i in range(Num_Buildings):
    j=i+1
    Init_Demand_Types[j] = np.column_stack((Electricity_Input[:,i:i+1], Heating_Input[:,i:i+1], Cooling_Input[:,i:i+1]))    # Populate the dictionary with 8760x3 arrays of demand values per building

# Set up dictionaries that will hold CHP info
Power_to_Heat = {}
CHP_Variable_Cost = {}
Max_Unit_Size = {}
CHP_Capital_Cost = {}
CHP_Fuel_Type = {}
CHP_Heat_Rate = {}

# Populate the dictionary
for i in range(Num_Engines):
    j=i+1
    Power_to_Heat[j] = CHP_Info[i,1]
    CHP_Variable_Cost[j] = CHP_Info[i,6]
    Max_Unit_Size[j] = CHP_Info[i,0]
    CHP_Capital_Cost[j] = CHP_Info[i,8]
    CHP_Fuel_Type[j] = CHP_Info[i,9]
    CHP_Heat_Rate[j] = CHP_Info[i,4]

# Set up dictionaries that will hold chiller info
Chiller_Nominal_COP = {}
Chiller_Capital_Cost = {}
for i in range(Num_Chillers):
    j = i+1
    Chiller_Nominal_COP[j] = Chiller_Info[i,0]
    Chiller_Capital_Cost[j] = Chiller_Info[i,1]

# Create additional dictionaries that have the metadata for use in calculating constraints
# Note that the order below should be the order of the columns in the Building_Info.csv sheet
Dwelling_Units = {}
Jobs = {}
GFA = {}
Site_GFA = {}
Stories = {}
Floor_Height = {}
Dimensions = {}
Site_Dimensions = {}
Solar_Roof_Area = {}
Res_GFA = {}
Off_GFA = {}
Ret_GFA = {}
Sup_GFA = {}
Rest_GFA = {}
Edu_GFA = {}
Med_GFA = {}
Lod_GFA = {}
Ind_GFA = {}
Building_Capex = {}
Building_Revenue = {}

for i in range(Num_Buildings):
    j=i+1
    Dwelling_Units[j] = Building_Info[i,0]
    Jobs[j] = Building_Info[i,1]
    GFA[j] = Building_Info[i,2]
    Site_GFA[j] = Building_Info[i,3]
    Stories[j] = Building_Info[i,4]
    Floor_Height[j] = Building_Info[i,5]
    Dimensions[j] = (Building_Info[i,6], Building_Info[i,7])
    Site_Dimensions[j] = (Building_Info[i,8], Building_Info[i,9])
    Solar_Roof_Area[j] = Building_Info[i,10]
    Res_GFA[j] = Building_Info[i,11]
    Off_GFA[j] = Building_Info[i,12]
    Ret_GFA[j] = Building_Info[i,13]
    Sup_GFA[j] = Building_Info[i,14]
    Rest_GFA[j] = Building_Info[i,15]
    Edu_GFA[j] = Building_Info[i,16]
    Med_GFA[j] = Building_Info[i,17]
    Lod_GFA[j] = Building_Info[i,18]
    Ind_GFA[j] = Building_Info[i,19]
    Building_Capex[j] = Building_Info[i,20]
    Building_Revenue[j] = Building_Info[i,21]
    
# Import the Pipe Information and Create Dictionaries with Relevant Information
Heat_Pipe_Info = np.genfromtxt('Heat_Pipe_Info_Toy.csv', 'float', delimiter=',')
Heat_Pipe_Info = RH.RemoveHeaders(Heat_Pipe_Info)

Heat_Pipe_Text = np.genfromtxt('Heat_Pipe_Info_Toy.csv', '|S', delimiter=',')

Cool_Pipe_Info = np.genfromtxt('Cool_Pipe_Info_Toy.csv', 'float', delimiter=',')
Cool_Pipe_Info = RH.RemoveHeaders(Cool_Pipe_Info)

Cool_Pipe_Text = np.genfromtxt('Cool_Pipe_Info_Toy.csv', '|S', delimiter=',')

Num_Heat_Pipes = len(Heat_Pipe_Info[:,0])
Num_Cool_Pipes = len(Cool_Pipe_Info[:,0])

Heat_Pipe_Diameters = {}                 # m
Heat_Pipe_Min_Flows = {}                 # m3/hr
Heat_Pipe_Max_Flows = {}                 # m3/hr
Heat_Pipe_Max_Pressures = {}             # kPa
Heat_Pipe_HW_Roughnesses = {}            # Dimensionless
Heat_Pipe_CapEx = {}                     # $/m
Heat_Pipe_Insulation_Thicknesses = {}    # m
Heat_Pipe_Thermal_Conductivities = {}    # W/m-K
Heat_Pipe_Pump_Regression_Val_1 = {}     # dimensionless, but for flows in m3/hr and power in kW
Heat_Pipe_Pump_Regression_Val_0 = {}     # dimensionless, but for flows in m3/hr and power in kW
Heat_Pipe_Names = []                     # text only from the Heat_Pipe_Text array and must only have one header line in the Heat_Pipe_Info File
Heat_Pipe_Lookup = {}                    # uses the Heat_Pipe_Text as the key and the j value as the value so that the index can be looked up later
Heat_Pipe_Numbers = []                   # A list of numbers that correspond to the heat metadata
Heat_Pipe_U1 = {}                        # W/m-K, heat transfer coefficient defining transfer between pipe and ground
Heat_Pipe_U2 = {}                        # W/m-K, heat transfer coefficient defining transfer between pipes
Heat_Pipe_U = {}                         # W/m-K, overall heat transfer coefficient
Heat_Pipe_Kinked_Regression_Vals = {}    # dimensionless, but for flows in m3/hr and power in kW

Cool_Pipe_Diameters = {}                 # m
Cool_Pipe_Min_Flows = {}                 # m3/hr
Cool_Pipe_Max_Flows = {}                 # m3/hr
Cool_Pipe_Max_Pressures = {}             # kPa
Cool_Pipe_HW_Roughnesses = {}            # Dimensionless
Cool_Pipe_CapEx = {}                     # $/m
Cool_Pipe_Insulation_Thicknesses = {}    # m
Cool_Pipe_Thermal_Conductivities = {}    # W/m-K
Cool_Pipe_Pump_Regression_Val_1 = {}     # dimensionless, but for flows in m3/hr and power in kW
Cool_Pipe_Pump_Regression_Val_0 = {}     # dimensionless, but for flows in m3/hr and power in kW
Cool_Pipe_Names = []                     # text only from the Cool_Pipe_Text array and must only have one header line in the Cool_Pipe_Info file
Cool_Pipe_Lookup = {}                    # uses the Cool_Pipe_Text as the key and the j value as the value so that the index can be looked up later
Cool_Pipe_Numbers = []                   # A list of numbers that correspond to the heat metadata
Cool_Pipe_U1 = {}                        # W/m-K, heat transfer coefficient defining transfer between pipe and ground
Cool_Pipe_U2 = {}                        # W/m-K, heat transfer coefficient defining transfer between pipes
Cool_Pipe_U = {}                         # W/m-K, overall heat transfer coefficient
Cool_Pipe_Kinked_Regression_Vals = {}    # dimensionless, but for flows in m3/hr and power in kW

for i in range(Num_Heat_Pipes):
    j = i+1
    Heat_Pipe_Diameters[j] = Heat_Pipe_Info[i,0]
    Heat_Pipe_Min_Flows[j] = Heat_Pipe_Info[i,1]
    Heat_Pipe_Max_Flows[j] = Heat_Pipe_Info[i,2]
    Heat_Pipe_HW_Roughnesses[j] = Heat_Pipe_Info[i,3]
    Heat_Pipe_CapEx[j] = Heat_Pipe_Info[i,4]
    Heat_Pipe_Insulation_Thicknesses[j] = Heat_Pipe_Info[i,5]
    Heat_Pipe_Thermal_Conductivities[j] = Heat_Pipe_Info[i,6]
    Heat_Pipe_Pump_Regression_Val_1[j] = Heat_Pipe_Info[i,7]
    Heat_Pipe_Pump_Regression_Val_0[j] = Heat_Pipe_Info[i,8]
    Heat_Pipe_Names += [Heat_Pipe_Text[j,0]]
    Heat_Pipe_Lookup[Heat_Pipe_Text[j,0]] = j
    Heat_Pipe_Numbers += [j]
    Pipe_Depth_to_Center = Depth_of_Pipes+Heat_Pipe_Diameters[j]/2    # Find the distance of depth to the center to calculate the pipe heat interaction
    Pipe_Distance_Center_to_Center = Distance_Betweeen_Pipes+Heat_Pipe_Diameters[j]   # Find the distance between centers of the supply and return pipes to calculate the pipe heat interaction
    R_m = 1/(4*mp.pi*Soil_Thermal_Conductivity)*mp.log(1+(2*Pipe_Depth_to_Center/Pipe_Distance_Center_to_Center)**2)    # Calculate the heat resistance term between supply and return pipes
    Heat_Pipe_U1[j] = (1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])/((1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])**2-R_m**2)
    Heat_Pipe_U2[j] = R_m/((1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])**2-R_m**2)
    Heat_Pipe_U[j] = (1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])/((1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])**2-R_m**2)-R_m/((1/Soil_Thermal_Conductivity+1/Heat_Pipe_Info[i,6])**2-R_m**2)
    Heat_Pipe_Kinked_Regression_Vals[j] = Heat_Pipe_Info[i,9]

for i in range(Num_Cool_Pipes):
    j = i+1
    Cool_Pipe_Diameters[j] = Cool_Pipe_Info[i,0]
    Cool_Pipe_Min_Flows[j] = Cool_Pipe_Info[i,1]
    Cool_Pipe_Max_Flows[j] = Cool_Pipe_Info[i,2]
    Cool_Pipe_HW_Roughnesses[j] = Cool_Pipe_Info[i,3]
    Cool_Pipe_CapEx[j] = Cool_Pipe_Info[i,4]
    Cool_Pipe_Insulation_Thicknesses[j] = Cool_Pipe_Info[i,5]
    Cool_Pipe_Thermal_Conductivities[j] = Cool_Pipe_Info[i,6]
    Cool_Pipe_Pump_Regression_Val_1[j] = Cool_Pipe_Info[i,7]
    Cool_Pipe_Pump_Regression_Val_0[j] = Cool_Pipe_Info[i,8]
    Cool_Pipe_Names += [Cool_Pipe_Text[j,0]]
    Cool_Pipe_Lookup[Cool_Pipe_Text[j,0]] = j
    Cool_Pipe_Numbers += [j]
    Pipe_Depth_to_Center = Depth_of_Pipes+Cool_Pipe_Diameters[j]/2    # Find the distance of depth to the center to calculate the pipe heat interaction
    Pipe_Distance_Center_to_Center = Distance_Betweeen_Pipes+Cool_Pipe_Diameters[j]   # Find the distance between centers of the supply and return pipes to calculate the pipe heat interaction
    R_m = 1/(4*mp.pi*Soil_Thermal_Conductivity)*mp.log(1+(2*Pipe_Depth_to_Center/Pipe_Distance_Center_to_Center)**2)    # Calculate the heat resistance term between supply and return pipes
    Cool_Pipe_U1[j] = (1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])/((1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])**2-R_m**2)
    Cool_Pipe_U2[j] = R_m/((1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])**2-R_m**2)
    Cool_Pipe_U[j] = (1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])/((1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])**2-R_m**2)-R_m/((1/Soil_Thermal_Conductivity+1/Cool_Pipe_Info[i,6])**2-R_m**2)
    Cool_Pipe_Kinked_Regression_Vals[j] = Cool_Pipe_Info[i,9]

# Import the list of available nodes and the (x,y) coordinates (in km) from csv
Sites = np.genfromtxt('Sites_Toy.csv', 'float', delimiter=',')
Sites = RH.RemoveHeaders(Sites)

# Import the matrix of allowable connections. This should be a csv with a "0" where a connection is disallowed and a "1" where a connection is allowed. Diagonal should always be zero, and it should be symmetric
Connections = np.genfromtxt('Connections_Toy.csv', 'float', delimiter = ',')
Connections= RH.RemoveHeaders(Connections)

# Create a counter for the number of possible sites
Num_Sites = len(Sites[:,0])

# Create a list for just the site numbers, also called nodes ### Can this be improved? I had tried it before, not sure if this is the one I experimented with. Should be inspected later on.
# global Nodes
Nodes = np.zeros((Num_Sites))
for n in range(Num_Sites):
    Nodes[n] = n
Nodes = [int(i) for i in Nodes]

# Isolate just the x and y coordinates
Site_Coordinates = Sites[:,-2:]

# Create lists of the hot arcs and cold arcs
Heat_Arcs = []
Cool_Arcs = []
Trench_Arcs = []
Arc_Distances = {}
for i in Nodes:
    for j in Nodes:
        if Connections[i,j] > 0.5:
            Trench_Arcs += [(i,j)]
            Arc_Distances[(i,j)] = PD.PairedDistance(Site_Coordinates[i-1,:], Site_Coordinates[j-1,:])
        for m in Heat_Pipe_Numbers:
            if Connections[i,j] > 0.5:
                Heat_Arcs += [(i,j,m)]
        for n in Cool_Pipe_Numbers:
            if Connections[i,j] > 0.5:
                Cool_Arcs += [(i,j,n)]

# Isolate just the x and y coordinates
Site_Coordinates = Sites[:,-2:]

# Import the matrix of added costs for trenching not related to standard trench costs
Added_Trench_Costs = np.genfromtxt('Trench_Cost_Additions_Toy.csv', 'float', delimiter = ',')
Added_Trench_Costs = RH.RemoveHeaders(Added_Trench_Costs)

'''-----------------------------------------------------------------------------------------------'''
# Prep Parameters for the Location Optimization #
'''-----------------------------------------------------------------------------------------------'''
# Create the modified input matrices for calculating annual heat and cool flows
Mod_Heating_Input = np.zeros((8760,Num_Buildings))
Mod_Cooling_Input = np.zeros((8760,Num_Buildings))

for building in range(Num_Buildings):
    Test_Vector = Heating_Input[:,building]
    Mod_Heating_Input[:,building] = np.where(Test_Vector < Min_Flow_Required*max(Test_Vector), Min_Flow_Required*max(Test_Vector), Test_Vector)
    Test_Vector = Cooling_Input[:,building]
    Mod_Cooling_Input[:,building] = np.where(Test_Vector < Min_Flow_Required*max(Test_Vector), Min_Flow_Required*max(Test_Vector), Test_Vector)

# For each arc in Heat_Arcs and Cool_Arcs, come up with the multiple linear regression based on flow rate, supply temperature, and return temperature.
# This is a multi-step process whereby first a Latin Hypercube Sample of the multivariate space is performed to create the sampling variables.
# Next, each point in the LHS sample has the heat loss calculated through an iterative process. After that, a multiple linear regression is performed
# to find the coefficients and R-squared value of each of the heat losses as a function of the two temperatures and flow rate. Note that this function
# reports the coefficients in the reverse order of how they are entered.
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

Heat_Loss_Reg_Coefs = {}
Cool_Loss_Reg_Coefs = {}

# Prep the trench cost dictionary by summing added costs and the basic trench costs
# Note that the costs are divided by two because later on, but the supply and return trench costs will be triggered
Total_Trench_Costs = {}
for t in Trench_Arcs:
    Total_Trench_Costs[t] = (Added_Trench_Costs[t[0],t[1]] + Trench_Cost)/2

# Now prep the average ground temperature for use in the thermal loss calculations
Average_Ground_Temperature = np.average(Hourly_Ground_Temperature)

# Set up a vector of hours that will be used in computing delay times
Tau_Hours = np.zeros((8760))
for hour in range(8760):
    Tau_Hours[hour] = hour
Tau_Hours = Tau_Hours.astype(int)

# Prep the dictionaries that will hold fixed pipe construction cost for each arc per diameter and for each resource
Heat_Arc_Capex = {}
Cool_Arc_Capex = {}

# Prep the dictionaries that will hold the max flows on each arc
Heat_Arc_Max_Flows = {}
Heat_Arc_Min_Flows = {}
Cool_Arc_Max_Flows = {}
Cool_Arc_Min_Flows = {}

# Now calculate the Heat and Cooling Loss Costs that will be incurred when a pipe is built. Here we are following the idea that 1) losses are greater when more pipes are built,
# 2) losses are greater when longer pipes are built, and 3) losses are greater when pipes are larger. Loss percent as a function of total utilization is what we really care about
# but this is hard to derive and optimize since it is very interdependent. Instead, follow the method of Tol and Svendsen (2012, doi: 10.1016/j.energy.2011.12.002) and minimize
# the pipe length and volume as a proxy for energy savings. We want a driver that will shrink number of pipes, length, and diameter. Assigning the costs below to the binary variables
# that control whether or not a pipe is built is the way we will tackle this.
# At the same time...
# Now calculate the pumping costs that will be incurred to move a certain amount of hot or cold water through a pipe. Given that piping in parallel only uses the critical path to
# determine the pump sizing, we will do this by minimizing losses. Here head loss is the predominant loss and that will be tracked back to electricity costs for pumping. This will
# allow equation of heat losses and capital expenses, and though it will not govern the pump cost in full, it is some approximation over the year of the costs that are wasted in 
# moving water throughout the network.
Heat_Pump_Reg_Val_1 = {}
Heat_Pump_Reg_Val_0 = {}
Heat_Pump_Kinked_Reg_Val = {}
Cool_Pump_Reg_Val_1 = {}
Cool_Pump_Reg_Val_0 = {}
Cool_Pump_Kinked_Reg_Val = {}

# Add data to all of these dictionaries
for h in Heat_Arcs:
    #print(h) ## NOTE: OFFLINED!
    Heat_Arc_Capex[h] = 2*Heat_Pipe_CapEx[h[2]]*Arc_Distances[(h[0],h[1])]           # Note the factor of 2 is because supply and return pipes are required and are assumed both to be insulated the same (identical)
    Heat_Arc_Max_Flows[h] = Heat_Pipe_Max_Flows[h[2]]
    Heat_Arc_Min_Flows[h] = Heat_Pipe_Min_Flows[h[2]]
    Heat_Pump_Reg_Val_1[h] = Heat_Pipe_Pump_Regression_Val_1[h[2]]*Arc_Distances[(h[0],h[1])]   # Convert from kWh lost/m pipe/m3 flow to kWh lost/m3 flow ## END modified from "USD lost/m3 flow"
    Heat_Pump_Reg_Val_0[h] = Heat_Pipe_Pump_Regression_Val_0[h[2]]*Arc_Distances[(h[0],h[1])]   # Convert from kWh lost/m pipe/m3 flow to kWh lost/m3 flow ## END modified from "USD lost/m3 flow"
    Heat_Pump_Kinked_Reg_Val[h] = Heat_Pipe_Kinked_Regression_Vals[h[2]]*Arc_Distances[(h[0],h[1])] # Convert from kWh lost/m pipe/m3 flow to kWh lost/m3 flow ## END modified from "USD lost/m3 flow"
    Samples = pd.lhs(4, samples = Num_LHS_Points)
    Variables = [[],[],[],[]]
    Heat_Losses = []
    for i in range(len(Samples)):
        if h[2] == 1:
            Samples[i,0] = Heat_Pipe_Max_Flows[h[2]]-Samples[i,0]*(Heat_Pipe_Max_Flows[h[2]]-Heat_Pipe_Min_Flows[h[2]])*3/4
        else:
            Samples[i,0] = Heat_Pipe_Max_Flows[h[2]]-Samples[i,0]*(Heat_Pipe_Max_Flows[h[2]]-Heat_Pipe_Min_Flows[h[2]])
        Samples[i,1] = Samples[i,1]*(Max_Supply_Temp_Heating-Min_Supply_Temp_Heating)+Min_Supply_Temp_Heating
        Samples[i,2] = Samples[i,2]*(Heating_Loop_Max_delta_T-Heating_Loop_Min_delta_T)+Heating_Loop_Min_delta_T
        Samples[i,3] = Samples[i,3]*(Max_Ground_Temperature-Min_Ground_Temperature)+Min_Ground_Temperature
          
        Variables[0] += [Samples[i,0]]
        Variables[1] += [Samples[i,1]]
        Variables[2] += [Samples[i,2]]
        Variables[3] += [Samples[i,3]]
      
        delta_L = Arc_Distances[(h[0],h[1])]/Num_Test_Points
          
        for n in range(Num_Test_Points):
            if n == 0:
                Delta_ST = Samples[i,1]-(delta_L*(Heat_Pipe_U[h[2]]*(Samples[i,1]-Samples[i,3])+Heat_Pipe_U2[h[2]]*Samples[i,2]))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
                Delta_RT = (Samples[i,1]-Samples[i,2])+delta_L*(Heat_Pipe_U[h[2]]*(Samples[i,1]-Samples[i,2]-Samples[i,3])-Heat_Pipe_U2[h[2]]*Samples[i,2])/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
            else:
                Delta_ST = Delta_ST-(delta_L*(Heat_Pipe_U[h[2]]*(Delta_ST-Samples[i,3])+Heat_Pipe_U2[h[2]]*(Delta_ST-Delta_RT)))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
                Delta_RT = Delta_RT+delta_L*(Heat_Pipe_U[h[2]]*(Delta_RT-Samples[i,3])-Heat_Pipe_U2[h[2]]*(Delta_ST-Delta_RT))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
        Heat_Losses += [Samples[i,0]*Density_Water/3600*(Specific_Heat_Water/1000)*((Samples[i,1]-Delta_ST)+(Delta_RT-(Samples[i,1]-Samples[i,2])))]        # kWh
                 
    results = reg_m(Heat_Losses, Variables)
      
    Heat_Loss_Reg_Coefs[h] = [results.params[3], results.params[2], results.params[1], results.params[0], results.params[4]]             # Note that this will have 5 entries. The first is the multiplier on Flow Rate. The second is on Max Temperature. The third is on temperature difference. The fourth is on Ground Temperature. The fifth is the constant.    
    
for c in Cool_Arcs:
#    print(c) ## NOTE: OFFLINED
    Cool_Arc_Capex[c] = 2*Cool_Pipe_CapEx[c[2]]*Arc_Distances[(c[0],c[1])]           # Note the factor of 2 is because supply and return pipes are required and are assumed both to be insulated the same (identical)
    Cool_Arc_Max_Flows[c] = Cool_Pipe_Max_Flows[c[2]]
    Cool_Arc_Min_Flows[c] = Cool_Pipe_Min_Flows[c[2]]
    Cool_Pump_Reg_Val_1[c] = Cool_Pipe_Pump_Regression_Val_1[c[2]]*Arc_Distances[(c[0],c[1])]   # Convert from kWh lost/m pipe/m3 flow to USD lost/m3 flow
    Cool_Pump_Reg_Val_0[c] = Cool_Pipe_Pump_Regression_Val_0[c[2]]*Arc_Distances[(c[0],c[1])]   # Convert from kWh lost/m pipe/m3 flow to USD lost/m3 flow
    Cool_Pump_Kinked_Reg_Val[c] = Cool_Pipe_Kinked_Regression_Vals[c[2]]*Arc_Distances[(c[0],c[1])] # Convert from kWh lost/m pipe/m3 flow to USD lost/m3 flow
    Samples = pd.lhs(4, samples = Num_LHS_Points)
    Variables = [[],[],[],[]]
    Heat_Losses = []
    for i in range(len(Samples)):
        if c[2] == 1:
            Samples[i,0] = Cool_Pipe_Max_Flows[c[2]]-Samples[i,0]*(Cool_Pipe_Max_Flows[c[2]]-Cool_Pipe_Min_Flows[c[2]])*3/4
        else:
            Samples[i,0] = Cool_Pipe_Max_Flows[c[2]]-Samples[i,0]*(Cool_Pipe_Max_Flows[c[2]]-Cool_Pipe_Min_Flows[c[2]])
        Samples[i,1] = Samples[i,1]*(Max_Supply_Temp_Cooling-Min_Supply_Temp_Cooling)+Min_Supply_Temp_Cooling
        Samples[i,2] = Samples[i,2]*(Cooling_Loop_Max_delta_T-Cooling_Loop_Min_delta_T)+Cooling_Loop_Min_delta_T
        Samples[i,3] = Samples[i,3]*(Max_Ground_Temperature-Min_Ground_Temperature)+Min_Ground_Temperature
          
        Variables[0] += [Samples[i,0]]
        Variables[1] += [Samples[i,1]]
        Variables[2] += [Samples[i,2]]
        Variables[3] += [Samples[i,3]]
      
        delta_L = Arc_Distances[(c[0],c[1])]/Num_Test_Points
          
        for n in range(Num_Test_Points):
            if n == 0:
                Delta_ST = Samples[i,1]-(delta_L*(Cool_Pipe_U[c[2]]*(Samples[i,1]-Samples[i,3])-Cool_Pipe_U2[c[2]]*Samples[i,2]))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
                Delta_RT = (Samples[i,1]+Samples[i,2])+delta_L*(Cool_Pipe_U[c[2]]*((Samples[i,1]+Samples[i,2])-Samples[i,3])+Cool_Pipe_U2[c[2]]*Samples[i,2])/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
            else:
                Delta_ST = Delta_ST-(delta_L*(Cool_Pipe_U[c[2]]*(Delta_ST-Samples[i,3])+Cool_Pipe_U2[c[2]]*(Delta_ST-Delta_RT)))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
                Delta_RT = Delta_RT+delta_L*(Cool_Pipe_U[c[2]]*(Delta_RT-Samples[i,3])-Cool_Pipe_U2[c[2]]*(Delta_ST-Delta_RT))/(Samples[i,0]*Density_Water/3600)/(Specific_Heat_Water)
 
        Heat_Losses += [Samples[i,0]*Density_Water/3600*Specific_Heat_Water*((Samples[i,1]-Delta_ST)+(Delta_RT-(Samples[i,1]+Samples[i,2])))/1000]        # kWh
          
    results = reg_m(Heat_Losses, Variables)
    
    Cool_Loss_Reg_Coefs[c] = [results.params[3], results.params[2], results.params[1], results.params[0], results.params[4]]

'''-----------------------------------------------------------------------------------------------'''
# Create the Parameters that Will Govern Mutations within the GA #
'''-----------------------------------------------------------------------------------------------'''
# Create the Low and High Sequence values that control mutation in optimization for the buildings
Low_Seq = []
High_Seq = []
for i in range(Num_Sites):
    Low_Seq += [Building_Min]
    High_Seq += [Num_Buildings]
    
# Create the Low and High Sequence values that control the location of the Central Plant
Low_Seq += [0]
High_Seq += [Num_Sites-1]

# Create the Low and High Sequence values that control mutation in optimization for CHP engines
Low_Seq += [Supply_Min]
High_Seq += [Num_Engines]

# Create the Low and High Sequence values that control mutation in optimization for the chillers
Low_Seq += [Supply_Min]
High_Seq += [Num_Chillers]

# Create the Low and High Sequence values that control mutation for solar
# Low_Seq += [1] # MODIFIED
# Low_Seq += [0] # MODIFIED
# High_Seq += [Num_Comm_Solar] # MODIFIED
# High_Seq += [Max_Solar] # MODIFIED

# Create the Low and High Sequence values that control mutation for loop temperature and thermal reset
Low_Seq += [Min_Supply_Temp_Heating]
Low_Seq += [Min_Summer_Heating_Reset_del_T]
Low_Seq += [Min_Supply_Temp_Cooling]
Low_Seq += [Min_Winter_Cooling_Reset_del_T]
High_Seq += [Max_Supply_Temp_Heating]
High_Seq += [Max_Summer_Heating_Reset_del_T]
High_Seq += [Max_Supply_Temp_Cooling]
High_Seq += [Max_Winter_Cooling_Reset_del_T]

'''-----------------------------------------------------------------------------------------------'''
# Set up the Results Matrix #
'''-----------------------------------------------------------------------------------------------'''
Num_Outputs = 24
Results = np.zeros(((Number_Generations+1)*Population_Size, len(High_Seq)+Num_Outputs+2)) # MODIFIED: +2 added to make up for removed solar variables
Vars_Plus_Output = len(Results[0,:])


'''-----------------------------------------------------------------------------------------------'''
# Generate dictionaries of supply engine, chiller, and solar types. #
'''-----------------------------------------------------------------------------------------------'''
CHP_Types = {}
CHP_Types[1] = CHP.EPA_CHP_Gas_Turbine_1
CHP_Types[2] = CHP.EPA_CHP_Microturbine_2
CHP_Types[3] = CHP.EPA_CHP_Reciprocating_Engine_3
CHP_Types[4] = CHP.EPA_CHP_Steam_Turbine_1
CHP_Types[5] = CHP.EPA_CHP_Fuel_Cell_4
CHP_Types[6] = CHP.CHP_Biomass_4


# CHP_Types[1] = CHP.EPA_CHP_Gas_Turbine_1
# CHP_Types[2] = CHP.EPA_CHP_Gas_Turbine_2
# CHP_Types[3] = CHP.EPA_CHP_Gas_Turbine_3
# CHP_Types[4] = CHP.EPA_CHP_Gas_Turbine_4
# CHP_Types[5] = CHP.EPA_CHP_Gas_Turbine_5
# CHP_Types[6] = CHP.EPA_CHP_Microturbine_1
# CHP_Types[7] = CHP.EPA_CHP_Microturbine_2
# CHP_Types[8] = CHP.EPA_CHP_Microturbine_3
# CHP_Types[9] = CHP.EPA_CHP_Reciprocating_Engine_1
# CHP_Types[10] = CHP.EPA_CHP_Reciprocating_Engine_2
# CHP_Types[11] = CHP.EPA_CHP_Reciprocating_Engine_3
# CHP_Types[12] = CHP.EPA_CHP_Reciprocating_Engine_4
# CHP_Types[13] = CHP.EPA_CHP_Reciprocating_Engine_5
# CHP_Types[14] = CHP.EPA_CHP_Steam_Turbine_1
# CHP_Types[15] = CHP.EPA_CHP_Steam_Turbine_2
# CHP_Types[16] = CHP.EPA_CHP_Steam_Turbine_3
# CHP_Types[17] = CHP.EPA_CHP_Fuel_Cell_1
# CHP_Types[18] = CHP.EPA_CHP_Fuel_Cell_2
# CHP_Types[19] = CHP.EPA_CHP_Fuel_Cell_3
# CHP_Types[20] = CHP.EPA_CHP_Fuel_Cell_4
# CHP_Types[21] = CHP.EPA_CHP_Fuel_Cell_5
# CHP_Types[22] = CHP.EPA_CHP_Fuel_Cell_6
# CHP_Types[23] = CHP.CHP_Biomass_1
# CHP_Types[24] = CHP.CHP_Biomass_2
# CHP_Types[25] = CHP.CHP_Biomass_3
# CHP_Types[26] = CHP.CHP_Biomass_4
# CHP_Types[27] = CHP.CHP_Biomass_5
# CHP_Types[28] = CHP.CHP_Biomass_6
# CHP_Types[29] = CHP.CHP_Biomass_7
# CHP_Types[30] = CHP.CHP_Biomass_8
# CHP_Types[31] = CHP.CHP_Biomass_9
# CHP_Types[32] = CHP.CHP_Biomass_10

Chiller_Types = {}
Chiller_Types[1] = EC.Electric_Chiller_3
Chiller_Types[2] = EC.Electric_Chiller_8
Chiller_Types[3] = AC.Absorption_Chiller_6



# Chiller_Types[1] = EC.Electric_Chiller_1
# Chiller_Types[2] = EC.Electric_Chiller_2
# Chiller_Types[3] = EC.Electric_Chiller_3
# Chiller_Types[4] = EC.Electric_Chiller_4
# Chiller_Types[5] = EC.Electric_Chiller_5
# Chiller_Types[6] = EC.Electric_Chiller_6
# Chiller_Types[7] = EC.Electric_Chiller_7
# Chiller_Types[8] = EC.Electric_Chiller_8
# Chiller_Types[9] = EC.Electric_Chiller_9
# Chiller_Types[10] = AC.Absorption_Chiller_1
# Chiller_Types[11] = AC.Absorption_Chiller_2
# Chiller_Types[12] = AC.Absorption_Chiller_3
# Chiller_Types[13] = AC.Absorption_Chiller_4
# Chiller_Types[14] = AC.Absorption_Chiller_5
# Chiller_Types[15] = AC.Absorption_Chiller_6
# Chiller_Types[16] = AC.Absorption_Chiller_7
# Chiller_Types[17] = AC.Absorption_Chiller_8


 # DEACTIVATED FIR NOW # MODIFIED
# Residential_Solar_Types = {} # TO NOTE: I haven't yet updated the SolarPanels.py or the method of calculating SP output in the code here, based on my RQ1 folder and updated codes there
# Residential_Solar_Types[1] = SP.Residential_Solar_1
# Residential_Solar_Types[2] = SP.Residential_Solar_2
# Residential_Solar_Types[3] = SP.Residential_Solar_3

# Commercial_Solar_Types = {} # DEACTIVATED FIR NOW # MODIFIED
# Commercial_Solar_Types[1] = SP.Commercial_Solar_1
# Commercial_Solar_Types[2] = SP.Commercial_Solar_2
# Commercial_Solar_Types[3] = SP.Commercial_Solar_3


'''-----------------------------------------------------------------------------------------------'''
# Create the Main Function to Be Called by the Genetic Algorithm #
'''-----------------------------------------------------------------------------------------------'''
def SupplyandDemandOptimization(Building_Var_Inputs, solver='gurobi'):
    Internal_Start = timeit.default_timer()

    '''-----------------------------------------------------------------------------------------------'''
    # Pull in and Rename the Input Variables #
    '''-----------------------------------------------------------------------------------------------'''
    Site_Vars = {}
    for i in range(Num_Sites):
        Site_Vars[i] = Building_Var_Inputs[i]
    #print(Building_Var_Inputs)
    Plant_Location_Var = int(Building_Var_Inputs[Num_Sites])
    Engine_Var = Building_Var_Inputs[Num_Sites+1]
    Chiller_Var = Building_Var_Inputs[Num_Sites+2]
    # Comm_Solar_Type_Var = 0 #Building_Var_Inputs[Num_Sites+3] # MODIFIED
    # Comm_Solar_Var = 0 #Building_Var_Inputs[Num_Sites+4] # MODIFIED
    Heat_Sup_Temp_Var = Building_Var_Inputs[Num_Sites+3]#5] # MODIFIED
    Heat_Sup_Temp_Reset_Var = Building_Var_Inputs[Num_Sites+4]#6] # MODIFIED
    Cool_Sup_Temp_Var = Building_Var_Inputs[Num_Sites+5]#7] # MODIFIED
    Cool_Sup_Temp_Reset_Var = Building_Var_Inputs[Num_Sites+6]#8] # MODIFIED
    
    '''-----------------------------------------------------------------------------------------------'''
    # Rectify Plant Location with Other Site Locations #
    '''-----------------------------------------------------------------------------------------------'''
    # The process here is to identify what is located at the node where the central plant needs to be 
    # installed. Store it, and then relocate it to the first vacant site on the property. If there is no vacant
    # site, the building will just be removed from the development.
    if Site_Vars[Plant_Location_Var] != 0:
        Relocation = Site_Vars[Plant_Location_Var]
        for i in range(Num_Sites):
            if Site_Vars[i] == 0:
                Site_Vars[i] = Relocation
                break
    
    Site_Vars[Plant_Location_Var] = Num_Buildings+1               # Assign a way to find the plant later
    
    '''-----------------------------------------------------------------------------------------------'''
    # Generate Test Electricity, Heating, and Cooling #
    '''-----------------------------------------------------------------------------------------------'''
    # Use the Site_Vars dictionary and the dictionary of demands to create an aggregate function of demand
    Test_Aggregate_Demand = 0
    for i in range(Num_Sites):
        Building_Type = Site_Vars[i]
        if Building_Type != 0 and Building_Type != Num_Buildings+1:
            Test_Aggregate_Demand += np.column_stack((Init_Demand_Types[Building_Type][:,0], Init_Demand_Types[Building_Type][:,1], Init_Demand_Types[Building_Type][:,2]))


    # Handling the trivial case where no buildings exist in the mix: make it utterly undesirable
    if np.isscalar(Test_Aggregate_Demand):
        Run_Result = np.zeros((1,Vars_Plus_Output))
        # Add the variables first
        for i in range(Num_Sites):
            Run_Result[0][i] = Site_Vars[i]
        Run_Result[0][Num_Sites] = Plant_Location_Var
        Run_Result[0][Num_Sites+1] = Engine_Var
        Run_Result[0][Num_Sites+2] = Chiller_Var
        Run_Result[0][Num_Sites+3] = 0
        Run_Result[0][Num_Sites+4] = 0
        Run_Result[0][Num_Sites+5] = Heat_Sup_Temp_Var
        Run_Result[0][Num_Sites+6] = Heat_Sup_Temp_Reset_Var
        Run_Result[0][Num_Sites+7] = Cool_Sup_Temp_Var
        Run_Result[0][Num_Sites+8] = Cool_Sup_Temp_Reset_Var
        return ((float('inf'), float('inf'), -float('inf'), ), (Max_GFA-0, Max_FAR-0, Max_Average_Height-0, Max_Res-0, Max_Off-0, Max_Ret-0, Max_Sup-0, Max_Rest-0, Max_Edu-0, Max_Med-0, Max_Lod-0, Max_Ind-0, 0-Min_Res, 0-Min_Off, 0-Min_Ret, 0-Min_Sup, 0-Min_Rest, 0-Min_Edu, 0-Min_Med, 0-Min_Lod, 0-Min_Ind, ), Run_Result)

    


    # Now use the aggregate function of demand to size the heating and cooling units and find their capital costs
    Test_Electricity = Test_Aggregate_Demand[:,0]   # Vector of electricity values in kWh
    Test_Heat = Test_Aggregate_Demand[:,1]          # Vector of heat values in kWh
    if Chiller_Var < 10:
        Test_Electricity += Test_Aggregate_Demand[:,2]/Chiller_Nominal_COP[Chiller_Var]     # Vector of electricity values in kWh
    else:
        Test_Heat += Test_Aggregate_Demand[:,2]/Chiller_Nominal_COP[Chiller_Var]            # Vector of heat values in kWn
    Engine_Units = np.ceil(max(max(Test_Electricity), max(Test_Heat)*Power_to_Heat[Engine_Var])/Max_Unit_Size[Engine_Var])  # Round number
    Engine_Units_Cap_Cost = CHP_Capital_Cost[Engine_Var]*Engine_Units   # USD
    Chiller_Units_Cap_Cost = max(Test_Aggregate_Demand[:,2])/tons_to_kW*Chiller_Capital_Cost[Chiller_Var]   # USD

    # Now find the per kWh capital cost allocation of the CHP and chiller costs
    Total_Electricity = sum(Test_Electricity)*Project_Life          # kWh
    Total_Heat = sum(Test_Heat)*Project_Life                        # kWh
    Total_Cooling = sum(Test_Aggregate_Demand[:,2])*Project_Life    # kWh

    Unit_CapEx_CHP = Engine_Units_Cap_Cost/(Total_Electricity+Total_Heat)   # USD/kWh
    Unit_CapEx_Chiller = Chiller_Units_Cap_Cost/Total_Cooling               # USD/kWh

    # Now calculate the cost per unit of electricity, heating, and cooling for the test scenarios
    # First identify the fuel type cost
    if CHP_Fuel_Type[Engine_Var] == 1:
        Fuel_Cost = Natural_Gas_Cost        # USD/Btu
    elif CHP_Fuel_Type[Engine_Var] == 2:
        Fuel_Cost = Hydrogen_Cost           # USD/Btu
    else:
        Fuel_Cost = Biomass_Cost            # USD/Btu

    # Next find the total variable cost for the CHP unit, the total resource produced (in kWh), since the costs are set per kWh electricity, and then assign values
    CHP_Total_Var_Cost= CHP_Heat_Rate[Engine_Var]*Fuel_Cost+CHP_Variable_Cost[Engine_Var]+Unit_CapEx_CHP         # USD/kWh
    CHP_Total_Resource = Power_to_Heat[Engine_Var]+1                                                             # Unitless
    Unit_Electricity_Cost = Power_to_Heat[Engine_Var]/CHP_Total_Resource*CHP_Total_Var_Cost                      # USD/kWh
    Unit_Heat_Cost = 1/CHP_Total_Resource*CHP_Total_Var_Cost                                                     # USD/kWh

    # Now assign a cost per kWh of cooling
    if Chiller_Var > 9:
        Unit_Cooling_Cost = Unit_Heat_Cost/Chiller_Nominal_COP[Chiller_Var]+Unit_CapEx_Chiller              # USD/kWh
    else:
        Unit_Cooling_Cost = Unit_Electricity_Cost/Chiller_Nominal_COP[Chiller_Var]+Unit_CapEx_Chiller       # USD/kWh

    '''-----------------------------------------------------------------------------------------------'''
    # Prep Additional Inputs for Pipe Layout Optimization. #
    '''-----------------------------------------------------------------------------------------------'''
    # Find the test return temperature and the temperature difference
    Test_Heat_Return_Temp = max(Heat_Sup_Temp_Var-Heating_Loop_Max_delta_T, Min_Secondary_Temp_Heating+Heat_Exchanger_Approach, Heating_Setpoint+Fan_Coil_Approach+Heat_Exchanger_Approach)
    Test_Cool_Return_Temp = min(Cool_Sup_Temp_Var+Cooling_Loop_Max_delta_T, Max_Secondary_Temp_Cooling-Heat_Exchanger_Approach, Cooling_Setpoint-Fan_Coil_Approach-Heat_Exchanger_Approach)
    Test_Heat_Delta_T = Heat_Sup_Temp_Var - Test_Heat_Return_Temp
    Test_Cool_Delta_T = Test_Cool_Return_Temp - Cool_Sup_Temp_Var
    
    '''-----------------------------------------------------------------------------------------------'''
    # Run the Pipe Layout Optimization Using PuLP as the Optimization Package. #
    '''-----------------------------------------------------------------------------------------------'''
    # First create dictionaries of the maximum and annual heating and cooling demand at each of the sites and single values for the peak and total heating demand and peak cooling demand.
    # Also create dictionaries of the maximum and total heating and cooling supply at each of the sites. For now everything except the single central plant will have a zero supply.   
    Max_Site_Heats = {}
    Max_Site_Cools = {}
    Total_Peak_Heat = 0
    Total_Peak_Cool = 0
    Annual_Site_Heats = {}
    Annual_Site_Cools = {}
    Total_Annual_Heat = 0
    Total_Annual_Cool = 0
    
    Site_Vars = {key:int(val) for key, val in Site_Vars.items()} ## MODIFIED
    global Nodes
    for n in Nodes:
        if Site_Vars[n] != 0 and Site_Vars[n] != Num_Buildings+1:
            Max_Site_Heats[n] = np.ceil(max(Heating_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)       # m3/hr
            Max_Site_Cools[n] = np.ceil(max(Cooling_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)       # m3/hr
            Total_Peak_Heat += np.ceil(max(Heating_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)        # m3/hr
            Total_Peak_Cool += np.ceil(max(Cooling_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)        # m3/hr
            Annual_Site_Heats[n] = np.ceil(sum(Mod_Heating_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)    # m3/yr
            Annual_Site_Cools[n] = np.ceil(sum(Mod_Cooling_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)    # m3/yr
            Total_Annual_Heat += np.ceil(sum(Mod_Heating_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)      # m3/yr
            Total_Annual_Cool += np.ceil(sum(Mod_Cooling_Input[:,Site_Vars[n]-1])*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)      # m3/yr
        else:
            Max_Site_Heats[n] = 0
            Max_Site_Cools[n] = 0
            Annual_Site_Heats[n] = 0
            Annual_Site_Cools[n] = 0

    # Now create dictionaries of the maximum and total heating and cooling supply at each of the sites. For now everything except the single central plant will have a zero supply.
    # This could be changed in the future to accommodate multiple supply locations (polygeneration).
    Max_Site_Heat_Supplies = {}
    Max_Site_Cool_Supplies = {}
    Annual_Site_Heat_Supplies = {}
    Annual_Site_Cool_Supplies = {}
    for n in Nodes:
        if Site_Vars[n] == Num_Buildings+1:
            Max_Site_Heat_Supplies[n] = Total_Peak_Heat                      # m3/hr
            Max_Site_Cool_Supplies[n] = Total_Peak_Cool                      # m3/hr
            Annual_Site_Heat_Supplies[n] = Total_Annual_Heat                 # m3/hr
            Annual_Site_Cool_Supplies[n] = Total_Annual_Cool                 # m3/hr
        else:
            Max_Site_Heat_Supplies[n] = 0
            Max_Site_Cool_Supplies[n] = 0
            Annual_Site_Heat_Supplies[n] = 0
            Annual_Site_Cool_Supplies[n] = 0        
    
    # Now instantiate the problem using the PuLP package
    IntInfraLocOpt = LpProblem("Integrated_Infrastructure_Pipe_Optimization", LpMinimize)
    
    ### Now set up the variables for optimization. ###
    
    # Create the binary variables "y" that govern the existence of heating pipes to be built for each connection. "y" is indexed over i, j, and k where i and j are the dimensions 
    # of the Connections matrix and k is the number of possible heating pipes.
    Heat_Pipe_Y_Vars = LpVariable.dicts("Y",Heat_Arcs,0,1,LpInteger)
    
    # Create the binary variables "w" that govern the existence of cooling pipes to be built for each connection. "w" is indexed over i, j, and l where i and j are the dimensions 
    # of the Connections matrix and l is the number of possible cooling pipes.
    Cool_Pipe_W_Vars = LpVariable.dicts("W",Cool_Arcs,0,1,LpInteger)
    
    # Create the continuous variables "x" that govern the peak flow that will go through each heating pipe. "x" is indexed over i, j, and k where i and j are the dimensions of the 
    # Connections matrix and k is the number of possible heating pipes.
    Heat_Pipe_X_Vars = LpVariable.dicts("X",Heat_Arcs,0,None,LpContinuous)
    
    # Create the continuous variables "z" that govern the peak flow that will go through each cooling pipe. "z" is indexed over i, j, and l where i and j are the dimensions of the 
    # Connections matrix and l is the number of possible cooling pipes.
    Cool_Pipe_Z_Vars = LpVariable.dicts("Z",Cool_Arcs,0,None,LpContinuous)
    
    # Create the continuous variables "u" that govern the annual flow that will go through each heating pipe. "u" is indexed over i, j, and k where i and j are the dimensions of the 
    # Connections matrix and k is the number of possible heating pipes.
    Heat_Pipe_U_Vars = LpVariable.dicts("U",Heat_Arcs,0,None,LpContinuous)
    
    # Create the continuous variables "s" that govern the annual flow that will go through each cooling pipe. "s" is indexed over i, j, and l where i and j are the dimensions of the 
    # Connections matrix and l is the number of possible cooling pipes.
    Cool_Pipe_S_Vars = LpVariable.dicts("S",Cool_Arcs,0,None,LpContinuous)
    
    # Create the binary variables "v" that govern the existence or nonexistence of a trench between nodes i and j. This is triggered if a trench is chosen. i and j are the dimensions
    # of the Connections matrix.
    Trench_V_Vars = LpVariable.dicts("V",Trench_Arcs,0,1,LpInteger)
        
    ### Now add the objective function. ### ## MODIFIED: coefficients of 2 were missing from the second terms in heat pump and cooling pump loss expressions + the losses for the cooling had a -8760 coeff, which was changed to 8760
    IntInfraLocOpt += lpSum([2*Heat_Pump_Reg_Val_1[h]*Unit_Electricity_Cost*Heat_Pipe_X_Vars[h]+\
                             2*Heat_Pipe_Y_Vars[h]*Heat_Pump_Reg_Val_0[h]*Unit_Electricity_Cost for h in Heat_Arcs]) + \
                      lpSum([Heat_Arc_Capex[h]*Heat_Pipe_Y_Vars[h] for h in Heat_Arcs]) + \
                      lpSum([2*Project_Life*Heat_Pump_Kinked_Reg_Val[h]*Unit_Electricity_Cost*Heat_Pipe_U_Vars[h] for h in Heat_Arcs]) + \
                      lpSum([8760*Project_Life*Unit_Heat_Cost*(Heat_Loss_Reg_Coefs[h][0]*Heat_Pipe_U_Vars[h]/8760 +\
                                                               Heat_Pipe_Y_Vars[h]*Heat_Loss_Reg_Coefs[h][1]*Heat_Sup_Temp_Var +\
                                                                   Heat_Pipe_Y_Vars[h]*Heat_Loss_Reg_Coefs[h][2]*Test_Heat_Delta_T +\
                                                               Heat_Pipe_Y_Vars[h]*Heat_Loss_Reg_Coefs[h][3]*Average_Ground_Temperature +\
                                                                   Heat_Pipe_Y_Vars[h]*Heat_Loss_Reg_Coefs[h][4])
                             for h in Heat_Arcs]) + \
                      lpSum([2*Cool_Pump_Reg_Val_1[c]*Unit_Electricity_Cost*Cool_Pipe_Z_Vars[c] +\
                             2*Cool_Pipe_W_Vars[c]*Cool_Pump_Reg_Val_0[c]*Unit_Electricity_Cost for c in Cool_Arcs]) + \
                      lpSum([Cool_Arc_Capex[c]*Cool_Pipe_W_Vars[c] for c in Cool_Arcs]) + \
                      lpSum([2*Project_Life*Cool_Pump_Kinked_Reg_Val[c]*Unit_Electricity_Cost*Cool_Pipe_S_Vars[c] for c in Cool_Arcs]) + \
                      lpSum([8760*Project_Life*Unit_Cooling_Cost*(Cool_Loss_Reg_Coefs[c][0]*Cool_Pipe_S_Vars[c]/8760 +\
                                                                   Cool_Pipe_W_Vars[c]*Cool_Loss_Reg_Coefs[c][1]*Cool_Sup_Temp_Var +\
                                                                       Cool_Pipe_W_Vars[c]*Cool_Loss_Reg_Coefs[c][2]*Test_Cool_Delta_T +\
                                                                    Cool_Pipe_W_Vars[c]*Cool_Loss_Reg_Coefs[c][3]*Average_Ground_Temperature +\
                                                                        Cool_Pipe_W_Vars[c]*Cool_Loss_Reg_Coefs[c][4])
                             for c in Cool_Arcs]) + \
                      lpSum([Total_Trench_Costs[t]*Trench_V_Vars[t] for t in Trench_Arcs])
    
    ### Now add the constraints for the problem. ###
    
    ### Uniqueness Constraints ###
    # Add a constraint that defines that only one heating pipe can be built for each connection i to j.
    for t in Trench_Arcs:
        IntInfraLocOpt += lpSum([Heat_Pipe_Y_Vars[(t[0],t[1],k)] for k in Heat_Pipe_Numbers]) <= 1, "Heat Pipe %s to %s Uniqueness" %t
        #pdb.set_trace()
        
    # Add a constraint that defines that only one cooling pipe can be built for each connection i to j.
    for t in Trench_Arcs:
        IntInfraLocOpt += lpSum([Cool_Pipe_W_Vars[(t[0],t[1],l)] for l in Cool_Pipe_Numbers]) <= 1, "Cool Pipe %s to %s Uniqueness" %t
                             
    ### Trench Constraint ###
    # Add a constraint that governs that a trench must be built if there is any pipe between i and j.
    for t in Trench_Arcs:
        IntInfraLocOpt += lpSum([Heat_Pipe_Y_Vars[(t[0],t[1],k)] for k in Heat_Pipe_Numbers]) + lpSum([Cool_Pipe_W_Vars[(t[0],t[1],l)] for l in Cool_Pipe_Numbers]) + lpSum([Heat_Pipe_Y_Vars[(t[1],t[0],k)] for k in Heat_Pipe_Numbers]) + lpSum([Cool_Pipe_W_Vars[(t[1],t[0],l)] for l in Cool_Pipe_Numbers]) <= 4*Trench_V_Vars[t], "Trench Pipe %s to %s Existence" %t
               
    ### Max Flow Constraints ###
    # Add a constraint that governs max flow in the peak heating pipe case
    for h in Heat_Arcs:
        IntInfraLocOpt += Heat_Pipe_X_Vars[h] <= Heat_Arc_Max_Flows[h]*Heat_Pipe_Y_Vars[h], "Heat Pipe Peak Max Flow %s to %s in %s" %h
        
    # Add a constraint that governs max flow in the peak cooling pipe case
    for c in Cool_Arcs:
        IntInfraLocOpt += Cool_Pipe_Z_Vars[c] <= Cool_Arc_Max_Flows[c]*Cool_Pipe_W_Vars[c], "Cool Pipe Peak Max Flow %s to %s in %s" %c
                   
    # Add a constraint that governs max flow in the annual heating pipe case
    for h in Heat_Arcs:
        IntInfraLocOpt += Heat_Pipe_U_Vars[h] <= 8760*Heat_Arc_Max_Flows[h]*Heat_Pipe_Y_Vars[h], "Heat Pipe Annual Max Flow %s to %s in %s" %h
                    
    # Add a constraint that governs max flow in the annual cooling pipe case
    for c in Cool_Arcs:
        IntInfraLocOpt += Cool_Pipe_S_Vars[c] <= 8760*Cool_Arc_Max_Flows[c]*Cool_Pipe_W_Vars[c], "Cool Pipe Annual Max Flow %s to %s in %s" %c
                   
    ### Min Flow Constraints ###
    # Add a constraint that governs min flow in the peak heating pipe case
    for h in Heat_Arcs:
        IntInfraLocOpt += Heat_Pipe_X_Vars[h] >= Heat_Arc_Min_Flows[h]*Heat_Pipe_Y_Vars[h], "Heat Pipe Peak Min Flow %s to %s in %s" %h
       
    # Add a constraint that governs min flow in the peak cooling pipe case
    for c in Cool_Arcs:
        IntInfraLocOpt += Cool_Pipe_Z_Vars[c] >= Cool_Arc_Min_Flows[c]*Cool_Pipe_W_Vars[c], "Cool Pipe Peak Min Flow %s to %s in %s" %c
                   
    ### Flow Balance Constraints ###
    # Add a constraint that governs the flow balance for peak heating at each node
    for n in Nodes:
        IntInfraLocOpt += (Max_Site_Heat_Supplies[n] + lpSum([Heat_Pipe_X_Vars[(i,j,k)] for (i,j,k) in Heat_Arcs if j == n]) == Max_Site_Heats[n] + lpSum([Heat_Pipe_X_Vars[(i,j,k)] for (i,j,k) in Heat_Arcs if i == n])), "Heat Pipe Peak Node %s Flow Balance" %n
         
    # Add a constraint that governs the flow balance for peak cooling at each node
    for n in Nodes:
        IntInfraLocOpt += (Max_Site_Cool_Supplies[n] + lpSum([Cool_Pipe_Z_Vars[(i,j,l)] for (i,j,l) in Cool_Arcs if j == n]) == Max_Site_Cools[n] + lpSum([Cool_Pipe_Z_Vars[(i,j,l)] for (i,j,l) in Cool_Arcs if i == n])), "Cool Pipe Peak Node %s Flow Balance" %n
         
    # Add a constraint that governs the flow balance for annual heating at each node
    for n in Nodes:
        IntInfraLocOpt += (Annual_Site_Heat_Supplies[n] + lpSum([Heat_Pipe_U_Vars[(i,j,k)] for (i,j,k) in Heat_Arcs if j == n]) == Annual_Site_Heats[n] + lpSum([Heat_Pipe_U_Vars[(i,j,k)] for (i,j,k) in Heat_Arcs if i == n])), "Heat Pipe Annual Node %s Flow Balance" %n
     
    # Add a constraint that governs the flow balance for annual cooling at each node
    for n in Nodes:
        IntInfraLocOpt += (Annual_Site_Cool_Supplies[n] + lpSum([Cool_Pipe_S_Vars[(i,j,l)] for (i,j,l) in Cool_Arcs if j == n]) == Annual_Site_Cools[n] + lpSum([Cool_Pipe_S_Vars[(i,j,l)] for (i,j,l) in Cool_Arcs if i == n])), "Cool Pipe Annual Node %s Flow Balance" %n
           
    ### Now solve the optimization ###
    IntInfraLocOpt.writeLP("IntegratedInfrastructureLocationOptimization.lp")
    if solver=='gurobi':
        IntInfraLocOpt.solve(GUROBI_CMD(msg=False)) ## MODIFIED
    else:
        IntInfraLocOpt.solve()
    
    # Check if the solution is infeasible, and output it as a very undesirable solution
    if IntInfraLocOpt.status != LpStatusOptimal:
        # Infeasible case handling
        Run_Result = np.zeros((1,Vars_Plus_Output))
        # Add the variables first
        for i in range(Num_Sites):
            Run_Result[0][i] = Site_Vars[i]
        Run_Result[0][Num_Sites] = Plant_Location_Var
        Run_Result[0][Num_Sites+1] = Engine_Var
        Run_Result[0][Num_Sites+2] = Chiller_Var
        Run_Result[0][Num_Sites+3] = 0
        Run_Result[0][Num_Sites+4] = 0
        Run_Result[0][Num_Sites+5] = Heat_Sup_Temp_Var
        Run_Result[0][Num_Sites+6] = Heat_Sup_Temp_Reset_Var
        Run_Result[0][Num_Sites+7] = Cool_Sup_Temp_Var
        Run_Result[0][Num_Sites+8] = Cool_Sup_Temp_Reset_Var
        # Making the individual as undesirable as possible to avoid its being sustained in the solutions
        return ((float('inf'), float('inf'), -float('inf'), ), (Max_GFA-0, Max_FAR-0, Max_Average_Height-0, Max_Res-0, Max_Off-0, Max_Ret-0, Max_Sup-0, Max_Rest-0, Max_Edu-0, Max_Med-0, Max_Lod-0, Max_Ind-0, 0-Min_Res, 0-Min_Off, 0-Min_Ret, 0-Min_Sup, 0-Min_Rest, 0-Min_Edu, 0-Min_Med, 0-Min_Lod, 0-Min_Ind, ), Run_Result)


    
    ### Now grab the outputs of the optimization ###
    Optimal_Heat_Connections = np.zeros((Num_Sites,Num_Sites))          # Create an array that will hold the final pipes for heating
    Optimal_Cool_Connections = np.zeros((Num_Sites,Num_Sites))          # Create an array that will hold the final pipes for cooling
    Optimal_Heat_Binary = np.zeros((Num_Sites,Num_Sites))               # Create an array that will hold a 1 or 0 to specify if a pipe was built for heating
    Optimal_Cool_Binary = np.zeros((Num_Sites,Num_Sites))               # Create an array that will hold a 1 or 0 to specify if a pipe was built for cooling
    
    Heat_Peak_Flows = np.zeros((Num_Sites,Num_Sites))
    Heat_Annual_Flows = np.zeros((Num_Sites,Num_Sites))
    Cool_Peak_Flows = np.zeros((Num_Sites,Num_Sites))
    Cool_Annual_Flows = np.zeros((Num_Sites,Num_Sites))
     
    # Now assign the final connections. Here, if a pipe exists along a connection, a number replaces the zero. The number corresponds to the index of
    # the heating or cooling pipe in the Heat_Pipe_Names and Cool_Pipe_Names dictionaries. The index starts at 1 to avoid confusion.
    for h in Heat_Arcs:
        if Heat_Pipe_Y_Vars[h].value() > 0.5:
            Optimal_Heat_Binary[h[0],h[1]] = 1
            Optimal_Heat_Connections[h[0],h[1]] = h[2]
            Heat_Peak_Flows[h[0],h[1]] = Heat_Pipe_X_Vars[h].value()
            Heat_Annual_Flows[h[0],h[1]] = Heat_Pipe_U_Vars[h].value()     
    for c in Cool_Arcs:
        if Cool_Pipe_W_Vars[c].value() > 0.5:
            Optimal_Cool_Binary[c[0],c[1]] = 1
            Optimal_Cool_Connections[c[0],c[1]] = c[2]
            Cool_Peak_Flows[c[0],c[1]] = Cool_Pipe_Z_Vars[c].value()
            Cool_Annual_Flows[c[0],c[1]] = Cool_Pipe_S_Vars[c].value() 
                     
    '''-----------------------------------------------------------------------------------------------'''
    # Now calculate the hourly pumping requirement and energy loss given that we have the layout. #
    '''-----------------------------------------------------------------------------------------------'''
    # To find the hourly pumping and energy loss is an iterative process based on first finding the pressures at all nodes to 
    # derive the flows and then calculating the temperatures. These are then used to inform the flows, etc. Temperatures are
    # calculated from a time-shifted start temperature emblematic of the time it takes for the water to move through the pipes
    # especially at low flow rates. Here, the number of iterations is set based on testing the balance of more vs. fewer 
    # iterations. More requires more time, and achieves diminsing accuracy.
    
    # First set up and populate arrays that will hold the volume of water in each of the optimial pipes.
    Optimal_Heat_Pipe_Vols = np.zeros((Num_Sites, Num_Sites))
    Optimal_Cool_Pipe_Vols = np.zeros((Num_Sites, Num_Sites))
    
    Nodes = [int(i) for i in Nodes]
    for i in Nodes:
        for j in Nodes:
            if Optimal_Heat_Connections[i,j] >= 0.5:
                Optimal_Heat_Pipe_Vols[i,j] = PD.PairedDistance(Site_Coordinates[i,:], Site_Coordinates[j,:])*mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,j]]/2)**2
            if Optimal_Cool_Connections[i,j] >= 0.5:
                Optimal_Cool_Pipe_Vols[i,j] = PD.PairedDistance(Site_Coordinates[i,:], Site_Coordinates[j,:])*mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,j]]/2)**2

    '''-----------------------------------------------------------------------------------------------'''
    # Create Branches for Optimal Heat and Cool Piping Results #
    '''-----------------------------------------------------------------------------------------------'''
    # Find the branches that exit the power plant and create lists in a dictionary so all of the nodes they hit can be recounted later
    Optimal_Heat_Branches = {}
    Optimal_Heat_Branch_Num = 1

    for node in range(Num_Sites):
        if Optimal_Heat_Binary[Plant_Location_Var-1,node] > 0.5:
            Optimal_Heat_Branches[Optimal_Heat_Branch_Num] = [Plant_Location_Var-1, node]
            Optimal_Heat_Branch_Num += 1
            
    # Populate the branches with all of the nodes that they touch
    branches_to_test = Queue()
    for bran in range(len(Optimal_Heat_Branches.keys())):
        branch = bran+1
        branches_to_test.put(branch)
    while not branches_to_test.empty():
        branch = branches_to_test.get(block = False)
        Terminated = False
        QuitMerge = False
        while Terminated == False:
            Current_End_Node = Optimal_Heat_Branches[branch][-1]
            if Current_End_Node in Optimal_Heat_Branches[branch][:-1]:
                Optimal_Heat_Branches[branch] += ['Looped']
                Terminated = True
                continue
            if sum(Optimal_Heat_Binary[:,Current_End_Node]) > 1.5:
                Optimal_Heat_Branches[branch] += ['Merged']
                for b in range(branch-1):
                    if Current_End_Node in Optimal_Heat_Branches[b+1]:
                        Terminated = True
                        QuitMerge = True
                        break
                if QuitMerge == True:
                    QuitMerge = False
                    continue
            if sum(Optimal_Heat_Binary[Current_End_Node,:]) == 0:
                Terminated = True
                continue
            elif sum(Optimal_Heat_Binary[Current_End_Node,:]) == 1:
                for Test_Node in range(Num_Sites):
                    if Optimal_Heat_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Heat_Branches[branch] += [Test_Node]
                        continue
            else:
                countdown = sum(Optimal_Heat_Binary[Current_End_Node,:])
                for Test_Node in range(Num_Sites):
                    if countdown == sum(Optimal_Heat_Binary[Current_End_Node,:]) and countdown > 0 and Optimal_Heat_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Heat_Branches[branch] += ['Split', Test_Node]
                        countdown -= 1
                    elif countdown > 0 and Optimal_Heat_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Heat_Branches[Optimal_Heat_Branch_Num] = Optimal_Heat_Branches[branch][:-2]+['Split', Test_Node]
                        branches_to_test.put(Optimal_Heat_Branch_Num)
                        Optimal_Heat_Branch_Num += 1
                        countdown -= 1
                    elif countdown == 0:
                        continue

    # Repeat for cooling
    # Find the branches that exit the power plant and create lists in a dictionary so all of the nodes they hit can be recounted later
    Optimal_Cool_Branches = {}
    Optimal_Cool_Branch_Num = 1

    for node in range(Num_Sites):
        if Optimal_Cool_Binary[Plant_Location_Var-1,node] > 0.5:
            Optimal_Cool_Branches[Optimal_Cool_Branch_Num] = [Plant_Location_Var-1, node]
            Optimal_Cool_Branch_Num += 1
            
    # Populate the branches with all of the nodes that they touch
    branches_to_test = Queue()
    for bran in range(len(Optimal_Cool_Branches.keys())):
        branch = bran+1
        branches_to_test.put(branch)
    while not branches_to_test.empty():
        branch = branches_to_test.get(block = False)
        Terminated = False
        QuitMerge = False
        while Terminated == False:
            Current_End_Node = Optimal_Cool_Branches[branch][-1]
            if Current_End_Node in Optimal_Cool_Branches[branch][:-1]:
                Optimal_Cool_Branches[branch] += ['Looped']
                Terminated = True
                continue
            if sum(Optimal_Cool_Binary[:,Current_End_Node]) > 1.5:
                Optimal_Cool_Branches[branch] += ['Merged']
                for b in range(branch-1):
                    if Current_End_Node in Optimal_Cool_Branches[b+1]:
                        Terminated = True
                        QuitMerge = True
                        break
                if QuitMerge == True:
                    QuitMerge = False
                    continue
            if sum(Optimal_Cool_Binary[Current_End_Node,:]) == 0:
                Terminated = True
                continue
            elif sum(Optimal_Cool_Binary[Current_End_Node,:]) == 1:
                for Test_Node in range(Num_Sites):
                    if Optimal_Cool_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Cool_Branches[branch] += [Test_Node]
                        continue
            else:
                countdown = sum(Optimal_Cool_Binary[Current_End_Node,:])
                for Test_Node in range(Num_Sites):
                    if countdown == sum(Optimal_Cool_Binary[Current_End_Node,:]) and countdown > 0 and Optimal_Cool_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Cool_Branches[branch] += ['Split', Test_Node]
                        countdown -= 1
                    elif countdown > 0 and Optimal_Cool_Binary[Current_End_Node, Test_Node] == 1:
                        Optimal_Cool_Branches[Optimal_Cool_Branch_Num] = Optimal_Cool_Branches[branch][:-2]+['Split', Test_Node]
                        branches_to_test.put(Optimal_Cool_Branch_Num)
                        Optimal_Cool_Branch_Num += 1
                        countdown -= 1
                    elif countdown == 0:
                        continue

    '''-----------------------------------------------------------------------------------------------'''
    # Instantiate Helper Functions #
    '''-----------------------------------------------------------------------------------------------'''
    def CalcHeatLoss(BHAFlow, Pipe_Vol):
        x = np.where(BHAFlow<Pipe_Vol, Pipe_Vol, BHAFlow)
        return x

    '''-----------------------------------------------------------------------------------------------'''
    # Calculate Heating and Pumping Losses #
    '''-----------------------------------------------------------------------------------------------'''
                    
    # Calculate the heat loss and pumping loss iteratively
    # Set up the arrays that will hold node temperatures, arc losses, and arc flows for all hours
    Optimal_Heat_Node_Pressures = np.zeros((8760,Num_Sites))
    Optimal_Heat_Arc_Supply_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Heat_Arc_Return_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Heat_Arc_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Heat_Arc_Flows = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Heat_Hourly_Demand = np.zeros((8760,Num_Sites))
    Optimal_Heat_Hourly_Supply_Temps = np.ones((8760,Num_Sites))*Heat_Sup_Temp_Var
    for i in range(8760):
        if i >= Summer_Thermal_Reset_Hour and i < Winter_Thermal_Reset_Hour:
            Optimal_Heat_Hourly_Supply_Temps[i,:] -= Heat_Sup_Temp_Reset_Var
    Optimal_Heat_Hourly_Return_Temps = np.ones((8760,Num_Sites))*Test_Heat_Return_Temp
    Optimal_Heat_Hourly_Building_Return_Temps = np.zeros((8760,Num_Sites))
    for num in range(len(Site_Vars)):
        if Site_Vars[num] != Num_Buildings+1 and Site_Vars[num] != 0:
            Optimal_Heat_Hourly_Demand[:,num] = Heating_Input[:,Site_Vars[num]-1]
    Optimal_Useful_Heat_Flow = Optimal_Heat_Hourly_Demand*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600
    Optimal_Heat_Flow_Required = np.zeros((8760,Num_Sites))
    for i in Nodes:
        if Site_Vars[i] != Num_Buildings+1 and Site_Vars[i] != 0:
            Minimum_Flow_Provided = Min_Flow_Required*max(Heating_Input[:,Site_Vars[i]-1]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)
            Optimal_Heat_Flow_Required[:,i] = np.where(Optimal_Useful_Heat_Flow[:,i] < Minimum_Flow_Provided, Minimum_Flow_Provided, Optimal_Useful_Heat_Flow[:,i])
    Optimal_Heat_Building_Flows = copy.deepcopy(Optimal_Heat_Flow_Required)
    Optimal_Heat_Flow_Gal_per_Min = Optimal_Heat_Flow_Required/3600*Metric_to_Imperial_Flow
    Optimal_Heat_Plant_Pressures  = np.zeros((8760,Num_Sites))
    Optimal_Heat_Plant_Flows = np.zeros((8760,Num_Sites))
    # Optimal_Heat_Arc_Return_Pipe_Temps = np.zeros((8760,Num_Sites,Num_Sites)) # OFFLINED # MODIFIED

    for iter in range(Num_Iterations):
        Looped = False
        Split = False
        Merged = False
        LoopMerge = False
        MergedHold = False
        MergedLoop = False
        if iter > 0:
            Max_Building_Temp = max(Min_Secondary_Temp_Heating+Heat_Exchanger_Approach, Heating_Setpoint+Fan_Coil_Approach+Heat_Exchanger_Approach)
            Test_Temps = Optimal_Heat_Hourly_Supply_Temps-Heating_Loop_Max_delta_T
            Optimal_Heat_Hourly_Building_Return_Temps = np.where(Test_Temps < Max_Building_Temp, Max_Building_Temp, Test_Temps)
            Optimal_Useful_Heat_Flow = np.nan_to_num(Optimal_Heat_Hourly_Demand*1000/Specific_Heat_Water/(Optimal_Heat_Hourly_Supply_Temps-Optimal_Heat_Hourly_Building_Return_Temps)/Density_Water*3600)
            for i in Nodes:
                if Site_Vars[i] != Num_Buildings+1 and Site_Vars[i] != 0:
                    Minimum_Flow_Provided = Min_Flow_Required*max(Heating_Input[:,Site_Vars[i]-1]*1000/Specific_Heat_Water/(Optimal_Heat_Hourly_Supply_Temps[:,i]-Optimal_Heat_Hourly_Building_Return_Temps[:,i])/Density_Water*3600)
                    Optimal_Heat_Flow_Required[:,i] = np.where(Optimal_Useful_Heat_Flow[:,i] < Minimum_Flow_Provided, Minimum_Flow_Provided, Optimal_Useful_Heat_Flow[:,i])
            Optimal_Heat_Building_Flows = copy.deepcopy(Optimal_Heat_Flow_Required)
            Optimal_Heat_Flow_Gal_per_Min = Optimal_Heat_Flow_Required/3600*Metric_to_Imperial_Flow
        for bran in range(len(Optimal_Heat_Branches.keys())):
            branch = bran + 1
            Test_Branch = Optimal_Heat_Branches[branch]
            # Calculate the pumping flows first
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[-1-node]
                if Current_Node == 'Looped':
                    Looped = True
                    LoopMerge = True
                    continue
                elif Current_Node == 'Split':
                    Split = True
                    continue
                elif Current_Node == 'Merged':
                    Merged = True
                    continue
                elif sum(Optimal_Heat_Binary[Current_Node,:]) < 0.5:    # Edge case of terminal node
                    Optimal_Heat_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                    if Merged == True:
                        Merged = False
                        MergedHold = True
                    continue
                elif Looped == True:
                    Optimal_Heat_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                    Looped = False
                    continue
                elif Merged == True and MergedHold == True and Split == True:
                    Split = False
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Heat_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                continue
                        elif i == 1:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if LoopMerge == True and Current_Node == Test_Branch[-2]:
                        Merged = False
                        LoopMerge = False
                        MergedLoop = True
                        MergedHold = True
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node] - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node])
                    else:
                        Merged = False
                        MergedHold = True
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node])
                elif Merged == True and Split == True:
                    Split = False
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Heat_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            continue
                        elif i == 1:
                            Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if LoopMerge == True and Current_Node == Test_Branch[-2]:
                        Merged = False
                        LoopMerge = False
                        MergedLoop = True
                        MergedHold = True
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node] - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*Metric_to_Imperial_Flow
                        Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                    else:
                        Merged = False
                        MergedHold = True
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                        Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                elif Merged == True and MergedHold == True:
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if MergedLoop == True:
                        MergedLoop = False
                        Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        non_text_counter = 0
                        for i in range(len(Test_Branch)):
                            if isinstance(Test_Branch[-1-i], str) == True:
                                continue
                            elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                Loop_Branch_Node  = Test_Branch[i]
                                break
                            else:
                                non_text_counter += 1
                        for i in range(Num_Sites):
                            if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node] - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow- Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        else:
                            Merged = False
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    else:
                        Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        for i in range(Num_Sites):
                            if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node] - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600)
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow- Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        else:
                            Merged = False
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif MergedHold == True and Split == True:
                    Split = False
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Heat_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                continue
                        elif i == 1:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var-1:
                        Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                        Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                        Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif Merged == True:
                    if Test_Branch[-1] == 'Merged' and node == 1:
                        Merged = False
                        MergedHold = True
                        if Optimal_Heat_Node_Pressures[0,Current_Node] == 0:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                        continue
                    else:
                        if isinstance(Test_Branch[-1-(node-1)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-1)]
                        else:
                            if isinstance(Test_Branch[-1-(node-2)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-2)]
                            else:
                                if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                    Downstream_Node = Test_Branch[-1-(node-3)]
                                else:
                                    Downstream_Node = Test_Branch[-1-(node-4)]
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node] - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*3600
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Heat_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Heat_Delta_T/Density_Water*Metric_to_Imperial_Flow
                        else:
                            Merged = False
                            MergedHold = True
                            Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                            HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var-1:
                                Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                                Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]
                                Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif MergedHold == True:
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if MergedLoop == True:
                        MergedLoop = False
                        Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        non_text_counter = 0
                        for i in range(len(Test_Branch)):
                            if isinstance(Test_Branch[-1-i], str) == True:
                                continue
                            elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                Loop_Branch_Node  = Test_Branch[i]
                                break
                            else:
                                non_text_counter += 1
                        for i in range(Num_Sites):
                            if Optimal_Heat_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                        Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                        Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    else:
                        Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Heat_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        for i in range(Num_Sites):
                            if Optimal_Heat_Binary[i,Downstream_Node] > 0.5:
                                Total_Arc_Area += mp.pi*(Heat_Pipe_Diameters[Optimal_Heat_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                        Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                        Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                        if Current_Node != Plant_Location_Var-1:
                            Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                            Optimal_Heat_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]
                            Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif Split == True:
                    Split = False
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Heat_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            Current_Arc_0 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_0 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_0 = Pressure_Loss_0 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            continue
                        elif i == 1:
                            Current_Arc_1 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_1 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_1 = Pressure_Loss_1 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            Current_Arc_2 = Optimal_Heat_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_2 = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Heat_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_2 = Pressure_Loss_2 + Optimal_Heat_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var-1:
                        Optimal_Heat_Node_Pressures[:,Current_Node] = Downstream_Pressures
                        Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]
                        Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                else:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                    HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                    Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                    Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var-1:
                        Optimal_Heat_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                        Optimal_Heat_Flow_Required[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]
                        Optimal_Heat_Flow_Gal_per_Min[:,Current_Node] += Optimal_Heat_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                        
        for Downstream_Node in Nodes:
            Current_Node = Plant_Location_Var
            if Optimal_Heat_Binary[Current_Node, Downstream_Node] > 0.5:
                Current_Arc = Optimal_Heat_Connections[Current_Node,Downstream_Node]
                HW_Friction = 0.2083*(100/Heat_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Heat_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Heat_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                Optimal_Heat_Plant_Flows[:,Downstream_Node] = Optimal_Heat_Flow_Required[:,Downstream_Node]
                Optimal_Heat_Plant_Pressures[:,Downstream_Node] = Pressure_Loss+Optimal_Heat_Node_Pressures[:,Downstream_Node]
                        
        # if iter == 0: # OFFLINED # MODIFIED
        #     Optimal_Heat_Ideal_Arc_Flows = copy.deepcopy(Optimal_Heat_Arc_Flows)
        #     Optimal_Heat_Ideal_Plant_Flows = copy.deepcopy(Optimal_Heat_Plant_Flows)

        # Now work forward to calculate the heat losses
        Optimal_Heat_Arc_Supply_Losses = np.zeros((8760,Num_Sites,Num_Sites))
        Optimal_Heat_Arc_Return_Losses = np.zeros((8760,Num_Sites,Num_Sites))
        for bran in range(len(Optimal_Heat_Branches.keys())):
            branch = len(Optimal_Heat_Branches.keys())-bran
            Test_Branch = Optimal_Heat_Branches[branch]
            Merged = False
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[node]
                if isinstance(Current_Node, str) == True:
                    continue
                if node < len(Test_Branch)-1:
                    if Test_Branch[node+1] == 'Merged':
                        Merged = True
                if node > 0:
                    if isinstance(Test_Branch[node-1], str) == False:
                        Upstream_Node = Test_Branch[node-1]
                    else:
                        if isinstance(Test_Branch[node-2], str) == False:
                            Upstream_Node  = Test_Branch[node-2]
                        else:
                            if isinstance(Test_Branch[node-3], str) == False:
                                Upstream_Node = Test_Branch[node-3]
                            else:
                                Upstream_Node = Test_Branch[node-4]
                    Current_Arc = Optimal_Heat_Connections[Upstream_Node, Current_Node]
                    # Now calculate the time delay parameter
                    delta_L = PD.PairedDistance(Site_Coordinates[Upstream_Node,:],Site_Coordinates[Current_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Heat_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Upstream_Node,:],Site_Coordinates[Current_Node,:])
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Heat_Arc_Flows[:,Upstream_Node,Current_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                                        
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = Test_Tau.astype(int)
                    Start_Temp = Optimal_Heat_Hourly_Supply_Temps[:,Upstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Heat_Hourly_Supply_Temps[Tau_Hours-Test_Tau,Upstream_Node], Start_Temp)
                    BHAF = CalcHeatLoss(Optimal_Heat_Arc_Flows[:,Upstream_Node,Current_Node], Pipe_Volume)
                    Test_RT_1 = Optimal_Heat_Hourly_Return_Temps[:,Upstream_Node]
                    Test_RT_2 = Optimal_Heat_Hourly_Return_Temps[:,Current_Node]
                    for n in range(Num_Heat_Test_Points):
                        if n == 0:
                            Delta_ST = Start_Temp-(delta_L*(Heat_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)+Heat_Pipe_U2[Current_Arc]*(Start_Temp-Test_RT_1)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                        else:
                            Test_RT = n/Num_Heat_Test_Points*(Test_RT_1-Test_RT_2)+Test_RT_1
                            Delta_ST = Delta_ST-(delta_L*(Heat_Pipe_U[Current_Arc]*(Delta_ST-Hourly_Ground_Temperature)+Heat_Pipe_U2[Current_Arc]*(Delta_ST-Test_RT)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Heat_Arc_Supply_Losses[:,Upstream_Node,Current_Node] = BHAF*Density_Water*(Specific_Heat_Water)*(Start_Temp-Delta_ST)        # J
                    if Merged == True:
                        Total_Flow = np.zeros((8760))
                        Supply_Numerator = np.zeros((8760))
                        for i in range(Num_Sites):
                            if i == Current_Node:
                                continue
                            elif i == Upstream_Node:
                                Supply_Numerator += Delta_ST*Optimal_Heat_Arc_Flows[:,Upstream_Node,Current_Node]
                                Total_Flow += Optimal_Heat_Arc_Flows[:,Upstream_Node,Current_Node]
                            elif Optimal_Heat_Connections[i,Current_Node] > 0:
                                Supply_Numerator += (Optimal_Heat_Hourly_Supply_Temps[:,i]*Optimal_Heat_Arc_Flows[:,i,Current_Node]-Optimal_Heat_Arc_Supply_Losses[:,i,Current_Node]/Density_Water/Specific_Heat_Water)
                                Total_Flow += Optimal_Heat_Arc_Flows[:,i,Current_Node]
                            else:
                                continue
                        Optimal_Heat_Hourly_Supply_Temps[:,Current_Node] = Supply_Numerator/Total_Flow
                    else:
                        Optimal_Heat_Hourly_Supply_Temps[:,Current_Node] = Delta_ST
        Max_Building_Temp = max(Min_Secondary_Temp_Heating+Heat_Exchanger_Approach, Heating_Setpoint+Fan_Coil_Approach+Heat_Exchanger_Approach)            
        Optimal_Heat_Hourly_Building_delta_T = np.nan_to_num(Optimal_Heat_Hourly_Supply_Temps)-np.nan_to_num(Optimal_Heat_Hourly_Demand*1000/(Optimal_Heat_Building_Flows*Density_Water/3600)/Specific_Heat_Water)
        Optimal_Heat_Hourly_Building_delta_T = np.nan_to_num(np.where(Optimal_Heat_Hourly_Building_delta_T < Max_Building_Temp, Max_Building_Temp, Optimal_Heat_Hourly_Building_delta_T))
        for bran in range(len(Optimal_Heat_Branches.keys())):
            branch = len(Optimal_Heat_Branches.keys())-bran
            Test_Branch = Optimal_Heat_Branches[branch]
            FirstNum = False
            Merged = False
            Plant_Return_Flow = np.zeros((8760))
            Plant_Return_Temperature = np.zeros((8760))
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[-1-node]
                if isinstance(Current_Node, str) == True:
                    continue
                if 0 < node and node < len(Test_Branch)-1:
                    if Test_Branch[-1-(node-1)] == 'Split':
                        Merged = True
                if FirstNum == False:
                    Optimal_Heat_Hourly_Return_Temps[:,Current_Node] = Optimal_Heat_Hourly_Building_delta_T[:,Current_Node]
                    FirstNum = True
                    continue
                if Current_Node == 0:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Heat_Connections[Current_Node, Downstream_Node]
                    delta_L = PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Heat_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])
                    BHAF = CalcHeatLoss(Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node], Pipe_Volume)
                    Test_ST_1 = Optimal_Heat_Hourly_Supply_Temps[:,Downstream_Node]
                    Test_ST_2 = Optimal_Heat_Hourly_Supply_Temps[:,Current_Node]
                    ## ADDED TO AVOID INDEX ERROR CAUSED BY ZERO OPTIMAL FLOW
                    # Tau = Pipe_Volume/Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]         # hours
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                    
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = Test_Tau.astype(int)
                    Start_Temp = Optimal_Heat_Hourly_Return_Temps[:,Downstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Heat_Hourly_Return_Temps[Tau_Hours-Test_Tau,Downstream_Node], Start_Temp)
                    for n in range(Num_Heat_Test_Points):
                        if n == 0:
                            Delta_RT = Start_Temp-delta_L*(Heat_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)-Heat_Pipe_U2[Current_Arc]*(Test_ST_1-Start_Temp))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                        else:
                            Test_ST = n/Num_Heat_Test_Points*(Test_ST_1-Test_ST_2)+Test_ST_1
                            Delta_RT = Delta_RT-(delta_L*(Heat_Pipe_U[Current_Arc]*(Delta_RT-Hourly_Ground_Temperature)-Heat_Pipe_U2[Current_Arc]*(Test_ST-Delta_RT)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Heat_Arc_Return_Losses[:,Current_Node,Downstream_Node] = BHAF*Density_Water*(Specific_Heat_Water)*(Start_Temp-Delta_RT)        # J
                    Plant_Return_Flow += Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]
                    Plant_Return_Temperature += Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]*Delta_RT
                elif node > 0:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Heat_Connections[Current_Node, Downstream_Node]
                    delta_L = PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Heat_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])
                    BHAF = CalcHeatLoss(Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node], Pipe_Volume)
                    Test_ST_1 = Optimal_Heat_Hourly_Supply_Temps[:,Downstream_Node]
                    Test_ST_2 = Optimal_Heat_Hourly_Supply_Temps[:,Current_Node]
                    ## ADDED TO AVOID INDEX ERROR CAUSED BY ZERO OPTIMAL FLOW
                    # Tau = Pipe_Volume/Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]         # hours
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                    
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = Test_Tau.astype(int)
                    Start_Temp = Optimal_Heat_Hourly_Return_Temps[:,Downstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Heat_Hourly_Return_Temps[Tau_Hours-Test_Tau,Downstream_Node], Start_Temp)
                    for n in range(Num_Heat_Test_Points):
                        if n == 0:
                            Delta_RT = Start_Temp-delta_L*(Heat_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)-Heat_Pipe_U2[Current_Arc]*(Test_ST_1-Start_Temp))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                        else:
                            Test_ST = n/Num_Heat_Test_Points*(Test_ST_1-Test_ST_2)+Test_ST_1
                            Delta_RT = Delta_RT-(delta_L*(Heat_Pipe_U[Current_Arc]*(Delta_RT-Hourly_Ground_Temperature)-Heat_Pipe_U2[Current_Arc]*(Test_ST-Delta_RT)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Heat_Arc_Return_Losses[:,Current_Node,Downstream_Node] = BHAF*Density_Water*(Specific_Heat_Water)*(Start_Temp-Delta_RT)        # J
                    if Merged == True:
                        Total_Flow = np.zeros((8760))
                        Return_Numerator = np.zeros((8760))
                        for i in range(Num_Sites):
                            if i == Current_Node:
                                continue
                            elif i == Downstream_Node:
                                Return_Numerator += Delta_RT*Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]
                                Total_Flow += Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]
                            elif Optimal_Heat_Connections[Current_Node,i] > 0:
                                Return_Numerator += (Optimal_Heat_Hourly_Return_Temps[:,i]*Optimal_Heat_Arc_Flows[:,Current_Node,i]-Optimal_Heat_Arc_Return_Losses[:,Current_Node,i]/Density_Water/Specific_Heat_Water)
                                Total_Flow += Optimal_Heat_Arc_Flows[:,Current_Node,i]
                            else:
                                continue
                        Optimal_Heat_Hourly_Return_Temps[:,Current_Node] = (Return_Numerator + Optimal_Heat_Building_Flows[:,Current_Node]*Optimal_Heat_Hourly_Building_delta_T[:,Current_Node])/(Total_Flow+Optimal_Heat_Building_Flows[:,Current_Node])
                    else:
                        Optimal_Heat_Hourly_Return_Temps[:,Current_Node] = (Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node]*Delta_RT + Optimal_Heat_Building_Flows[:,Current_Node]*Optimal_Heat_Hourly_Building_delta_T[:,Current_Node])/(Optimal_Heat_Arc_Flows[:,Current_Node,Downstream_Node] + Optimal_Heat_Building_Flows[:,Current_Node])
            Optimal_Heat_Hourly_Return_Temps[:,0] = np.nan_to_num(Plant_Return_Temperature/Plant_Return_Flow)
        Optimal_Heat_Arc_Losses = Optimal_Heat_Arc_Supply_Losses + Optimal_Heat_Arc_Return_Losses 

    '''-----------------------------------------------------------------------------------------------'''
    # Calculate Cooling and Pumping Losses #
    '''-----------------------------------------------------------------------------------------------'''
                    
    # Calculate the Cool loss and pumping loss iteratively
    # Set up the arrays that will hold node temperatures, arc losses, and arc flows for all hours
    Optimal_Cool_Node_Pressures = np.zeros((8760,Num_Sites))
    Optimal_Cool_Arc_Supply_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Cool_Arc_Return_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Cool_Arc_Losses = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Cool_Arc_Flows = np.zeros((8760,Num_Sites,Num_Sites))
    Optimal_Cool_Hourly_Demand = np.zeros((8760,Num_Sites))
    Optimal_Cool_Hourly_Supply_Temps = np.ones((8760,Num_Sites))*Cool_Sup_Temp_Var
    for i in range(8760):
        if i < Summer_Thermal_Reset_Hour or i >= Winter_Thermal_Reset_Hour:
            Optimal_Cool_Hourly_Supply_Temps[i,:] += Cool_Sup_Temp_Reset_Var
    Optimal_Cool_Hourly_Return_Temps = np.ones((8760,Num_Sites))*Test_Cool_Return_Temp
    Optimal_Cool_Hourly_Building_Return_Temps = np.zeros((8760,Num_Sites))
    for num in range(len(Site_Vars)):
        if Site_Vars[num] != Num_Buildings+1:
            Optimal_Cool_Hourly_Demand[:,num] = Cooling_Input[:,Site_Vars[num]-1]
    Optimal_Useful_Cool_Flow = Optimal_Cool_Hourly_Demand*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600
    Optimal_Cool_Flow_Required = np.zeros((8760,Num_Sites))
    for i in Nodes:
        if Site_Vars[i] != Num_Buildings+1 and Site_Vars[i] != 0:
            Minimum_Flow_Provided = Min_Flow_Required*max(Cooling_Input[:,Site_Vars[i]-1]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)
            Optimal_Cool_Flow_Required[:,i] = np.where(Optimal_Useful_Cool_Flow[:,i] < Minimum_Flow_Provided, Minimum_Flow_Provided, Optimal_Useful_Cool_Flow[:,i])
    Optimal_Cool_Building_Flows = copy.deepcopy(Optimal_Cool_Flow_Required)
    Optimal_Cool_Flow_Gal_per_Min = Optimal_Cool_Flow_Required/3600*Metric_to_Imperial_Flow
    Optimal_Cool_Plant_Pressures  = np.zeros((8760,Num_Sites))
    Optimal_Cool_Plant_Flows = np.zeros((8760,Num_Sites))
    # Optimal_Cool_Arc_Return_Pipe_Temps = np.zeros((8760,Num_Sites,Num_Sites)) # OFFLINED # MODIFIED

    for iter in range(Num_Iterations):
        Looped = False
        Split = False
        Merged = False
        LoopMerge = False
        MergedHold = False
        MergedLoop = False
        if iter > 0:
            Min_Building_Temp = min(Max_Secondary_Temp_Cooling-Heat_Exchanger_Approach, Cooling_Setpoint-Fan_Coil_Approach-Heat_Exchanger_Approach)
            Test_Temps = Optimal_Cool_Hourly_Supply_Temps+Cooling_Loop_Max_delta_T
            Optimal_Cool_Hourly_Building_Return_Temps = np.where(Test_Temps < Min_Building_Temp, Test_Temps, Min_Building_Temp)
            Optimal_Useful_Cool_Flow = np.nan_to_num(Optimal_Cool_Hourly_Demand*1000/Specific_Heat_Water/(Optimal_Cool_Hourly_Building_Return_Temps-Optimal_Cool_Hourly_Supply_Temps)/Density_Water*3600)
            for i in Nodes:
                if Site_Vars[i] != Num_Buildings+1 and Site_Vars[i] != 0:
                    Minimum_Flow_Provided = Min_Flow_Required*max(Cooling_Input[:,Site_Vars[i]-1]*1000/Specific_Heat_Water/(Optimal_Cool_Hourly_Building_Return_Temps[:,i]-Optimal_Cool_Hourly_Supply_Temps[:,i])/Density_Water*3600)
                    Optimal_Cool_Flow_Required[:,i] = np.where(Optimal_Useful_Cool_Flow[:,i] < Minimum_Flow_Provided, Minimum_Flow_Provided, Optimal_Useful_Cool_Flow[:,i])
            Optimal_Cool_Building_Flows = copy.deepcopy(Optimal_Cool_Flow_Required)
            
            Optimal_Cool_Building_Flows = np.maximum(0.00000001,Optimal_Cool_Building_Flows) ### MODIFIED # ADDED TO AVOID NAN IN THE FOLLOWING SECTIONS
            
            Optimal_Cool_Flow_Gal_per_Min = Optimal_Cool_Flow_Required/3600*Metric_to_Imperial_Flow
        for bran in range(len(Optimal_Cool_Branches.keys())):
            branch = bran + 1
            Test_Branch = Optimal_Cool_Branches[branch]
            # Calculate the pumping flows first
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[-1-node]
                if Current_Node == 'Looped':
                    Looped = True
                    LoopMerge = True
                    continue
                elif Current_Node == 'Split':
                    Split = True
                    continue
                elif Current_Node == 'Merged':
                    Merged = True
                    continue
                elif sum(Optimal_Cool_Binary[Current_Node,:]) < 0.5:    # Edge case of terminal node
                    Optimal_Cool_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                    if Merged == True:
                        Merged = False
                        MergedHold = True
                    continue
                elif Looped == True:
                    Optimal_Cool_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                    Looped = False
                    continue
                elif Merged == True and MergedHold == True and Split == True:
                    Split = False
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Cool_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                continue
                        elif i == 1:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_1 = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if LoopMerge == True and Current_Node == Test_Branch[-2]:
                        Merged = False
                        LoopMerge = False
                        MergedLoop = True
                        MergedHold = True
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node] - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node])
                    else:
                        Merged = False
                        MergedHold = True
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node])
                elif Merged == True and Split == True:
                    Split = False
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Cool_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            continue
                        elif i == 1:
                            Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if LoopMerge == True and Current_Node == Test_Branch[-2]:
                        Merged = False
                        LoopMerge = False
                        MergedLoop = True
                        MergedHold = True
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node] - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*Metric_to_Imperial_Flow
                        Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                    else:
                        Merged = False
                        MergedHold = True
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                            Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                        Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                elif Merged == True and MergedHold == True:
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if MergedLoop == True:
                        MergedLoop = False
                        Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        non_text_counter = 0
                        for i in range(len(Test_Branch)):
                            if isinstance(Test_Branch[-1-i], str) == True:
                                continue
                            elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                Loop_Branch_Node  = Test_Branch[i]
                                break
                            else:
                                non_text_counter += 1
                        for i in range(Num_Sites):
                            if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node] - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow- Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        else:
                            Merged = False
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    else:
                        Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        for i in range(Num_Sites):
                            if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node] - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600)
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*(Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow- Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*Metric_to_Imperial_Flow)
                        else:
                            Merged = False
                            MergedHold = True
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Flow_Factor*Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif MergedHold == True and Split == True:
                    Split = False
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Cool_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_0]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                continue
                        elif i == 1:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_0 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_0 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_1]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            if Downstream_Connections[i] == Downstream_Node:
                                if MergedLoop == True:
                                    MergedLoop = False
                                    Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    non_text_counter = 0
                                    for i in range(len(Test_Branch)):
                                        if isinstance(Test_Branch[-1-i], str) == True:
                                            continue
                                        elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                            Loop_Branch_Node  = Test_Branch[i]
                                            break
                                        else:
                                            non_text_counter += 1
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                else:
                                    Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                                    Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc_2]/2)**2
                                    Total_Arc_Area = 0
                                    for i in range(Num_Sites):
                                        if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                            Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                                    Flow_Factor = Current_Arc_Area/Total_Arc_Area
                                    HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                    Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                                    Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            else:
                                Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                                HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                                Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                                Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                                Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var:
                        Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                        Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                        Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif Merged == True:
                    if Test_Branch[-1] == 'Merged' and node == 1:
                        Merged = False
                        MergedHold = True
                        if Optimal_Cool_Node_Pressures[0,Current_Node] == 0:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = np.ones((8760))*Pressure_Differential
                        continue
                    else:
                        if isinstance(Test_Branch[-1-(node-1)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-1)]
                        else:
                            if isinstance(Test_Branch[-1-(node-2)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-2)]
                            else:
                                if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                    Downstream_Node = Test_Branch[-1-(node-3)]
                                else:
                                    Downstream_Node = Test_Branch[-1-(node-4)]
                        if LoopMerge == True and Current_Node == Test_Branch[-2]:
                            Merged = False
                            LoopMerge = False
                            MergedLoop = True
                            MergedHold = True
                            Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node] - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*3600
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow - Optimal_Cool_Hourly_Demand[:,Downstream_Node]*1000/Specific_Heat_Water/Test_Cool_Delta_T/Density_Water*Metric_to_Imperial_Flow
                        else:
                            Merged = False
                            MergedHold = True
                            Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                            HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                            Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                            if Current_Node != Plant_Location_Var:
                                Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                                Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]
                                Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif MergedHold == True:
                    MergedHold = False
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    if MergedLoop == True:
                        MergedLoop = False
                        Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        non_text_counter = 0
                        for i in range(len(Test_Branch)):
                            if isinstance(Test_Branch[-1-i], str) == True:
                                continue
                            elif non_text_counter == 1 and isinstance(Test_Branch[-1-i], str) == False:
                                Loop_Branch_Node  = Test_Branch[i]
                                break
                            else:
                                non_text_counter += 1
                        for i in range(Num_Sites):
                            if Optimal_Cool_Binary[i,Downstream_Node] > 0.5 and i != Loop_Branch_Node:
                                Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                        Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                        Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                    else:
                        Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                        Current_Arc_Area = mp.pi*(Cool_Pipe_Diameters[Current_Arc]/2)**2
                        Total_Arc_Area = 0
                        for i in range(Num_Sites):
                            if Optimal_Cool_Binary[i,Downstream_Node] > 0.5:
                                Total_Arc_Area += mp.pi*(Cool_Pipe_Diameters[Optimal_Cool_Connections[i,Downstream_Node]]/2)**2
                        Flow_Factor = Current_Arc_Area/Total_Arc_Area
                        HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*(Flow_Factor*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node]))**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                        Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                        Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                        if Current_Node != Plant_Location_Var:
                            Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                            Optimal_Cool_Flow_Required[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]
                            Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Flow_Factor*Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                elif Split == True:
                    Split = False
                    Downstream_Connections = []
                    for i in range(Num_Sites):
                        if Optimal_Cool_Binary[Current_Node,i] > 0.5:
                            Downstream_Connections += [i]
                    for i in range(len(Downstream_Connections)):
                        if i == 0:
                            Current_Arc_0 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_0 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_0])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_0]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_0 = Density_Water*9.81*HW_Friction_0/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_0 = Pressure_Loss_0 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            continue
                        elif i == 1:
                            Current_Arc_1 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_1 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_1])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_1]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_1 = Density_Water*9.81*HW_Friction_1/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_1 = Pressure_Loss_1 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Pressure_0,Pressure_1)
                        else:
                            Current_Arc_2 = Optimal_Cool_Connections[Current_Node,Downstream_Connections[i]]
                            HW_Friction_2 = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc_2])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Connections[i]])**1.852/((Cool_Pipe_Diameters[Current_Arc_2]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                            Pressure_Loss_2 = Density_Water*9.81*HW_Friction_2/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Connections[i],:])/1000         # kPa
                            Pressure_2 = Pressure_Loss_2 + Optimal_Cool_Node_Pressures[:,Downstream_Connections[i]]
                            Downstream_Pressures = np.maximum(Downstream_Pressures, Pressure_2)
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var:
                        Optimal_Cool_Node_Pressures[:,Current_Node] = Downstream_Pressures
                        Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]
                        Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                else:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                    HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                    Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                    Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                    if Current_Node != Plant_Location_Var:
                        Optimal_Cool_Node_Pressures[:,Current_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                        Optimal_Cool_Flow_Required[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]
                        Optimal_Cool_Flow_Gal_per_Min[:,Current_Node] += Optimal_Cool_Flow_Required[:,Downstream_Node]/3600*Metric_to_Imperial_Flow
                        
        for Downstream_Node in Nodes:
            Current_Node = Plant_Location_Var
            if Optimal_Cool_Binary[Current_Node, Downstream_Node] > 0.5:
                Current_Arc = Optimal_Cool_Connections[Current_Node,Downstream_Node]
                HW_Friction = 0.2083*(100/Cool_Pipe_HW_Roughnesses[Current_Arc])**1.852*abs(Optimal_Cool_Flow_Gal_per_Min[:,Downstream_Node])**1.852/((Cool_Pipe_Diameters[Current_Arc]*m_to_in)**4.8655)      # ft Head Loss/100 Feet Pipe (also m/100 m pipe)
                Pressure_Loss = Density_Water*9.81*HW_Friction/100*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/1000         # kPa
                Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                Optimal_Cool_Plant_Flows[:,Downstream_Node] = Optimal_Cool_Flow_Required[:,Downstream_Node]
                Optimal_Cool_Plant_Pressures[:,Downstream_Node] = Pressure_Loss+Optimal_Cool_Node_Pressures[:,Downstream_Node]
                        
        # if iter == 0: # OFFLINED # MODIFIED
        #     Optimal_Cool_Ideal_Arc_Flows = copy.deepcopy(Optimal_Cool_Arc_Flows)
        #     Optimal_Cool_Ideal_Plant_Flows = copy.deepcopy(Optimal_Cool_Plant_Flows)

        # Now work forward to calculate the heat losses from cooling
        Optimal_Cool_Arc_Supply_Losses = np.zeros((8760,Num_Sites,Num_Sites))
        Optimal_Cool_Arc_Return_Losses = np.zeros((8760,Num_Sites,Num_Sites))
        for bran in range(len(Optimal_Cool_Branches.keys())):
            branch = len(Optimal_Cool_Branches.keys())-bran
            Test_Branch = Optimal_Cool_Branches[branch]
            Merged = False
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[node]
                if isinstance(Current_Node, str) == True:
                    continue
                if node < len(Test_Branch)-1:
                    if Test_Branch[node+1] == 'Merged':
                        Merged = True
                if node > 0:
                    if isinstance(Test_Branch[node-1], str) == False:
                        Upstream_Node = Test_Branch[node-1]
                    else:
                        if isinstance(Test_Branch[node-2], str) == False:
                            Upstream_Node  = Test_Branch[node-2]
                        else:
                            if isinstance(Test_Branch[node-3], str) == False:
                                Upstream_Node = Test_Branch[node-3]
                            else:
                                Upstream_Node = Test_Branch[node-4]
                    Current_Arc = Optimal_Cool_Connections[Upstream_Node, Current_Node]
                    delta_L = PD.PairedDistance(Site_Coordinates[Upstream_Node,:],Site_Coordinates[Current_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Cool_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Upstream_Node,:],Site_Coordinates[Current_Node,:])
                    ## ADDED TO AVOID INDEX ERROR CAUSED BY ZERO OPTIMAL FLOW
                    # Tau = Pipe_Volume/Optimal_Cool_Arc_Flows[:,Upstream_Node,Current_Node]         # hours
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Cool_Arc_Flows[:,Upstream_Node,Current_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                    
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = np.nan_to_num(Test_Tau.astype(int))
                    Start_Temp = Optimal_Cool_Hourly_Supply_Temps[:,Upstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Cool_Hourly_Supply_Temps[Tau_Hours-Test_Tau,Upstream_Node], Start_Temp)
                    BHAF = CalcHeatLoss(Optimal_Cool_Arc_Flows[:,Upstream_Node,Current_Node], Pipe_Volume)
                    Test_RT_1 = Optimal_Cool_Hourly_Return_Temps[:,Upstream_Node]
                    Test_RT_2 = Optimal_Cool_Hourly_Return_Temps[:,Current_Node]
                    for n in range(Num_Heat_Test_Points):
                        if n == 0:
                            Delta_ST = Start_Temp-(delta_L*(Cool_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)+Cool_Pipe_U2[Current_Arc]*(Start_Temp-Test_RT_1)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                        else:
                            Test_RT = n/Num_Heat_Test_Points*(Test_RT_1-Test_RT_2)+Test_RT_1
                            Delta_ST = Delta_ST-(delta_L*(Cool_Pipe_U[Current_Arc]*(Delta_ST-Hourly_Ground_Temperature)+Cool_Pipe_U2[Current_Arc]*(Delta_ST-Test_RT)))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Cool_Arc_Supply_Losses[:,Upstream_Node,Current_Node] = BHAF*Density_Water*(Specific_Heat_Water)*(Delta_ST-Start_Temp)        # J
                    if Merged == True:
                        Total_Flow = np.zeros((8760))
                        Supply_Numerator = np.zeros((8760))
                        for i in range(Num_Sites):
                            if i == Current_Node:
                                continue
                            elif i == Upstream_Node:
                                Supply_Numerator += Delta_ST*Optimal_Cool_Arc_Flows[:,Upstream_Node,Current_Node]
                                Total_Flow += Optimal_Cool_Arc_Flows[:,Upstream_Node,Current_Node]
                            elif Optimal_Cool_Connections[i,Current_Node] > 0:
                                Supply_Numerator += (Optimal_Cool_Hourly_Supply_Temps[:,i]*Optimal_Cool_Arc_Flows[:,i,Current_Node]+Optimal_Cool_Arc_Supply_Losses[:,i,Current_Node]/Density_Water/Specific_Heat_Water)
                                Total_Flow += Optimal_Cool_Arc_Flows[:,i,Current_Node]
                            else:
                                continue
                        Optimal_Cool_Hourly_Supply_Temps[:,Current_Node] = Supply_Numerator/Total_Flow
                    else:
                        Optimal_Cool_Hourly_Supply_Temps[:,Current_Node] = Delta_ST
        Min_Building_Temp = min(Max_Secondary_Temp_Cooling-Heat_Exchanger_Approach, Cooling_Setpoint-Fan_Coil_Approach-Heat_Exchanger_Approach)
        Optimal_Cool_Hourly_Building_delta_T = np.nan_to_num(Optimal_Cool_Hourly_Supply_Temps)+np.nan_to_num(Optimal_Cool_Hourly_Demand*1000/(Optimal_Cool_Building_Flows*Density_Water/3600)/Specific_Heat_Water) # has NAN!!!!!!!!!!!!!!!!!!!!
        Optimal_Cool_Hourly_Building_delta_T = np.nan_to_num(np.where(Optimal_Cool_Hourly_Building_delta_T > Min_Building_Temp, Min_Building_Temp, Optimal_Cool_Hourly_Building_delta_T))
        for bran in range(len(Optimal_Cool_Branches.keys())):
            branch = len(Optimal_Cool_Branches.keys())-bran
            Test_Branch = Optimal_Cool_Branches[branch]
            FirstNum = False
            Merged = False
            Plant_Return_Flow = np.zeros((8760))
            Plant_Return_Temperature = np.zeros((8760))
            for node in range(len(Test_Branch)):
                Current_Node = Test_Branch[-1-node]
                if isinstance(Current_Node, str) == True:
                    continue
                if 0 < node and node < len(Test_Branch)-1:
                    if Test_Branch[-1-(node-1)] == 'Split':
                        Merged = True
                if FirstNum == False:
                    Optimal_Cool_Hourly_Return_Temps[:,Current_Node] = Optimal_Cool_Hourly_Building_delta_T[:,Current_Node] # Has NAN!!!!!!!!!!!!!!!!!!!!!
                    FirstNum = True
                    continue
                if Current_Node == 0:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Cool_Connections[Current_Node, Downstream_Node]
                    delta_L = PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Cool_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])
                    BHAF = CalcHeatLoss(Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node], Pipe_Volume)
                    Test_ST_1 = Optimal_Cool_Hourly_Supply_Temps[:,Downstream_Node]
                    Test_ST_2 = Optimal_Cool_Hourly_Supply_Temps[:,Current_Node]
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                    
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = Test_Tau.astype(int)
                    Start_Temp = Optimal_Cool_Hourly_Return_Temps[:,Downstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Cool_Hourly_Return_Temps[Tau_Hours-Test_Tau,Downstream_Node], Start_Temp)
                    for n in range(Num_Heat_Test_Points):
                        if n == 0: ## Start_Temp: HAS NAN !!!!!!!!!!!!!!!!!!!!!!!!
                            Delta_RT = np.nan_to_num(Start_Temp-delta_L*(Cool_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)-Cool_Pipe_U2[Current_Arc]*(Test_ST_1-Start_Temp))/(BHAF*Density_Water/3600)/(Specific_Heat_Water))
                        else:
                            Test_ST = n/Num_Heat_Test_Points*(Test_ST_1-Test_ST_2)+Test_ST_1
                            Delta_RT = Delta_RT-delta_L*(Cool_Pipe_U[Current_Arc]*(Delta_RT-Hourly_Ground_Temperature)-Cool_Pipe_U2[Current_Arc]*(Test_ST-Delta_RT))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Cool_Arc_Return_Losses[:,Current_Node,Downstream_Node] = np.nan_to_num(BHAF*Density_Water*(Specific_Heat_Water)*(Delta_RT-Start_Temp))        # J  # HAS NAN!!!!!!!!!!!!!!!
                    Plant_Return_Flow += Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]
                    Plant_Return_Temperature += Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]*Delta_RT
                elif node > 0:
                    if isinstance(Test_Branch[-1-(node-1)], str) == False:
                        Downstream_Node = Test_Branch[-1-(node-1)]
                    else:
                        if isinstance(Test_Branch[-1-(node-2)], str) == False:
                            Downstream_Node = Test_Branch[-1-(node-2)]
                        else:
                            if isinstance(Test_Branch[-1-(node-3)], str) == False:
                                Downstream_Node = Test_Branch[-1-(node-3)]
                            else:
                                Downstream_Node = Test_Branch[-1-(node-4)]
                    Current_Arc = Optimal_Cool_Connections[Current_Node, Downstream_Node]
                    delta_L = PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])/Num_Heat_Test_Points
                    Pipe_Volume = (Cool_Pipe_Diameters[Current_Arc]/2)**2*mp.pi*PD.PairedDistance(Site_Coordinates[Current_Node,:],Site_Coordinates[Downstream_Node,:])
                    BHAF = CalcHeatLoss(Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node], Pipe_Volume)
                    Test_ST_1 = Optimal_Cool_Hourly_Supply_Temps[:,Downstream_Node]
                    Test_ST_2 = Optimal_Cool_Hourly_Supply_Temps[:,Current_Node]
                    ## ADDED TO AVOID INDEX ERROR CAUSED BY ZERO OPTIMAL FLOW
                    Tau = np.maximum(np.minimum(np.nan_to_num(Pipe_Volume/Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]), Tau_Hours), 0)         # hours # MODIFIED --> CHANGED TO MINIMUM TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau  # ADDED --> ADDED TO DROP NEGATIVE VALUES TO AVOID NEGATIVE INDICES HERE: Tau_Hours-Test_Tau
                    
                    Test_Tau = np.round(Tau,0)      # hours to find start temperature
                    Test_Tau = Test_Tau.astype(int)
                    Start_Temp = Optimal_Cool_Hourly_Return_Temps[:,Downstream_Node]
                    Start_Temp = np.where(Test_Tau > 0, Optimal_Cool_Hourly_Return_Temps[Tau_Hours-Test_Tau,Downstream_Node], Start_Temp)
                    for n in range(Num_Heat_Test_Points):
                        if n == 0:
                            Delta_RT = Start_Temp-delta_L*(Cool_Pipe_U[Current_Arc]*(Start_Temp-Hourly_Ground_Temperature)-Cool_Pipe_U2[Current_Arc]*(Test_ST_1-Start_Temp))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                        else:
                            Test_ST = n/Num_Heat_Test_Points*(Test_ST_1-Test_ST_2)+Test_ST_1
                            Delta_RT = Delta_RT-delta_L*(Cool_Pipe_U[Current_Arc]*(Delta_RT-Hourly_Ground_Temperature)-Cool_Pipe_U2[Current_Arc]*(Test_ST-Delta_RT))/(BHAF*Density_Water/3600)/(Specific_Heat_Water)
                    Optimal_Cool_Arc_Return_Losses[:,Current_Node,Downstream_Node] = np.nan_to_num(BHAF*Density_Water*(Specific_Heat_Water)*(Delta_RT-Start_Temp))        # J    # HAS NAN!!!!!!!!!!!!!!!
                    if Merged == True:
                        Total_Flow = np.zeros((8760))
                        Return_Numerator = np.zeros((8760))
                        for i in range(Num_Sites):
                            if i == Current_Node:
                                continue
                            elif i == Downstream_Node:
                                Return_Numerator += Delta_RT*Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]
                                Total_Flow += Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]
                            elif Optimal_Cool_Connections[Current_Node,i] > 0:
                                Return_Numerator += (Optimal_Cool_Hourly_Return_Temps[:,i]*Optimal_Cool_Arc_Flows[:,Current_Node,i]+Optimal_Cool_Arc_Return_Losses[:,Current_Node,i]/Density_Water/Specific_Heat_Water)
                                Total_Flow += Optimal_Cool_Arc_Flows[:,Current_Node,i]
                            else:
                                continue
                        Optimal_Cool_Hourly_Return_Temps[:,Current_Node] = (Return_Numerator + Optimal_Cool_Building_Flows[:,Current_Node]*Optimal_Cool_Hourly_Building_delta_T[:,Current_Node])/(Total_Flow+Optimal_Cool_Building_Flows[:,Current_Node])
                    else:
                        Optimal_Cool_Hourly_Return_Temps[:,Current_Node] = (Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node]*Delta_RT + Optimal_Cool_Building_Flows[:,Current_Node]*Optimal_Cool_Hourly_Building_delta_T[:,Current_Node])/(Optimal_Cool_Arc_Flows[:,Current_Node,Downstream_Node] + Optimal_Cool_Building_Flows[:,Current_Node])
            Optimal_Cool_Hourly_Return_Temps[:,0] = np.nan_to_num(Plant_Return_Temperature/Plant_Return_Flow)
        Optimal_Cool_Arc_Losses = np.nan_to_num(Optimal_Cool_Arc_Supply_Losses + Optimal_Cool_Arc_Return_Losses)

    '''-----------------------------------------------------------------------------------------------'''
    # Calculate Capital Costs, Pumping Electricity Consumption Values, and Heat Losses #
    '''-----------------------------------------------------------------------------------------------'''
    Plant_Heat_Pumping_Loss_Electricity = np.zeros((8760))
    Plant_Cool_Pumping_Loss_Electricity = np.zeros((8760))
    Network_Heat_Loss = np.zeros((8760))
    Network_Cool_Loss = np.zeros((8760))
      
    # First need to calculate the pumping electricity used by the plant in each hour based on pressures. Also find the hourly
    # total flows from the plant. Note that the pressures are doubled to account for pressure loss in supply and return pipes.
    # The heat loss in each hour for heating and cooling also has to be calculated.
    for node in Nodes:
        Plant_Heat_Pumping_Loss_Electricity += Optimal_Heat_Plant_Flows[:,node]*(2*Optimal_Heat_Plant_Pressures[:,node] - np.ones((8760))*Pressure_Differential)/3600/Pump_Efficiency   # kWh
        Plant_Cool_Pumping_Loss_Electricity += Optimal_Cool_Plant_Flows[:,node]*(2*Optimal_Cool_Plant_Pressures[:,node] - np.ones((8760))*Pressure_Differential)/3600/Pump_Efficiency   # kWh
        for node2 in Nodes:
            Network_Heat_Loss += Optimal_Heat_Arc_Losses[:,node,node2]/kWh_to_J
            Network_Cool_Loss += np.nan_to_num(Optimal_Cool_Arc_Losses[:,node,node2]/kWh_to_J) ## NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Now calculate the capital costs
    Optimal_Heat_Pipe_Capital_Cost = 0
    Optimal_Cool_Pipe_Capital_Cost = 0
    Optimal_Trench_Capital_Cost = 0
    
    for i in range(Num_Sites):
        for j in range(Num_Sites):
            if Optimal_Heat_Connections[i,j] > 0.5:
                Optimal_Heat_Pipe_Capital_Cost += Heat_Arc_Capex[(i,j,Optimal_Heat_Connections[i,j])]
            if Optimal_Cool_Connections[i,j] > 0.5:
                Optimal_Cool_Pipe_Capital_Cost += Cool_Arc_Capex[(i,j,Optimal_Cool_Connections[i,j])]
            if Optimal_Heat_Connections[i,j] > 0.5 or Optimal_Cool_Connections[i,j] > 0.5:
                Optimal_Trench_Capital_Cost += Total_Trench_Costs[(i,j)]
        
    '''-----------------------------------------------------------------------------------------------'''
    # Calculate Overall Community Demand #
    '''-----------------------------------------------------------------------------------------------'''
    # Use the Site_Vars dictionary and the dictionary of demands to create an aggregate function of demand
    Aggregate_Demand = 0
    for i in range(Num_Sites):
        Building_Type = Site_Vars[i]
        if Building_Type != 0 and Building_Type != Num_Buildings+1:
            Aggregate_Demand += Diversifier_Peak*np.column_stack((Init_Demand_Types[Building_Type][:,0], Init_Demand_Types[Building_Type][:,1], Init_Demand_Types[Building_Type][:,2]))

    # Now calculate the electricity for municipal lighting needs. To do this, we assume that the streets are laid
    # out in a grid that go in the x (E-W) direction and the y (N-S) direction, and that in the x direction they 
    # have a certain spacing that is different in the y direction. From there, lighting parameters are assumed and
    # set in the parameters section. The ones used here are from the San Jose Street Lighting Guide but could be 
    # changed.
    Site_Width = max(Site_Coordinates[:,0])
    Site_Length = max(Site_Coordinates[:,1])
    Aspect_Ratio = Site_Width/Site_Length
    Num_Sites_Length = np.round(np.sqrt(Num_Sites/Aspect_Ratio),0)
    # Num_Sites_Width = np.round(Num_Sites_Length*Aspect_Ratio,0) # OFFLINED # MODIFIED # UNUSED
    Num_Streets_E_W = np.floor(Num_Sites_Length/E_W_Street_Spacing)         # Note this is streets running E-W, the x direction
    Num_Streets_N_S = np.floor(Num_Sites_Length/N_S_Street_Spacing)         # Note this is streets running N-S, the y direction
    E_W_Street_Length = Site_Width*Num_Streets_E_W
    N_S_Street_Length = Site_Length*Num_Streets_N_S
    Total_Street_Length = E_W_Street_Length+N_S_Street_Length
    
    # Add in municipal lighting loads
    ## REWRITTEN BASED ON THE CORRECTED CURFEW MODIFIER FROM SF_CaseStudy_Ch1_Pouya.py
    hours = np.array(range(8760))
    hours %= 24
    hours_lights_on = np.logical_or(((hours >= 19) * (hours <= 23)), ((hours >= 0) * (hours <= 6)))
    hours_lights_half_power = ((hours >= 2) * (hours <= 6))*(1-Curfew_Modifier)
    ## hours_lights_on-hours_lights_half_power results in 1 for hours with lights on, and curfew_modifier for half-powered hours
    Aggregate_Demand[:,0] += (hours_lights_on-hours_lights_half_power)*(np.ceil(Total_Street_Length/Light_Spacing)*Lights_Per_Side*Light_Power)


    '''     for i in range(len(Aggregate_Demand[:,0])): # OLD & Erroneous version of the curfew
            if i == 0:
                k = 1
            elif (i+1)/19.0 == mp.floor((i+1)/19.0):
                k = 1
            elif (i+1)/2.0 == mp.floor((i+1)/2.0):
                k = 2
            elif (i+1)/5.0 == mp.floor((i+1)/5.0):
                k = 1
            elif (i+1)/6.0 == mp.floor((i+1)/6.0):
                k = 0
            if k == 1:
                Aggregate_Demand[i,0] += np.ceil(Total_Street_Length/Light_Spacing)*Lights_Per_Side*Light_Power
            elif k == 2:
                Aggregate_Demand[i,0] += np.ceil(Total_Street_Length/Light_Spacing)*Lights_Per_Side*Light_Power*Curfew_Modifier
            else:
                Aggregate_Demand[i,0] += 0 '''

    # Save the loads at this point for use later
    Final_Demand = copy.deepcopy(Aggregate_Demand)
        
    # Now add in the previously calculated losses. For electricity, loss data is a combination of a 6% loss on average
    # in the U.S. and calculations by Mungkung, et al. on the percentage makeup of those losses at the low voltage level.
    Electrical_Loss = 0.8568*0.06   # Percentage
    ##################### MODIFIED: Changed to numpy array calculations--was a for loop before
    Aggregate_Demand[:,0] += Aggregate_Demand[:,0]*Electrical_Loss
    Aggregate_Demand[:,1] += Network_Heat_Loss
    Aggregate_Demand[:,2] += Network_Cool_Loss[i]
    """     for i in range(len(Aggregate_Demand[:,0])):
            Aggregate_Demand[i,0] = Aggregate_Demand[i,0]+Aggregate_Demand[i,0]*Electrical_Loss
            Aggregate_Demand[i,1] = Aggregate_Demand[i,1]+Network_Heat_Loss[i]
            Aggregate_Demand[i,2] = Aggregate_Demand[i,2]+Network_Cool_Loss[i] # Has NAN !!!!!!!!!! """

    # Collapse the heating, cooling, and electricity demand file to just heating or electricity based on chiller
    # Chiller_Hourly_Cooling_Results = []
    Chiller_COP_Results = np.zeros((8760)) ## MODIFIED for performance
    # Electrical_Demand = []
    Chiller_Costs = np.zeros((8760)) ## MODIFIED for performance

    Number_Iterations = 1
    Heat_Source_Temperature = 100 ## And this? IS it in deg C or F?? It's in deg F

    Engine_Demand = np.zeros((8760,2))
    # print('GOT TO THIS POINT!')

    for i in range(len(Aggregate_Demand[:,0])):
        # print(i)
        Hourly_Chiller_Result = Chiller_Types[Chiller_Var](Optimal_Cool_Hourly_Supply_Temps[i,Plant_Location_Var]*9/5+32, Hourly_Wet_Bulb[i]*9/5+32, Hourly_Temperature[i]*9/5+32, Aggregate_Demand[i,2], Number_Iterations, Heat_Source_Temperature)
        # Chiller_Hourly_Cooling_Results.append(Hourly_Chiller_Result[3])
        Chiller_COP_Results[i] = Hourly_Chiller_Result[4]
        Chiller_Costs[i] = Hourly_Chiller_Result[5]
        Engine_Demand[i,0] = Aggregate_Demand[i,0]+Hourly_Chiller_Result[1]
        Engine_Demand[i,1] = Aggregate_Demand[i,1]+Hourly_Chiller_Result[2]


#** TO CHANGE THE FOR LOOPS TO DIRECT NUMPY ARRAY HANDLING --> NEEDS MODIFYING THE CHILLER AND CHP PYTHON FILES
# =============================================================================
#     Hourly_Chiller_Result = Chiller_Types[Chiller_Var](Chilled_Water_Supply_Temperature, Hourly_Wet_Bulb*9/5+32, Hourly_Temperature*9/5+32, Aggregate_Demand[:,2], Number_Iterations, Heat_Source_Temperature)
#     Chiller_Hourly_Cooling_Results = Hourly_Chiller_Result[3]
#     Chiller_COP_Results = Hourly_Chiller_Result[4]
#     Chiller_Costs = Hourly_Chiller_Result[5]
#     Engine_Demand[:,0] = Aggregate_Demand[:,0]+Hourly_Chiller_Result[1]
#     Engine_Demand[:,1] = Aggregate_Demand[:,1]+Hourly_Chiller_Result[2]
# =============================================================================


    '''-----------------------------------------------------------------------------------------------'''
    # Calculate Power Production from any Solar Panels #
    '''-----------------------------------------------------------------------------------------------'''
    # Comm_Solar_Area = 0                   # m^2 # DEACTIVATED THE SOLAR FOR NOW # MODIFIED
    
    # Hourly_Solar_Generation = np.zeros((8760)) # MODIFIED: UNUSED
    # Hourly_Solar_Potential = np.zeros((8760)) # MODIFIED: UNUSED
    Excess_Electricity = np.zeros((8760))
    Capital_Solar_Cost = 0

    # DEACTIVATED THE SOLAR FOR NOW # MODIFIED
    # for i in range(Num_Sites):
    #     building = Site_Vars[i]
    #     if building != 0 and building != Num_Buildings+1: 
    #         Comm_Solar_Area += Comm_Solar_Var/100*Solar_Roof_Area[building]

    # DEACTIVATED THE SOLAR FOR NOW # MODIFIED
    # Calculate loads and suFbtract from total electrical demand; calculate costs and total solar capacity installed
    # for hour in range(8760):
    #     HC_Solar = Commercial_Solar_Types[Comm_Solar_Type_Var](hour, UTC, Comm_Solar_Area, Tilt, Azimuth, Latitude, Longitude, Hourly_DNI[hour], Hourly_DHI[hour], Hourly_GHI[hour], Hourly_Albedo[hour], Hourly_Temperature[hour], Hourly_Wind_Speed[hour], Altitude)
    #     # Hourly_Solar_Generation[hour] = HC_Solar[3]
    #     Engine_Demand[hour,0] = Engine_Demand[hour,0] - HC_Solar[3]
    #     if Engine_Demand[hour,0] < 0:
    #         Excess_Electricity[hour] = abs(Engine_Demand[hour,0])
    #         Engine_Demand[hour,0] = 0
    #     if hour == 0:
    #         Capital_Solar_Cost = HC_Solar[4]
        # Hourly_Solar_Potential[hour] = HC_Solar[0]

    # Save the loads with a difference name at this point for use later
    Post_Solar_Demand = copy.deepcopy(Engine_Demand)

    # Now run a control scheme that simply produces to the greatest demand and counts excess as waste. This is based on the 
    # Electric Equivalent Load Following control method published by Kavvadias, et al.
    Natural_Gas_Input_Results = np.zeros(8760)
    Biofuel_Input_Results = np.zeros(8760)
    Hydrogen_Input_Results = np.zeros(8760)
    # Grid_Fuel_Input_Results = np.zeros(8760) # OFFLINED # MODIFIED
    # Boiler_Fuel_Input_Results = np.zeros(8760) # OFFLINED # MODIFIED
    Hourly_Electricity_Results = np.zeros(8760)
    Maximum_Heat_Production = np.zeros(8760)
    CCHP_Capex = 0 # in $ # Modified # Added
    CCHP_Opex = 0
    Carbon_Emissions = np.zeros(8760)
    Last_Part_Load = np.zeros(Num_Engines)
    Last_Num_Engines = np.zeros(Num_Engines)
    Excess_Heat = np.zeros((8760))

## Modified # WRONG!! Since it doesn't consider the heat provided by TES at the peak hour, also it doesn't consider the parasitic consumption which might require more engines than what the load alone requires.
# =============================================================================
#     # Calculate how many engines are required
#     Max_Engine_Electricity = max(Engine_Demand[:,0])
#     Test_Max_Engine_Heat = max(Engine_Demand[:,1])*Power_to_Heat[Engine_Var]
#     Count_Engines = np.ceil(max(Max_Engine_Electricity/Max_Unit_Size[Engine_Var],Test_Max_Engine_Heat/Max_Unit_Size[Engine_Var]))
#     CCHP_Capex = Count_Engines*CHP_Capital_Cost[Engine_Var]
# =============================================================================
        
    for i in range(len(Engine_Demand[:,0])):
        Test_Electricity = Engine_Demand[i,1]*Power_to_Heat[Engine_Var]
        if Engine_Demand[i,0] > 0 and Engine_Demand[i,0] > Test_Electricity:
            if i == 0:
                Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[i]*9/5+32, Gas_Line_Pressure, Engine_Demand[i,0], 0, 0)#, Count_Engines)
                Last_Num_Engines = Hourly_Supply_Result[7]
                Last_Part_Load = Hourly_Supply_Result[8]
            else:
                Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[i]*9/5+32, Gas_Line_Pressure, Engine_Demand[i,0], Last_Num_Engines, Last_Part_Load)#, Count_Engines)
                Last_Num_Engines = Hourly_Supply_Result[7]
                Last_Part_Load = Hourly_Supply_Result[8]
                if CHP_Fuel_Type[Engine_Var] < 1.5:
                    Natural_Gas_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                elif CHP_Fuel_Type[Engine_Var] < 2.5:
                    Hydrogen_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                elif CHP_Fuel_Type[Engine_Var] < 3.5:
                    Biofuel_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                Hourly_Electricity_Results[i] += Hourly_Supply_Result[3]
                Maximum_Heat_Production[i] += Hourly_Supply_Result[2]
                CCHP_Opex += Hourly_Supply_Result[5]
                Carbon_Emissions[i] += Hourly_Supply_Result[6]
                Engine_Demand[i,0] -= Hourly_Supply_Result[3]
                if Engine_Demand[i,0] < 0:
                    Excess_Electricity[i] += abs(Engine_Demand[i,0])
                    Engine_Demand[i,0] = 0
                Engine_Demand[i,1] -= Hourly_Supply_Result[2]
                if Engine_Demand[i,1] < 0:
                    Excess_Heat[i] += abs(Engine_Demand[i,1])
                    Engine_Demand[i,1] = 0
        elif Engine_Demand[i,1] > 0 and Test_Electricity > Engine_Demand[i,0]:
            if i == 0:
                Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[i]*9/5+32, Gas_Line_Pressure, Test_Electricity, 0, 0)#, Count_Engines)
                Last_Num_Engines = Hourly_Supply_Result[7]
                Last_Part_Load = Hourly_Supply_Result[8]
            else:
                Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[i]*9/5+32, Gas_Line_Pressure, Test_Electricity, Last_Num_Engines, Last_Part_Load)#, Count_Engines)
                Last_Num_Engines = Hourly_Supply_Result[7]
                Last_Part_Load = Hourly_Supply_Result[8]
                if CHP_Fuel_Type[Engine_Var] < 1.5:
                    Natural_Gas_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                elif CHP_Fuel_Type[Engine_Var] < 2.5:
                    Hydrogen_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                elif CHP_Fuel_Type[Engine_Var] < 3.5:
                    Biofuel_Input_Results[i] += Hourly_Supply_Result[0]/kWh_to_Btu
                Hourly_Electricity_Results[i] += Hourly_Supply_Result[3]
                Maximum_Heat_Production[i] += Hourly_Supply_Result[2]
                CCHP_Capex = max(CCHP_Capex, Hourly_Supply_Result[4]) # Modified
                CCHP_Opex += Hourly_Supply_Result[5]
                Carbon_Emissions[i] += Hourly_Supply_Result[6]
                Engine_Demand[i,0] -= Hourly_Supply_Result[3]
                if Engine_Demand[i,0] < 0:
                    Excess_Electricity[i] += abs(Engine_Demand[i,0])
                    Engine_Demand[i,0] = 0
                Engine_Demand[i,1] -= Hourly_Supply_Result[2]
                if Engine_Demand[i,1] < 0:
                    Excess_Heat[i] += abs(Engine_Demand[i,1])
                    Engine_Demand[i,1] = 0
        else:
            pass

    # Rectify first hour
    Natural_Gas_Input_Results[0] = 0.0
    Hydrogen_Input_Results[0] = 0.0
    Biofuel_Input_Results[0] = 0.0
    Test_Electricity = Engine_Demand[0,1]*Power_to_Heat[Engine_Var]
    if Engine_Demand[0,0] > 0 and Engine_Demand[0,0] > Test_Electricity:
        Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[0]*9/5+32, Gas_Line_Pressure, Engine_Demand[0,0], Last_Num_Engines, Last_Part_Load)#, Count_Engines)
        Last_Num_Engines = Hourly_Supply_Result[7]
        Last_Part_Load = Hourly_Supply_Result[8]
        if CHP_Fuel_Type[Engine_Var] < 1.5:
            Natural_Gas_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        elif CHP_Fuel_Type[Engine_Var] < 2.5:
            Hydrogen_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        elif CHP_Fuel_Type[Engine_Var] < 3.5:
            Biofuel_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        Hourly_Electricity_Results[0] += Hourly_Supply_Result[3]
        Maximum_Heat_Production[0] += Hourly_Supply_Result[2]
        CCHP_Opex += Hourly_Supply_Result[5]
        Carbon_Emissions[0] += Hourly_Supply_Result[6]
        Engine_Demand[i,0] -= Hourly_Supply_Result[3]
        if Engine_Demand[i,0] < 0:
            Excess_Electricity[i] += abs(Engine_Demand[i,0])
            Engine_Demand[i,0] = 0
        Engine_Demand[i,1] -= Hourly_Supply_Result[2]
        if Engine_Demand[i,1] < 0:
            Excess_Heat[i] += abs(Engine_Demand[i,1])
            Engine_Demand[i,1] = 0
    elif Engine_Demand[0,1] > 0 and Test_Electricity > Engine_Demand[0,0]:
        Hourly_Supply_Result = CHP_Types[Engine_Var](Altitude, Hourly_Temperature[0]*9/5+32, Gas_Line_Pressure, Test_Electricity, Last_Num_Engines, Last_Part_Load)#, Count_Engines)
        Last_Num_Engines = Hourly_Supply_Result[7]
        Last_Part_Load = Hourly_Supply_Result[8]
        if CHP_Fuel_Type[Engine_Var] < 1.5:
            Natural_Gas_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        elif CHP_Fuel_Type[Engine_Var] < 2.5:
            Hydrogen_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        elif CHP_Fuel_Type[Engine_Var] < 3.5:
            Biofuel_Input_Results[0] += Hourly_Supply_Result[0]/kWh_to_Btu
        Hourly_Electricity_Results[0] += Hourly_Supply_Result[3]
        Maximum_Heat_Production[0] += Hourly_Supply_Result[2]
        CCHP_Opex += Hourly_Supply_Result[5]
        Carbon_Emissions[0] += Hourly_Supply_Result[6]
        Engine_Demand[i,0] -= Hourly_Supply_Result[3]
        if Engine_Demand[i,0] < 0:
            Excess_Electricity[i] += abs(Engine_Demand[i,0])
            Engine_Demand[i,0] = 0
        Engine_Demand[i,1] -= Hourly_Supply_Result[2]
        if Engine_Demand[i,1] < 0:
            Excess_Heat[i] += abs(Engine_Demand[i,1])
            Engine_Demand[i,1] = 0

    '''-----------------------------------------------------------------------------------------------'''
    # Calculate the values for constraints and objectives as well as other parameters of interest. #
    '''-----------------------------------------------------------------------------------------------'''
    # Calculate form-based values for constraints and the score of amenities that qualify toward the modified walkability index.
    # This is adapted from Rakha and Reinhart (http://static1.squarespace.com/static/53d65c30e4b0d86829f32e6f/t/53f3c509e4b06927b947aba6/1408484616999/SB12_TS04b_3_Rakha.pdf)
    # The categories have been reduced to Grocery, Restaurants, Shopping, and Schools. This excludes
    # Banks, Coffee, Books, and Entertainment since these are currently not included in the above
    # building types. Parks are added as a count of vacant areas (assuming these are parks). The process
    # is that a list is made for each residential building of the distances to all of the sites in the development.
    # Note that weights are included above even for amenity types that are not listed, but that amenity
    # lists are not yet included for the amenities that are not included. These would have to be added.
    # After this, the nearest ones are used to calculate the walkability index for each building. The 
    # overall index will be made up of the average of all of these and expanded to a 100 point scale.
    Total_GFA = 0
    Total_Buildings = 0
    Total_Res_Buildings = 0
    Total_Height = 0
    Total_Res_GFA = 0
    Total_Off_GFA = 0
    Total_Ret_GFA = 0
    Total_Sup_GFA = 0
    Total_Rest_GFA = 0
    Total_Edu_GFA = 0
    Total_Med_GFA = 0
    Total_Lod_GFA = 0
    Total_Ind_GFA = 0
    Total_Dwelling_Units = 0
    Total_Jobs = 0
    Walkability_Scores = []
    Annual_Rental_Revenue = 0
    Building_Construction_CapEx = 0

    for site in range(Num_Sites):
        Test_Bldg = Site_Vars[site]                                             # Grab the building at the site as long as it is not empty or the power plant
        Grocery_List = []                                                       # Make a list of all of the groceries in the development. The next lists are the same for different amenities.
        Restaurants_List = []
        Shopping_List = []
        Parks_List = []
        Schools_List = []
        # Grocery_Counter = Num_Countable_Grocery # OFFLINED # MODIFIED
        # Restaurants_Counter = Num_Countable_Restaurants # OFFLINED # MODIFIED
        # Shopping_Counter = Num_Countable_Shopping # OFFLINED # MODIFIED
        # Parks_Counter = Num_Countable_Parks # OFFLINED # MODIFIED
        # Schools_Counter = Num_Countable_Schools # OFFLINED # MODIFIED
        Test_Walkability_Score = 0
        if Test_Bldg != 0 and Test_Bldg != Num_Buildings+1:
            Total_GFA += GFA[Test_Bldg]                                         # Add values to all of the constraints
            Total_Buildings += 1
            Total_Height += Floor_Height[Test_Bldg]*Stories[Test_Bldg]
            Total_Res_GFA += Res_GFA[Test_Bldg]
            Total_Off_GFA += Off_GFA[Test_Bldg]
            Total_Ret_GFA += Ret_GFA[Test_Bldg]
            Total_Sup_GFA += Sup_GFA[Test_Bldg]
            Total_Rest_GFA += Rest_GFA[Test_Bldg]
            Total_Edu_GFA += Edu_GFA[Test_Bldg]
            Total_Med_GFA += Med_GFA[Test_Bldg]
            Total_Lod_GFA += Lod_GFA[Test_Bldg]
            Total_Ind_GFA += Ind_GFA[Test_Bldg]
            Total_Dwelling_Units += Dwelling_Units[Test_Bldg]
            Total_Jobs += Jobs[Test_Bldg]
            Annual_Rental_Revenue += Building_Revenue[Test_Bldg]
            Building_Construction_CapEx += Building_Capex[Test_Bldg]
            
            if Res_GFA[Test_Bldg] > 0:
                Total_Res_Buildings += 1
                for test_site in range(Num_Sites):
                    Walk_Bldg = Site_Vars[test_site]
                    Walk_Distance = (abs(Site_Coordinates[site,0]-Site_Coordinates[test_site,0])+abs(Site_Coordinates[site,1]-Site_Coordinates[test_site,1]))*meters_to_ft/5280         # miles
                    if Walk_Bldg == Num_Buildings+1:                            # Exclude the power plant
                        continue
                    elif Walk_Bldg == 0:                                        # Parks are denoted by empty land
                        Parks_List.append(Walk_Distance)
                        continue
                    else:
                        if Sup_GFA[Walk_Bldg] > 0:                              # For everything else add the distance if the amenity exists
                            Grocery_List.append(Walk_Distance)
                        if Rest_GFA[Walk_Bldg] > 0:
                            Restaurants_List.append(Walk_Distance)
                        if Ret_GFA[Walk_Bldg] > 0:
                            Shopping_List.append(Walk_Distance)
                        if Edu_GFA[Walk_Bldg] > 0:
                            Schools_List.append(Walk_Distance)
                
                Grocery_List.sort()                                             # Sort the distances in ascending order, meaning from closest to the Test Building to the farthest
                Restaurants_List.sort()
                Shopping_List.sort()
                Parks_List.sort()
                Schools_List.sort()
                
                if len(Grocery_List) > Num_Countable_Grocery:                   # Now for each list add the walk score of the list in turn
                    for g in range(Num_Countable_Grocery):
                        if Grocery_List[g] < 0.25:
                            Test_Walkability_Score += Grocery_Weights[g]
                            continue
                        elif Grocery_List[g] < 1:
                            Test_Walkability_Score += Grocery_Weights[g]*(Walk_Weight_1_m*Grocery_List[g]+Walk_Weight_1_b)
                            continue
                        elif Grocery_List[g] < 2:
                            Test_Walkability_Score += Grocery_Weights[g]*(Walk_Weight_2_m*Grocery_List[g]+Walk_Weight_2_b)
                        else:
                            continue
                else:
                    for g in range(len(Grocery_List)):
                        if Grocery_List[g] < 0.25:
                            Test_Walkability_Score += Grocery_Weights[g]
                            continue
                        elif Grocery_List[g] < 1:
                            Test_Walkability_Score += Grocery_Weights[g]*(Walk_Weight_1_m*Grocery_List[g]+Walk_Weight_1_b)
                            continue
                        elif Grocery_List[g] < 2:
                            Test_Walkability_Score += Grocery_Weights[g]*(Walk_Weight_2_m*Grocery_List[g]+Walk_Weight_2_b)
                        else:
                            continue
                        
                if len(Restaurants_List) > Num_Countable_Restaurants:                   # Now for each list add the walk score of the list in turn
                    for r in range(Num_Countable_Restaurants):
                        if Restaurants_List[r] < 0.25:
                            Test_Walkability_Score += Restaurants_Weights[r]
                            continue
                        elif Restaurants_List[r] < 1:
                            Test_Walkability_Score += Restaurants_Weights[r]*(Walk_Weight_1_m*Restaurants_List[r]+Walk_Weight_1_b)
                            continue
                        elif Restaurants_List[r] < 2:
                            Test_Walkability_Score += Restaurants_Weights[r]*(Walk_Weight_2_m*Restaurants_List[r]+Walk_Weight_2_b)
                        else:
                            continue
                else:
                    for r in range(len(Restaurants_List)):
                        if Restaurants_List[r] < 0.25:
                            Test_Walkability_Score += Restaurants_Weights[r]
                            continue
                        elif Restaurants_List[r] < 1:
                            Test_Walkability_Score += Restaurants_Weights[r]*(Walk_Weight_1_m*Restaurants_List[r]+Walk_Weight_1_b)
                            continue
                        elif Restaurants_List[r] < 2:
                            Test_Walkability_Score += Restaurants_Weights[r]*(Walk_Weight_2_m*Restaurants_List[r]+Walk_Weight_2_b)
                        else:
                            continue
                        
                if len(Shopping_List) > Num_Countable_Shopping:                   # Now for each list add the walk score of the list in turn
                    for s in range(Num_Countable_Shopping):
                        if Shopping_List[s] < 0.25:
                            Test_Walkability_Score += Shopping_Weights[s]
                            continue
                        elif Shopping_List[s] < 1:
                            Test_Walkability_Score += Shopping_Weights[s]*(Walk_Weight_1_m*Shopping_List[s]+Walk_Weight_1_b)
                            continue
                        elif Shopping_List[s] < 2:
                            Test_Walkability_Score += Shopping_Weights[s]*(Walk_Weight_2_m*Shopping_List[s]+Walk_Weight_2_b)
                        else:
                            continue
                else:
                    for s in range(len(Shopping_List)):
                        if Shopping_List[s] < 0.25:
                            Test_Walkability_Score += Shopping_Weights[s]
                            continue
                        elif Shopping_List[s] < 1:
                            Test_Walkability_Score += Shopping_Weights[s]*(Walk_Weight_1_m*Shopping_List[s]+Walk_Weight_1_b)
                            continue
                        elif Shopping_List[s] < 2:
                            Test_Walkability_Score += Shopping_Weights[s]*(Walk_Weight_2_m*Shopping_List[s]+Walk_Weight_2_b)
                        else:
                            continue
                        
                if len(Parks_List) > Num_Countable_Parks:                   # Now for each list add the walk score of the list in turn
                    for p in range(Num_Countable_Parks):
                        if Parks_List[p] < 0.25:
                            Test_Walkability_Score += Parks_Weights[p]
                            continue
                        elif Parks_List[p] < 1:
                            Test_Walkability_Score += Parks_Weights[p]*(Walk_Weight_1_m*Parks_List[p]+Walk_Weight_1_b)
                            continue
                        elif Parks_List[p] < 2:
                            Test_Walkability_Score += Parks_Weights[p]*(Walk_Weight_2_m*Parks_List[p]+Walk_Weight_2_b)
                        else:
                            continue
                else:
                    for p in range(len(Parks_List)):
                        if Parks_List[p] < 0.25:
                            Test_Walkability_Score += Parks_Weights[p]
                            continue
                        elif Parks_List[p] < 1:
                            Test_Walkability_Score += Parks_Weights[p]*(Walk_Weight_1_m*Parks_List[p]+Walk_Weight_1_b)
                            continue
                        elif Parks_List[p] < 2:
                            Test_Walkability_Score += Parks_Weights[p]*(Walk_Weight_2_m*Parks_List[p]+Walk_Weight_2_b)
                        else:
                            continue
                        
                if len(Schools_List) > Num_Countable_Schools:                   # Now for each list add the walk score of the list in turn
                    for s in range(Num_Countable_Schools):
                        if Schools_List[s] < 0.25:
                            Test_Walkability_Score += Schools_Weights[s]
                            continue
                        elif Schools_List[s] < 1:
                            Test_Walkability_Score += Schools_Weights[s]*(Walk_Weight_1_m*Schools_List[s]+Walk_Weight_1_b)
                            continue
                        elif Schools_List[s] < 2:
                            Test_Walkability_Score += Schools_Weights[s]*(Walk_Weight_2_m*Schools_List[s]+Walk_Weight_2_b)
                        else:
                            continue
                else:
                    for s in range(len(Schools_List)):
                        if Schools_List[s] < 0.25:
                            Test_Walkability_Score += Schools_Weights[s]
                            continue
                        elif Schools_List[s] < 1:
                            Test_Walkability_Score += Schools_Weights[s]*(Walk_Weight_1_m*Schools_List[s]+Walk_Weight_1_b)
                            continue
                        elif Schools_List[s] < 2:
                            Test_Walkability_Score += Schools_Weights[s]*(Walk_Weight_2_m*Schools_List[s]+Walk_Weight_2_b)
                        else:
                            continue

                Walkability_Scores.append(Test_Walkability_Score)               # Add the walkability score to the list of all walkability scores in the development
    
    # Now create the total walkability score
    Walkability_Multiplier = 100/(sum(Grocery_Weights)+sum(Restaurants_Weights)+sum(Shopping_Weights)+sum(Parks_Weights)+sum(Schools_Weights))
    if Total_Res_Buildings > 0:
        Total_Walkability_Score = np.nan_to_num(Walkability_Multiplier*(sum(Walkability_Scores)/Total_Res_Buildings))
    else:
        Total_Walkability_Score = 0

    # Now find the remaining values for constraints
    Site_FAR = np.nan_to_num(Total_GFA/Total_Site_GFA)
    Average_Height = np.nan_to_num(Total_Height/Total_Buildings)
    Res_Percent =  np.nan_to_num(Total_Res_GFA/Total_GFA)
    Off_Percent =  np.nan_to_num(Total_Off_GFA/Total_GFA)
    Ret_Percent =  np.nan_to_num(Total_Ret_GFA/Total_GFA)
    Sup_Percent =  np.nan_to_num(Total_Sup_GFA/Total_GFA)
    Rest_Percent =  np.nan_to_num(Total_Rest_GFA/Total_GFA)
    Edu_Percent =  np.nan_to_num(Total_Edu_GFA/Total_GFA)
    Med_Percent =  np.nan_to_num(Total_Med_GFA/Total_GFA)
    Lod_Percent =  np.nan_to_num(Total_Lod_GFA/Total_GFA)
    Ind_Percent =  np.nan_to_num(Total_Ind_GFA/Total_GFA)

    # To find Final Demand, add the actual heat or electricity used to do real cooling work
    # on the buildings and calculate total hourly demand and efficiency, and keep a running
    # total of fuel use and demand.
    Useful_Demand = np.zeros((8760))
    # Hourly_Efficiency = np.zeros((8760)) # MODIFIED: UNUSED
    # Hourly_CHP_Efficiency = np.zeros((8760)) # MODIFIED: UNUSED
    Total_Demand = 0.0
    CHP_Demand = 0.0
    Total_Fuel = 0.0
    CHP_Fuel = 0.0
    

    ## FOR LOOPS converted to DIRECT NUMPY ARRAY MANIPULATIONS
    COP_Flag = (Chiller_COP_Results > 1.0) ## WHY?
    Useful_Demand = (Final_Demand[:,0] + Final_Demand[:,1] +
                     COP_Flag * Final_Demand[:,2]/Chiller_COP_Results +
                     (1-COP_Flag) * Final_Demand[:,2])
#    if Total_Buildings > 0: ## MODIFIED: UNUSED
#        Hourly_Efficiency = np.nan_to_num(Useful_Demand/Fuel_Input_Results)   ## Added the nan_to_number ## UNUSED
#    else:
#        Hourly_Efficiency = 0
    Total_Demand = np.sum(Useful_Demand)
    CHP_Demand = np.sum(Post_Solar_Demand[:,0] + Post_Solar_Demand[:,1]) # + Hourly_GW_Treated*FO_MD_Power_per_m3/1000) ## ADDED ## REMOVED!
    Total_Fuel = np.sum(Natural_Gas_Input_Results+Hydrogen_Input_Results+Biofuel_Input_Results) ## MODIFIED

    CHP_Fuel = np.sum(Natural_Gas_Input_Results+Hydrogen_Input_Results+Biofuel_Input_Results) ## ADDED

### MODIFIED: FOR LOOPS REPLACED BY NP ARRAY MANIPULATIONS ABOVE
    """  for i in range(8760):
        if Chiller_COP_Results[i] > 1.0:
            Useful_Demand[i] = Final_Demand[i,0]+Final_Demand[i,1]+Final_Demand[i,2]/Chiller_COP_Results[i]
        else:
            Useful_Demand[i] = Final_Demand[i,0]+Final_Demand[i,1]+Final_Demand[i,2]
        Total_Demand += Useful_Demand[i]
        CHP_Demand += Post_Solar_Demand[i,0]+Post_Solar_Demand[i,1]
        # Total_Fuel += Hourly_Solar_Potential[i]+Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i] # MODIFIED: dropped the solar potential, it didn't make sense to have it in
        Total_Fuel += Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i]
        CHP_Fuel += Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i]
        # Hourly_Efficiency[i] = np.nan_to_num(Useful_Demand[i]/(Hourly_Solar_Potential[i]+Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i])) # MODIFIED: dropped the solar potential, it didn't make sense to have it in
        Hourly_Efficiency[i] = np.nan_to_num(Useful_Demand[i]/(Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i]))
        Hourly_CHP_Efficiency[i] = np.nan_to_num((Post_Solar_Demand[i,0]+Post_Solar_Demand[i,1])/(Natural_Gas_Input_Results[i]+Hydrogen_Input_Results[i]+Biofuel_Input_Results[i])) """
        
    # Overall_Efficiency = np.nan_to_num(Total_Demand/Total_Fuel) # MODIFIED
    # Overall_CHP_Efficiency = np.nan_to_num(CHP_Demand/CHP_Fuel) # MODIFIED
    if Total_Buildings > 0: ## REFER TO PAGE 79 of the thesis for the explanation
        Overall_Efficiency = np.nan_to_num(Total_Demand/Total_Fuel) ## Added the nan_to_number ## Note: total demand = raw demand + solar power and neglecting the excess electricity from PVs and CHP
        Overall_CHP_Efficiency = np.nan_to_num(CHP_Demand/CHP_Fuel)
    else:
        Overall_Efficiency = 0
        Overall_CHP_Efficiency = 0
    
    # Now count up the carbon emissions. In this case we will say that any hour where the grid
    # buy price is zero you cannot sell back to the grid. If the price is greater than 0, then 
    # anything sold back to the grid offsets the emissions in that hour for the kWh that are offset.
    # Since the carbon emissions have been counted already for all of combustion in the generators, 
    # the value of grid emissions can be subtracted from the current hour emissions.
    """     Total_Carbon_Emissions = 0.0
        for i in range(8760):
            if Sell_Price[i] > 0:
                Carbon_Emissions[i] -= Grid_Emissions[i]*(Excess_Electricity[i]/1000)*MT_to_lbs
                Total_Carbon_Emissions += Carbon_Emissions[i]/MT_to_lbs """
    Total_Carbon_Emissions = np.sum(Carbon_Emissions)/MT_to_lbs ## MODIFIED: not accounting for the negative carbon emissions considered for selling E back to the grid.
             
    # Now find the excess characteristics of the CHP plant
    Total_Excess_Electricity = sum(Excess_Electricity)
    Total_Excess_Heat = sum(Excess_Heat)
    
    # Now find the total costs of the development
    Total_CapEx = CCHP_Capex + max(Chiller_Costs) + Capital_Solar_Cost + Optimal_Heat_Pipe_Capital_Cost + Optimal_Cool_Pipe_Capital_Cost + Optimal_Trench_Capital_Cost + Building_Construction_CapEx
    Total_Annual_OpEx = CCHP_Opex-Annual_Rental_Revenue-sum(Sell_Price*Excess_Electricity)
    LCC =  copy.deepcopy(Total_CapEx)
    for i in range(Project_Life):
        LCC += Total_Annual_OpEx/(1+Discount_Rate)**i


    ## ADDED/MODIFIED: Accounting for construction carbon emissions based on LCA-EIO inputs + calculating SCC
    Total_Construction_Carbon = (Total_CapEx / USD_2007_to_2019)/10**6 * Const_Carbon_per_Mil_Dollar # in metric tons CO2 eq # For recording
    
    # Years = np.arange(Current_Year, Current_Year+Project_Life)
    # Annual_SCC = 0.8018*Years - 1585.7 # in 2007 $ per metric tons of CO2

    # SCC = (Construction_Carbon * Annual_SCC[0] + np.sum(Annual_SCC * Total_Carbon_Emissions)) * USD_2007_to_2019 # in 2019 $
    # Total_SCC = (Total_Construction_Carbon * Annual_SCC[0] + np.sum(Annual_SCC * Total_Carbon_Emissions)) * USD_2007_to_2019 # ADDED/ MODIFIED # in 2019 $
    Total_Carbon = Project_Life * Total_Carbon_Emissions + Total_Construction_Carbon # ADDED/ MODIFIED in metric tons CO2e


    Internal_Stop = timeit.default_timer()
    Internal_Time = Internal_Stop-Internal_Start
    
    # print(Internal_Time)
    
    Run_Result = np.zeros((1,Vars_Plus_Output))
    # Add the variables first
    for i in range(Num_Sites):
        Run_Result[0][i] = Site_Vars[i]
    Run_Result[0][Num_Sites] = Plant_Location_Var
    Run_Result[0][Num_Sites+1] = Engine_Var
    Run_Result[0][Num_Sites+2] = Chiller_Var
    Run_Result[0][Num_Sites+3] = 0 # Comm_Solar_Type_Var # MODIFIED
    Run_Result[0][Num_Sites+4] = 0 # Comm_Solar_Var # MODIFIED
    Run_Result[0][Num_Sites+5] = Heat_Sup_Temp_Var
    Run_Result[0][Num_Sites+6] = Heat_Sup_Temp_Reset_Var
    Run_Result[0][Num_Sites+7] = Cool_Sup_Temp_Var
    Run_Result[0][Num_Sites+8] = Cool_Sup_Temp_Reset_Var
    # Now the objectives
    Run_Result[0][Num_Sites+9] = Overall_Efficiency
    Run_Result[0][Num_Sites+10] = Overall_CHP_Efficiency
    Run_Result[0][Num_Sites+11] = LCC
    Run_Result[0][Num_Sites+12] = Total_CapEx
    Run_Result[0][Num_Sites+13] = Total_Annual_OpEx
    Run_Result[0][Num_Sites+14] = Total_Carbon # MODIFIED FROM Total_Carbon_Emissions
    Run_Result[0][Num_Sites+15] = Total_Walkability_Score
    # Now the constraint values
    Run_Result[0][Num_Sites+16] = Total_GFA
    Run_Result[0][Num_Sites+17] = Site_FAR
    Run_Result[0][Num_Sites+18] = Average_Height
    Run_Result[0][Num_Sites+19] = Res_Percent
    Run_Result[0][Num_Sites+20] = Off_Percent
    Run_Result[0][Num_Sites+21] = Ret_Percent
    Run_Result[0][Num_Sites+22] = Sup_Percent
    Run_Result[0][Num_Sites+23] = Rest_Percent
    Run_Result[0][Num_Sites+24] = Edu_Percent
    Run_Result[0][Num_Sites+25] = Med_Percent
    Run_Result[0][Num_Sites+26] = Lod_Percent
    Run_Result[0][Num_Sites+27] = Ind_Percent
    # Now add the watch variables
    Run_Result[0][Num_Sites+28] = Total_Dwelling_Units
    Run_Result[0][Num_Sites+29] = Total_Jobs
    Run_Result[0][Num_Sites+30] = Total_Excess_Electricity
    Run_Result[0][Num_Sites+31] = Total_Excess_Heat
    Run_Result[0][Num_Sites+32] = Internal_Time
    ## **** NEED TO NORMALIZE AND CORRECT THE OBJECTIVES AND CONSTRAINTS FOLLOWING CH3_SF_CASESTUDY_W_STORAGE.PY ****
    return ((LCC/Total_GFA, Total_Carbon/Total_GFA, Total_Walkability_Score, ), (Max_GFA-Total_GFA, Max_FAR-Site_FAR, Max_Average_Height-Average_Height, Max_Res-Res_Percent, Max_Off-Off_Percent, Max_Ret-Ret_Percent, Max_Sup-Sup_Percent, Max_Rest-Rest_Percent, Max_Edu-Edu_Percent, Max_Med-Med_Percent, Max_Lod-Lod_Percent, Max_Ind-Ind_Percent, Res_Percent-Min_Res, Off_Percent-Min_Off, Ret_Percent-Min_Ret, Sup_Percent-Min_Sup, Rest_Percent-Min_Rest, Edu_Percent-Min_Edu, Med_Percent-Min_Med, Lod_Percent-Min_Lod, Ind_Percent-Min_Ind, ), Run_Result)

'''-----------------------------------------------------------------------------------------------'''
# Instantiate the optimization. #
'''-----------------------------------------------------------------------------------------------'''
Samples = pd.lhs(len(High_Seq), samples = Population_Size)
for i in range(len(Samples)):
    for j in range(len(Samples[0])):
        Samples[i,j] = np.round(Samples[i,j]*High_Seq[j])
        if Samples[i,j] < Low_Seq[j]:
            Samples[i,j] = Low_Seq[j]
        elif Samples[i,j] > High_Seq[j]:
            Samples[i,j] = High_Seq[j]
        else:
            continue
        
# seedfile = '/afs/ir.stanford.edu/users/r/o/robbest/Research/RQ2/IntInfraLocOpt_seed_288.json'
seedfile = 'IntInfraLocOpt_seed_288.json'
with open(seedfile, 'w') as outfile:
    json.dump(Samples.tolist(), outfile)

creator.create("IntInfraPlanOpt", fitness_with_constraints.FitnessWithConstraints, weights=(-1.0, -1.0, 1.0))    # Update based on number of objectives
creator.create("Individual", list, fitness=creator.IntInfraPlanOpt)

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    contents = json.load(open(filename, 'r'))
    return pcls(ind_init(c) for c in contents)

toolbox = base.Toolbox()

toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, seedfile)

def evaluateInd(individual):
    #print('individual %s' %individual)
    result = SupplyandDemandOptimization(individual)
    return result

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutPolynomialBoundedInt.mutPolynomialBoundedInt, eta=Eta, low=Low_Seq, up=High_Seq, indpb=Mutation_Probability)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluateInd)
toolbox.register("map", futures.map)

hof = tools.ParetoFront()
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
logbook = tools.Logbook()


def main():
    pop = toolbox.population_guess()
    global Results
    Results = np.zeros((1, Vars_Plus_Output))
    Results = [Results]

    # Unregister unpicklable methods before sending the toolbox.
    toolbox.unregister("individual_guess")
    toolbox.unregister("population_guess")

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    evaluate_result = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, evaluate_result):
        ind.fitness.values = fit[0]
        ind.fitness.cvalues = fit[1]
        ind.fitness.n_constraints = len(fit[1])
        Results = np.append(Results, [fit[2]], axis=0)

    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen= -1, evals=len(invalid_ind), **record)

    for g in range(Number_Generations):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rp.random() < Crossover_Probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.fitness.cvalues = fit[1]
            ind.fitness.n_constraints = len(fit[1])
            Results = np.append(Results, [fit[2]], axis=0)

        pop = toolbox.select(pop + offspring, Population_Size)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        hof.update(pop)

    global logs
    logs = np.zeros(shape=(Number_Generations, 6))
    gens = logbook.select("gen")
    evalnum = logbook.select("evals")
    avgs = logbook.select("avg")
    stds = logbook.select("std")
    mins = logbook.select("min")
    maxs = logbook.select("max")

    for g in range(Number_Generations):
        logs[g,0] = gens[g]+1
        logs[g,1] = evalnum[g]
        logs[g,2] = avgs[g]
        logs[g,3] = stds[g]
        logs[g,4] = mins[g]
        logs[g,5] = maxs[g]


if __name__ == "__main__":
    main()
    TestRuns = Results[:,0,:] ## Summarized (MODIFIED)
#    TestRuns = np.zeros((len(Results), len(Results[0][0])))
#    for i in range(len(TestRuns)):
#        for j in range(len(TestRuns[0])):
#            TestRuns[i,j] = Results[i][0][j]
    full_hof = np.zeros((len(hof),len(Results[0][0])))
    test_hof = np.zeros((len(hof), len(hof[0])))
    for i in range(len(hof)):
        test_hof[i] = hof[i]
    for i in range(len(test_hof)):
        for j in range(len(test_hof[0])):
            test_hof[i,j] = int(test_hof[i,j])
    Mod_Results = TestRuns[:,:len(hof[0])]
    for i in range(len(Mod_Results)):
        for j in range(len(Mod_Results[0])):
            Mod_Results[i,j] = int(Mod_Results[i,j])
    for i in range(len(hof)):
        for j in range(len(Mod_Results)):
            if cmp(Mod_Results[j].tolist(), test_hof[i].tolist()) == 0:
                full_hof[i] = TestRuns[j,:]
                break
# ============================OLD CODE========================================
#     main()
#     TestRuns = np.zeros((len(Results), len(Results[0][0])))
#     for i in range(len(TestRuns)):
#         for j in range(len(TestRuns[0])):
#             TestRuns[i,j] = Results[i][0][j]
#     full_hof = []
#     Mod_Results = Results[:,0:len(hof[0])]
#     for i in range(len(hof)):
#         full_hof += [0]
#         for j in range(len(Mod_Results)):
#             Test_Strip = Mod_Results[j].tolist()
#             if reduce(lambda v1,v2: v1 and v2, map(lambda v:v in hof[i], Test_Strip)) == True:
#                 Full_Result = Results[j].tolist()
#                 full_hof[i] = hof[i]+Full_Result[-7:]
#                 break
# =============================================================================

    np.savetxt("results/IILP_Toy_Optimization_Full_HOF.txt", full_hof, fmt ='%f', delimiter = ',')
    np.savetxt("results/IILP_Toy_Optimization_TestRuns.txt", TestRuns[1:], fmt='%f', delimiter = ',')
    np.savetxt("results/IILP_Toy_Optimization_Logbook.txt", logs, fmt='%f', delimiter = ',')
    np.savetxt("results/IILP_Toy_Optimization_HOF.txt", hof, fmt='%f', delimiter = ',')

Stop = timeit.default_timer()

# Time = Stop-Start # OFFLINED # MODIFIED
# print(Time) # OFFLINED # MODIFIED
