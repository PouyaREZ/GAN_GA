import math as mp

def RemoveHeaders(Demand_Array):
    ''' Removes the headers from an array of hourly demand values if any are present. THe input is an array of
        demand values and the output is an array stripped of all leading lines that do not contain numeric
        values.
    '''
    if mp.isnan(Demand_Array[0,-1]) == 1:
        Demand_Array = Demand_Array[1:]
        Demand_Array = RemoveHeaders(Demand_Array)

    if mp.isnan(Demand_Array[-1,0]) == 1:
        Demand_Array = Demand_Array[:, 1:]
        Demand_Array = RemoveHeaders(Demand_Array)

    return Demand_Array