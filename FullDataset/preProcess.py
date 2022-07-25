import pandas as pds
import datetime
import numpy as np
import event as evt
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import os

def getyeardata(years,data):
    for i, year in enumerate(years):
        if i == 0:
            result = data[data.index.year == year]
        else:
            result = pds.concat([result, data[data.index.year == year]], sort=True)
    return result.sort_index()

def printpercentage(y):
    
    return(np.sum(y)/len(y))

def getdatas(train,test,val,data_scaled,truelabel):
    X_test = getyeardata(test,data_scaled)
    Y_test = getyeardata(test,truelabel)
    
    X_val = getyeardata(val,data_scaled)
    Y_val = getyeardata(val,truelabel)
    
    X_train = getyeardata(train,data_scaled)
    Y_train = getyeardata(train,truelabel)
    
    return X_test, Y_test, X_val, Y_val, X_train, Y_train

def getleaveoneout(test,val):
    
    years = [1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    
    train = [x for x in years if ((x not in test) and (x not in val))]
    test = [x for x in test]
    val = [x for x in val]
    
    return test, val, train

def getbalancedsplit(split, cat):
    
    if cat == 'nguyen':
         
        part1 = [2001,2004,2007]
        part2 = [2000,2006,2010]
        part3 = [1998,2002,2008]
        part4 = [2009,2011,2012]
        part5 = [1999,2003,2005]
        part6 = [2013,2014,2015]
        
    if cat == 'chinchilla':
        part1 = [2000,2005,2008]
        part2 = [1998,2007,2009]
        part3 = [2003,2004,2012]
        part4 = [2001,2010,2014]
        part5 = [1999,2002,2006]
        part6 = [2011,2013,2015]
    
    if split == 1:

        test = part1
        val = part2
        train = list(set(part3 + part4 + part5 + part6))

    if split == 2:

        test = part2
        val = part3
        train = list(set(part4 + part5 + part6 + part1))
        
    if split == 3:

        test = part3
        val = part4
        train = list(set(part5 + part6 + part1 + part2))
    
    if split == 4:

        test = part4
        val = part5
        train = list(set(part6 + part1 + part2 + part3))
        
    if split == 5:

        test = part5
        val = part6
        train = list(set(part1 + part2 + part3 + part4))
        
    if split == 6:
        
        test = part6
        val = part1
        train = list(set(part2 + part3 + part4 + part5))
        
    return test, val, train

def getsplit(split):
    if split == 1:

        test = [2013,2014,2015]
        val = [2010,2011,2012]
        train = [1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]

    if split == 5:

        test = [2001,2002,2003]
        val = [1998,1999,2000]
        train = [1997,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]

    if split == 4:

        test = [2004,2005,2006]
        val = [2001,2002,2003]
        train = [1997,1998,1999,2000,2007,2008,2009,2010,2011,2012,2013,2014,2015]

    if split == 3:

        test = [2007,2008,2009]
        val = [2004,2005,2006]
        train = [1997,1998,1999,2000,2001,2002,2003,2010,2011,2012,2013,2014,2015]

    if split == 2:

        test = [2010,2011,2012]
        val = [2007,2008,2009]
        train = [1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2013,2014,2015]

    if split == 6:

        test = [1998,1999,2000]
        val = [2013,2014,2015]
        train = [1997,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]
        
    return test, val, train

def clearempties(evtlist, data):
    
    evtlistnew = []

    for i in evtlist:
        if len(data[i.begin:i.end]) > 6:
            evtlistnew.append(i)
            
    return evtlistnew

def get_truelabel(data,events):
    
    x = pds.to_datetime(data.index)
    y = np.zeros(np.shape(data)[0])
    
    for e in events:
        n_true = np.where((x >= e.begin) & (x <= e.end))
        y[n_true] = 1
    
    label = pds.DataFrame(y, index = data.index, columns = ['label'])
    
    return label

def make_views(arr,win_size,step_size,writeable = False):
    """
    arr: any 2D array whose columns are distinct variables and 
    rows are data records at some timestamp t
    win_size: size of data window (given in data points along record/time axis)
    step_size: size of window step (given in data point along record/time axis)
    writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  
    Note that step_size is related to window overlap (overlap = win_size - step_size), in 
    case you think in overlaps.
  
    This function can work with C-like and F-like arrays, and with DataFrames.  Yay.
    """
  
    # If DataFrame, use only underlying NumPy array
    if type(arr) == type(pds.DataFrame()):
        arr['index'] = arr.index
        arr = arr.values
  
    # Compute Shape Parameter for as_strided
    n_records = arr.shape[0]
    n_columns = arr.shape[1]
    remainder = (n_records - win_size) % step_size 
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    shape = (num_windows, win_size, n_columns)
  
    # Compute Strides Parameter for as_strided
    next_win = step_size * arr.strides[0]
    next_row, next_col = arr.strides
    strides = (next_win, next_row, next_col)

    new_view_structure = as_strided(arr,shape = shape,strides = strides,writeable = writeable)
    return new_view_structure


def savetofolder(path, name, data,C):
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for i in data:
        df = pds.DataFrame(i)
        df = df.set_index(df[C])
        df = df.iloc[: , :-1]
        df.to_csv(path + name + str(count) + '.csv', header = None)
        count = count+1
        
def saveYtofolder(path,name,data):
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for i in data:
        df = pds.DataFrame(i)
        df = df.set_index(df[1])
        df = df.iloc[: , :-1]
        df.to_csv(path + name + str(count) + '.csv', header = None)
        count = count+1