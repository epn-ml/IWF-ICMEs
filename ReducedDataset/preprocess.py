import pandas as pds
import datetime
import numpy as np
import event as evt
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

def compare_lists(dicoflists):
    '''
    compare number of events in lists
    '''
    plt.figure()
    years = ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
    df = pds.DataFrame(columns = dicoflists, index = years)
    for lists in dicoflists:
        for year in years:
            df[lists][year] = len([x for x in dicoflists[lists] if (str(x.begin.year)==year)])
    ax = df.plot()
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('Year')
    
    print(df)

def getyeardata(years,data):
    '''
    get data for specific years
    '''
    for i, year in enumerate(years):
        if i == 0:
            result = data[data.index.year == year]
        else:
            result = pds.concat([result, data[data.index.year == year]], sort=True)
    return result.sort_index()

def printpercentage(y):
    '''
    print percentage of positive labels
    '''
    return(np.sum(y)/len(y))

def getdatas(train,test,val,data_scaled,truelabel):
    '''
    get split dataset
    '''
    
    X_test = getyeardata(test,data_scaled)
    Y_test = getyeardata(test,truelabel)
    
    X_val = getyeardata(val,data_scaled)
    Y_val = getyeardata(val,truelabel)
    
    X_train = getyeardata(train,data_scaled)
    Y_train = getyeardata(train,truelabel)
    
    return X_test, Y_test, X_val, Y_val, X_train, Y_train
    

def getbalancedsplit(split, cat):
    '''
    get balanced splits for each catalog
    '''
    if cat == 'wind':
         
        part1 = [2008,2012,2016]
        part2 = [2007,2009,2011]
        part3 = [2013,2014,2018]
        part4 = [2010,2015,2017,2019]
        
        
    if cat == 'stereoa':
        part1 = [2009,2013,2015]
        part2 = [2008,2010,2014,2018]
        part3 = [2007,2012,2017]
        part4 = [2011,2016,2019]
        
    if cat == 'stereob':
        part1 = [2007,2012]
        part2 = [2008,2013]
        part3 = [2009,2011]
        part4 = [2010,2014]

    if split == 1:

        test = part1
        val = part2
        train = list(set(part3 + part4))

    if split == 2:

        test = part2
        val = part3
        train = list(set(part4 + part1))
        
    if split == 3:

        test = part3
        val = part4
        train = list(set(part1 + part2))
    
    if split == 4:

        test = part4
        val = part1
        train = list(set(part2 + part3))
        
    return test, val, train


def clearempties(evtlist, data):
    
    evtlistnew = []

    for i in evtlist:
        if len(data[i.begin:i.end]) > 6:
            evtlistnew.append(i)
            
    return evtlistnew

def get_truelabel(data,events):
    '''
    get the true label for each point in time
    '''
    
    x = pds.to_datetime(data.index)
    y = np.zeros(np.shape(data)[0])
    
    for e in events:
        n_true = np.where((x >= e.begin) & (x <= e.end))
        y[n_true] = 1
    
    label = pds.DataFrame(y, index = data.index, columns = ['label'])
    
    return label

def make_views(arr,win_size,step_size,writeable = False):
    """
    
    see https://krbnite.github.io/Memory-Efficient-Windowing-of-Time-Series-Data-in-Python-3-Memory-Strides-in-Pandas/
    
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