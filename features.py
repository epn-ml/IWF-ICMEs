import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np
import time


'''
These functions compute extrafeatures from data loaded from space missions
Current computed features are :
Beta
Pdyn
Ith
Texrat
Rollingstd
'''

def computeBetawiki(data):
    '''
    compute the Beta according to wikipedia
    data is a Pandas dataframe
    the function assumes data already hat ['np', 'tp', 'bt'] features
    '''
    try:
        data['beta'] = 1e6*data['np']*constants.Boltzmann*data['tp']/(np.square(1e-9*data['bt'])/(2*constants.mu_0))
    except KeyError:
        print('Error computing Beta, np, tp or bt might not be loaded '
              'in dataframe')
    return data


def computePdyn(data):
    '''
    compute the dynamic pressure for data
    data is a Pandas dataframe
    the function assume data already has ['np','vt'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['np']*data['vt']**2
    except KeyError:
        print('Error computing Pdyn, vt or np might not be loaded '
              'in dataframe')

def computeith(data):
    '''
    compute the thermal index
    the function assumes data already hat ['vt', 'tp']
    '''
    try:
        data['ith'] = (500*data['vt'] + 1.75*1e5)/data['tp']
    except KeyError:
        print('Error computing Ith, vt or tp might not be loaded in dataframe')
        
        
def computeTexrat(data):
    '''
    compute the ratio of Tp/Tex
    the function assumes data already hat ['vt', 'tp']
    '''
    try:
        data['texrat'] = data['tp']*1e-3/(np.square(0.031*data['vt']-5.1))
    except KeyError:
        print( 'Error computing Texrat, tp or vt might not be loaded in dataframe')


def computeRollingStd(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the standard dev over
    a defined period of time (timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+'std'
    data[name] = data[feature].rolling(timeWindow, center=center,
                                       min_periods=1).std()
    return data
