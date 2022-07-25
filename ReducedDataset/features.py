import scipy.constants as constants
import pandas as pds
import numpy as np

def computeBetawiki(data):
    '''
    compute Beta according to wiki
    '''
    try:
        data['beta'] = 1e6*data['np']*constants.Boltzmann*data['tp']/(np.square(1e-9*data['bt'])/(2*constants.mu_0))
    except KeyError:
        print('KeyError')
        
    return data
                                                               
def computePdyn(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['np']*data['vt']**2
    except KeyError:
        print('Error computing Pdyn, V or Np might not be loaded '
              'in dataframe')
    return data
        
def computeTexrat(data):
    '''
    compute the ratio of Tp/Tex
    '''
    try:
        data['texrat'] = data['tp']*1e-3/(np.square(0.031*data['vt']-5.1))
    except KeyError:
        print( 'Error computing Texrat')
    
    return data
       