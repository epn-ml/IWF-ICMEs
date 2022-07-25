import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np


def computeBeta(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    The function assume data already has ['Np','B','Vth'] features
    '''
    try:
        data['Beta'] = 1e6 * data['Vth']*data['Vth']*constants.m_p*data['Np']*1e6*constants.mu_0/(1e-18*data['B']*data['B'])
    except KeyError:
        print('Error computing Beta,B,Vth or Np'
              ' might not be loaded in dataframe')
    return data


def computeRmsBob(data):
    '''
    compute the evolution of the rmsbob instantaneous for data
    data is a Pandas dataframe
    The function assume data already has ['B_rms] features
    '''
    try:
        data['RmsBob'] =np.sqrt(data['Bx_rms']**2+data['By_rms']**2+data['Bz_rms']**2)/data['B']
    except KeyError:
        print('Error computing rmsbob,B or rms of components'
              ' might not be loaded in dataframe')
    return data


def computePdyn(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['Np']*data['V']**2
    except KeyError:
        print('Error computing Pdyn, V or Np might not be loaded '
              'in dataframe')