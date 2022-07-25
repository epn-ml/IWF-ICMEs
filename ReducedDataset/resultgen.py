import numpy as np
import pandas as pds
import pickle
import matplotlib.pyplot as plt

def get_lags(splitn, sc):
    starts = []
    ends = []
    for i in range(1,splitn+1):
        with open('startlag'+ sc +str(i)+'.p', 'rb') as fp:
            startlist = pickle.load(fp)
            starts = starts + startlist
    
        with open('endlag'+ sc +str(i)+'.p', 'rb') as fp:
            endlist = pickle.load(fp)
            ends = ends + endlist
    return starts,ends



def get_cm(splitn,sc):
    
    cm = np.loadtxt('cm'+ sc +'1.txt')
    
    for i in range(2,splitn+1):
        cm = cm + np.loadtxt('cm'+sc+str(i)+'.txt')
    
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
        
    return cm, TN, FP, FN, TP

def get_eventcm(splitn,sc):
    
    cmevent = np.loadtxt('cmevent'+ sc +'1.txt')
    
    for i in range(2,splitn+1):
        cmevent = cmevent + np.loadtxt('cmevent'+sc +str(i)+'.txt')
    
    FPevent = cmevent[0,1]
    FNevent = cmevent[1,0]
    TPevent = cmevent[1,1]
        
    return cmevent, FPevent, FNevent, TPevent

def metrics(TN, FP, FN, TP):
    
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    dice = 2*TP/(2*TP+FP+FN)
    tss = TP/(TP+FN) + TN/(FP+TN) - 1
    iou = TP/(TP+FN+FP)
    
    print('POINTWISE RESULTS')
    print('Recall: ',recall)
    print('Precision: ',precision)
    print('Dice Coefficient: ', dice)
    print('True Skill Statistics: ',tss)
    print('Intersection over Union: ',iou)
    return recall, precision, dice, tss, iou
    
def eventmetrics(FP, FN, TP):
    
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    dice = 2*TP/(2*TP+FP+FN)
    iou = TP/(TP+FN+FP)
    
    print('EVENTWISE RESULTS')
    print('Recall: ',recall)
    print('Precision: ',precision)
    print('Dice Coefficient: ', dice)
    print('Intersection over Union: ',iou)
    print('False Positives: ', FP)
    print('False Negatives: ', FN)
    print('True Positives: ', TP)
    return recall, precision, dice, iou

def lags(starts,ends):
    starts = np.asarray(starts)/120
    ends = np.asarray(ends)/120 

    # fixed bin size
    bins = np.arange(-10, 10, 0.5) # fixed bin size

    plt.xlim([-10, 10])

    plt.hist(starts, bins=bins, alpha=0.5)
    #plt.title('Deviation of the start time')
    plt.xlabel('time [h]')
    plt.ylabel('count')

    plt.show()
    
    
    # fixed bin size
    bins = np.arange(-10, 10, 0.5) # fixed bin size

    plt.xlim([0, 10])

    plt.hist(np.abs(starts), bins=bins, alpha=0.5)
    #plt.title('Deviation of the start time')
    plt.xlabel('time [h]')
    plt.ylabel('count')

    plt.show()
    
    # fixed bin size
    bins = np.arange(-10, 10, 0.5) # fixed bin size

    plt.xlim([-10, 10])

    plt.hist(ends, bins=bins, alpha=0.5)
    #plt.title('Deviation of the end time')
    plt.xlabel('time [h]')
    plt.ylabel('count')

    plt.show()
    
    # fixed bin size
    bins = np.arange(-10, 10, 0.5) # fixed bin size

    plt.xlim([0, 10])

    plt.hist(np.abs(ends), bins=bins, alpha=0.5)
    #plt.title('Deviation of the end time')
    plt.xlabel('time [h]')
    plt.ylabel('count')

    plt.show()