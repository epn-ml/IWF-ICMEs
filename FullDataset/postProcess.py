import pandas as pds
import numpy as np
import datetime
import performance as prf
import event as evt
from tqdm import tqdm
from data_generator import *
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, plot_precision_recall_curve
from unetgen import UnetGen
import preProcess


def generate_result_old(test_image_paths, test_mask_paths, model,C,t):
    ## Generating the result
    image_size = (t,1,C)
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
    
        df_mask = pds.read_csv(test_mask_paths[i],header = None,index_col=[0])
        image = parse_image(test_image_paths[i], image_size)
        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        if i == 0:
            result = df_mask
        else:
            result = pds.concat([result, df_mask], sort=True)

    result = result.sort_index()
    result.index = pds.to_datetime(result.index)
    
    return result

def generate_result(X_test, Y_test, model,C,t):
    ## Generating the result
    image_size = (t,1,C)
    X_test_windowed = preProcess.make_views(X_test, win_size = t, step_size = t ,writeable = False)
    Y_test_windowed = preProcess.make_views(Y_test, win_size = t, step_size = t ,writeable = False)

    for i, test in tqdm(enumerate(X_test_windowed), total=len(X_test_windowed)):
    
        df_mask = pds.DataFrame(Y_test_windowed[i])
        df_mask = df_mask.set_index(df_mask[1])
        df_mask = df_mask.iloc[: , :-1]
        image = pds.DataFrame(test)
        image = image.set_index(image[C])
        image = image.iloc[: , :-1]
        predict_mask = model.predict(np.expand_dims(np.asarray(np.expand_dims(image, axis=0)).astype('float64'),2))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        if i == 0:
            result = df_mask
        else:
            result = pds.concat([result, df_mask], sort=True)

    result = result.sort_index()
    result['true'] = np.asarray(result['true']).astype('float64')
    result.index = pds.to_datetime(result.index)
    
    return result


def removeCreepy(eventList, thres=3.5):
    '''
    For a given list, remove the element whose duration is under the threshold
    '''
    return [x for x in eventList if x.duration > datetime.timedelta(hours=thres)]

def make_binary(serie, thresh):
    
    serie = (serie > thresh)*1
    serie = serie.interpolate()
    
    return serie

def makeEventList(y, label, delta=2):
    '''
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    '''
    listOfPosLabel = y[y == label]
    if len(listOfPosLabel) == 0:
        return []
    deltaBetweenPosLabel = listOfPosLabel.index[1:] - listOfPosLabel.index[:-1]
    deltaBetweenPosLabel.insert(0, datetime.timedelta(0))
    endOfEvents = np.where(deltaBetweenPosLabel > datetime.timedelta(minutes=delta))[0]
    indexBegin = 0
    eventList = []
    for i in endOfEvents:
        end = i
        eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[end]))
        indexBegin = i+1
    eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[-1]))
    return eventList

def get_roc(result):

    # calculate roc curve

    fpr, tpr, thresholds = roc_curve(result['true'], result['pred'])

    auc_roc = roc_auc_score(result['true'], result['pred'])


    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'r', label = 'WIND - AUC = %0.2f' % auc_roc)
    plt.plot([0, 1], [0, 1],'g--',label = 'No skill')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def get_pr(result):

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(result['true'], result['pred'])
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.title('Precision vs Recall')
    plt.plot(recall, precision, 'r', label = 'WIND - AUC = %0.2f' % pr_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()   
    
def vary_thresh(result,test_cloud, begin = 0.0001, end = 0.9999, n = 50):
    # vary the threshold to figure out the best value

    varythresh = np.linspace(begin,end,n)

    prec = np.zeros(len(varythresh))
    rec = np.zeros(len(varythresh))
    tp = np.zeros(len(varythresh))
    fn = np.zeros(len(varythresh))
    fp = np.zeros(len(varythresh))

    for i in range(len(varythresh)):
        resultbin = make_binary(result['pred'], varythresh[i])
        events = makeEventList(resultbin, 1, 10)
        ICMEs = removeCreepy(events, 2)
        TP, FN, FP, detected = prf.evaluate(ICMEs, test_cloud, thres=0.1)
        prec[i] = len(TP)/(len(TP)+len(FP))
        rec[i] = len(TP)/(len(TP)+len(FN))
        tp[i] = len(TP)
        fn[i] = len(FN)
        fp[i] = len(FP)
        
    # plot number of false negatives and false positives for different thresholds

    plt.title('False Negatives and False Positives')

    plt.plot(varythresh, fp,'-', color='k', label = 'FP - WIND')
    plt.plot(varythresh, fn,'--', color='k', label = 'FN - WIND')

    plt.legend(loc = 'upper right')
  
    # giving labels to the axises
    plt.xlabel('Thresholds')
    plt.ylabel('Number')
  
    # defining display layout 
    plt.tight_layout()
  
    # show plot
    plt.show()
    
    pr_auc = auc(rec, prec)
    pr_auc = pr_auc/(np.max(rec)-np.min(rec))
    
    plt.figure()
    plt.title('Precision vs Recall')
    plt.plot(rec, prec, 'r', label = 'WIND - AUC = %0.2f' % pr_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show() 
        
    return prec, rec, tp, fn, fp

def plot_durations(TP,FN,FP):
       
    tp = np.zeros(len(TP))
    fn = np.zeros(len(FN))
    fp = np.zeros(len(FP))
    
    for i in range(len(tp)):
        tp[i] = TP[i].duration.total_seconds()/60/60
        
    for i in range(len(fp)):
        fp[i] = FP[i].duration.total_seconds()/60/60
        
    for i in range(len(fn)):
        fn[i] = FN[i].duration.total_seconds()/60/60
    
    dic = {'TP':tp, 'FP':fp, 'FN':fn}
    df = pds.DataFrame.from_dict(dic, orient='index').transpose()
    
    ax = df.plot.hist(bins = 25, alpha =0.4)
    ax.set_xlabel('Duration')
    ax.set_title('Duration of True Positives, False Positives and False Negatives')
    
def compare_lists(dicoflists):
    plt.figure()
    years = ['1998','1999','2000','2001','2002','2003', '2004', '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
    df = pds.DataFrame(columns = dicoflists, index = years)
    
    for lists in dicoflists:
        for year in years:
            if (len([x for x in dicoflists[lists] if (str(x.begin.year)==year)]) > 0):
                df[lists][year] = len([x for x in dicoflists[lists] if (str(x.begin.year)==year)])
    print(df.sum())            
    ax = df.plot(figsize=(10,7))
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('Year')
    ax.legend(prop=dict(size=10))
    
    print(df)
    #ax.set_title('Number of Events in Eventlists')