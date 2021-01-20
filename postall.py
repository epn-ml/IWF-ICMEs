import pandas as pds
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import confusion_matrix
import postprocess as pp
import event as evt
import performance as prf



def load(predfolder, data, begindate, windows):
    
    prediction = pds.DataFrame(index = data[data.index>begindate].index)
    
    for width in windows :
        coll = np.zeros(len(prediction))
        df = pds.read_csv(predfolder+str(width)+'.csv', header=None, index_col=None)
        df.drop([0], inplace = True)
        df.set_index([0], inplace = True)
        dfv = df.values.reshape([len(df.values)])
        coll[prediction.index.isin(df.index)] = dfv
        prediction[str(width)] = coll
    
    prediction = pds.DataFrame(index = prediction.index, data = median_filter(prediction.values, (1,5)))

    return prediction

def predictandplot(prediction, threshhold, evtlist, beginyear, data, spacecraft):
    
    integral = prediction.sum(axis=1)
    ICMEs, pred = pp.turn_peaks_to_clouds(integral,threshhold)
    test_clouds = [x for x in evtlist if x.begin.year>beginyear]
    
    #Score by event
    print('Threshhold is:', threshhold)
    TP, FN, FP, detected = prf.evaluate(ICMEs, test_clouds, thres=0.01)
    print('True positive Rate is:',len(TP)/len(ICMEs))
    print('Precision is:',len(TP)/(len(TP)+len(FP)))
    print('Recall is:',len(TP)/(len(TP)+len(FN)))
    print('True Positives', len(TP))
    print('False Negatives', len(FN))
    print('False Positives', len(FP))
    
    #create confusionmatrix
    
#    truelabel = pp.get_label1(pred, test_clouds)

#    cm = confusion_matrix(truelabel, pred)

#    pp.plot_confusion_matrix(cm           = cm, 
#                          normalize    = True,
#                          target_names = ['None', 'ICME'],
#                          title        = "Confusion Matrix")
    
    #plot detected ICMEs

    for i in range(0, len(detected)):
        predstart = TP[i].begin
        predend = TP[i].end
        detected[i].iwfplot(data, 20, i, spacecraft + '-Detected-', predstart, predend)
        
    for i in range(0, len(FP)):
        FP[i].iwfplotnopred(data, 20, i, spacecraft + '-False Positive-')
        
#    for i in range(0, len(FN)):
#        FN[i].iwfplotnopred(data, 20, i, spacecraft + '-False Negative-')
        
    
