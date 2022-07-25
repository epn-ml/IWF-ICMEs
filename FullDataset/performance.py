import pandas as pds
import datetime
import numpy as np
import event as evt
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(predicted_list, test_list, thres=0.51, durationCreepies=2.5):
    '''
    for each cloud of validation_list, gives the list of clouds in the
    predicted_list that overlap the cloud among the threshold
    '''
    TP = []
    FN = []
    FP = []
    detected = []
    for event in test_list:
        corresponding = evt.find(event, predicted_list, thres, 'best')
        if corresponding is None:
            FN.append(event)
        else:
            TP.append(corresponding)
            detected.append(event)
    FP = [x for x in predicted_list if max(evt.overlapWithList(x, test_list, percent=True)) == 0]
    seum = [x for x in FP if x.duration < datetime.timedelta(hours=durationCreepies)]
    for event in seum:
        FP.remove(event)
        predicted_list.remove(event)

    return TP, FN, FP, detected

def plot_all(data, start, end,typ, prediction):

    sns.set_style('darkgrid')
    sns.set_context('paper')
    
        
    fig=plt.figure(figsize=(12,6), dpi=150) 
    
    #plt.title(typ+' data: '+start.strftime("%Y-%b")+'  end: '+end.strftime("%Y-%b"))
    

    data = data[start:end]
    datan = pds.read_csv('data_nguyen.csv')
    
    prediction = prediction[start:end]
    predictionn= pds.read_csv('prediction_nguyen.csv')

     #sharex means that zooming in works with all subplots
    ax1 = plt.subplot(411) 

    ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
    ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
    ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
    ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
         
    plt.ylabel('B [nT]')
    plt.legend(loc=3,ncol=4,fontsize=8)
     
    ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5) 

    plt.title(typ+' - data: '+start.strftime("%Y-%b")+'-'+end.strftime("%Y-%b"))
    
    ax2 = plt.subplot(412)
    im = np.tile(prediction['true'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax2.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    plt.ylabel('True Label')
    
    ax3 = plt.subplot(413,sharex = ax1)
    im = np.tile(prediction['pred'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax3.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax3.get_yticklabels(), visible=False)
    
    plt.ylabel('Our Pipeline')
    
    ax4 = plt.subplot(414,sharex = ax1)
    im = np.tile(predictionn['pred'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax4.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax4.get_yticklabels(), visible=False)
    
    plt.ylabel('Nguyen')
    

    
    ax1.get_shared_y_axes().join(ax1,ax2)
    ax1.get_shared_y_axes().join(ax1,ax3)
    ax1.get_shared_y_axes().join(ax1,ax4)

    plt.tight_layout()
    plt.show()

def plot_similaritymap(data, start, end,typ, prediction):

    sns.set_style('darkgrid')
    sns.set_context('paper')
    
        
    fig=plt.figure(figsize=(12,6), dpi=150) 
    
    #plt.title(typ+' data: '+start.strftime("%Y-%b")+'  end: '+end.strftime("%Y-%b"))
    

    data = data[start:end]
    
    prediction = prediction[start:end]

     #sharex means that zooming in works with all subplots
    ax1 = plt.subplot(311) 

    ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
    ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
    ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
    ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
         
    plt.ylabel('B [nT]')
    plt.legend(loc=3,ncol=4,fontsize=8)
     
    ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5) 

    plt.title(typ+' data: '+start.strftime("%Y-%b")+' - '+end.strftime("%Y-%b"))
    
    ax2 = plt.subplot(312,sharex = ax1)
    im = np.tile(prediction['pred'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax2.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    plt.ylabel('Predicted Label')
    
    ax3 = plt.subplot(313)
    im = np.tile(prediction['true'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax3.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax3.get_yticklabels(), visible=False)
    
    plt.ylabel('True Label')
    
    ax1.get_shared_y_axes().join(ax1,ax2)
    ax1.get_shared_y_axes().join(ax1,ax3)

    plt.tight_layout()
    plt.show()

