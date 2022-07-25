import pandas as pds
import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

'''
A lot of these functions borrow heavily from 
https://github.com/gautiernguyen/Automatic-detection-of-ICMEs-at-1-AU-a-deep-learning-approach
'''

class Event:

    def __init__(self, begin, end, param=None):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin

    def __eq__(self, other):
        '''
        return True if other overlaps self during 65/100 of the time
        '''
        return overlap(self, other) > 0.65*self.duration

    def iwfplot(self, data, delta, i, typ, predstart, predend):
        '''
        plot event in the style used by IWF including predicted event
        '''
        return plot_insitu_icmecat_mag_plasma(data, self.begin, self.end, delta, i, typ, predstart, predend)
    
    def iwfplotnopred(self, data, delta, typ):
        '''
        plot event in the style used by IWF
        '''        
        return plot_insitu_icmecat_mag_plasma_nopred(data, self.begin, self.end, delta, typ)
    
    def plot_similaritymap(self, data, delta, i, typ, prediction):
        '''
        plot event and show similarity vs predicted similarity
        '''
        return plot_similaritymap(data, self.begin, self.end, delta, i, typ, prediction)

    
def clearempties(evtlist, data):
    '''
    delete all events from eventlist that do not have sufficient data
    '''
    
    evtlistnew = []

    for i in evtlist:
        if len(data[i.begin:i.end]) > 6:
            evtlistnew.append(i)
            
    return evtlistnew
        
        
def overlap(event1, event2):
    '''return the time overlap between two events as a timedelta'''
    delta1 = min(event1.end, event2.end)
    delta2 = max(event1.begin, event2.begin)
    return max(delta1-delta2,
               datetime.timedelta(0))

def isInList(ref_event, event_list, thres):
    '''
    returns True if ref_event is overlapped thres percent of its duration by
    at least one in event_list
    '''
    return max(overlapWithList(ref_event,event_list)) > thres*ref_event.duration


def find(ref_event, event_list, thres, choice='first'):
    '''
    Return the event in event_list that overlap ref_event for a given threshold
    if it exists
    Choice give the preference of returned :
    first return the first of the lists
    Best return the one with max overlap
    merge return the combination of all of them
    '''
    if isInList(ref_event, event_list, thres):
        return(choseEventFromList(ref_event, event_list, choice))
    else:
        return None
    
def similarity(event1, event2):
    '''
    returns similarity of two events
    '''
    if event1 is None:
        return 0
    inter = overlap(event1, event2)
    return inter/(event1.duration+event2.duration-inter)

    
def read_cat(begin, end, iwinind, dateFormat="%Y/%m/%d %H:%M",
             sep=',', get_proba=False):
    '''
    read catalog and return eventlist
    '''
    evtList = []
    begin = pds.to_datetime(begin, format=dateFormat)
    end = pds.to_datetime(end, format=dateFormat)
    for i in iwinind:
        if (begin[i] < datetime.datetime(2021,2,3)):
            evtList.append(Event(begin[i], end[i]))
    if get_proba is True:
        for i, elt in enumerate(evtList):
            elt.proba = df['proba'][i]
    return evtList


def get_similarity(index, width, evtList):
    '''
    For a given list of event and a given window size (in hours) and
    a datetime index, return the associated serie of similarities
    '''
    y = np.zeros(len(index))
    for i, date in enumerate(index):
        window = Event(date-datetime.timedelta(hours=int(width)/2),
                       date+datetime.timedelta(hours=int(width)/2))
        seum = [similarity(x, window)for x in evtList if (window.begin < x.end) and (window.end > x.begin)]
        if len(seum) > 0:
            y[i] = max(seum)
    return pds.Series(index=index, data=y)

def overlapWithList(ref_event, event_list, percent=False):
    '''
    return the list of the overlaps between an event and the elements of
    an event list
    Have the possibility to have it as the percentage of fthe considered event
    in the list
    '''
    if percent:
        return [overlap(ref_event, elt)/elt.duration for elt in event_list]
    else:
        return [overlap(ref_event, elt) for elt in event_list]


def choseEventFromList(ref_event, event_list, choice='first'):
    '''
    return an event from even_list according to the choice adopted
    first return the first of the lists
    last return the last of the lists
    best return the one with max overlap
    merge return the combination of all of them
    '''
    if choice == 'first':
        return event_list[0]
    if choice == 'last':
        return event_list[-1]
    if choice == 'best':
        return event_list[np.argmax(overlapWithList(ref_event, event_list))]
    if choice == 'merge':
        return evt.merge(event_list[0], event_list[-1])
        
def plot_similaritymap(data, start, end, delta, i, typ, prediction):

    sns.set_style('darkgrid')
    sns.set_context('paper')
    
        
    fig=plt.figure(figsize=(12,6), dpi=150) 
    
    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))    

    data = data[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]
    
    prediction = prediction[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]

    ax1 = plt.subplot(311) 

    ax1.plot_date(data.index, data['bx'],'-r',label='Bx',linewidth=0.5)
    ax1.plot_date(data.index, data['by'],'-g',label='By',linewidth=0.5)
    ax1.plot_date(data.index, data['bz'],'-b',label='Bz',linewidth=0.5)
    ax1.plot_date(data.index, data['bt'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
    ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
    ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
    plt.ylabel('B [nT]')
    plt.legend(loc=3,ncol=4,fontsize=8)
     

    ax1.set_ylim(-np.nanmax(data['bt'])-5,np.nanmax(data['bt'])+5)  
    ax1.set_xlim(data.index[0],data.index[-1])  

    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
    ax2 = plt.subplot(312,sharex = ax1)
    im = np.tile(prediction['pred'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax2.pcolormesh(X,Y,im,cmap='binary')

    plt.setp(ax2.get_yticklabels(), visible=False)
         #plot vertical lines
    
    plt.ylabel('Predicted Label')
    
    ax3 = plt.subplot(313)
    im = np.tile(prediction['true'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax3.pcolormesh(X,Y,im,cmap='binary')
    plt.setp(ax3.get_yticklabels(), visible=False)
         #plot vertical lines
    
    plt.ylabel('True Label')
    
    ax1.get_shared_y_axes().join(ax1,ax2)
    ax1.get_shared_y_axes().join(ax1,ax3)

    plt.tight_layout()
    plt.show()
        
def plot_insitu_icmecat_mag_plasma_nopred(data, start, end, delta, typ):
    
         
     sns.set_style('darkgrid')
     sns.set_context('paper')
        
     fig=plt.figure(figsize=(9,6), dpi=150)
    
     data = data[start-datetime.timedelta(hours=delta):
                 end+datetime.timedelta(hours=delta)]
     
     ax1 = plt.subplot(411) 

     ax1.plot_date(data.index, data['bx'],'-r',label='Bx',linewidth=0.5)
     ax1.plot_date(data.index, data['by'],'-g',label='By',linewidth=0.5)
     ax1.plot_date(data.index, data['bz'],'-b',label='Bz',linewidth=0.5)
     ax1.plot_date(data.index, data['bt'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
     ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
     ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
     plt.ylabel('B [nT]')
     plt.legend(loc=3,ncol=4,fontsize=8)
     
     ax1.set_ylim(-np.nanmax(data['bt'])-5,np.nanmax(data['bt'])+5)   
     ax1.set_xlim(data.index[0],data.index[-1])
     
     plt.setp(ax1.get_xticklabels(), visible=False)

     plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
     ax2 = plt.subplot(412,sharex=ax1) 
     ax2.plot_date(data.index,data['vt'],'-k',label='V',linewidth=0.7)
    

     #plot vertical lines
     ax2.plot_date([start,start],[0,3000],'-k',linewidth=1)                    
     ax2.plot_date([end,end],[0,3000],'-k',linewidth=1)


     plt.ylabel('V [km/s]')
     
     #check plasma data exists
     if np.isnan(np.nanmin(data['vt']))==False:
         ax2.set_ylim(np.nanmin(data['vt'])-20,np.nanmax(data['vt'])+100)   
     
     
     plt.setp(ax2.get_xticklabels(), visible=False)


     ax3 = plt.subplot(413,sharex=ax1) 
     ax3.plot_date(data.index,data['np'],'-k',label='Np',linewidth=0.7)
     
     #plot vertical lines
     ax3.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax3.plot_date([end,end],[0,1000],'-k',linewidth=1)

     plt.ylabel('N [ccm-3]')
     
     if np.isnan(np.nanmin(data['np']))==False:
         ax3.set_ylim(0,np.nanmax(data['np'])+10)   
    
     
     plt.setp(ax3.get_xticklabels(), visible=False)


     ax4 = plt.subplot(414,sharex=ax1) 
     ax4.plot_date(data.index,data['tp'],'-k',label='Tp',linewidth=0.7)
    
     #plot vertical lines
     ax4.plot_date([start,start],[0,10*1e6],'-k',linewidth=1)                        
     ax4.plot_date([end,end],[0,10*1e6],'-k',linewidth=1)
     ax4.set_yscale('log')


     plt.ylabel('T [K]')
     
     #if np.isnan(np.nanmin(data['tp']))==False:
     #    ax4.set_ylim(0,1e2)   

     
     
     plt.tight_layout()
     plt.show()
        
        
def plot_insitu_icmecat_mag_plasma(data, start, end, delta, i, typ, predstart, predend):
    
         
     sns.set_style('darkgrid')
     sns.set_context('paper')
        
     fig=plt.figure(figsize=(9,6), dpi=150)
    
     data = data[start-datetime.timedelta(hours=delta):
                 end+datetime.timedelta(hours=delta)]
     
     ax1 = plt.subplot(411) 

     ax1.plot_date(data.index, data['bx'],'-r',label='Bx',linewidth=0.5)
     ax1.plot_date(data.index, data['by'],'-g',label='By',linewidth=0.5)
     ax1.plot_date(data.index, data['bz'],'-b',label='Bz',linewidth=0.5)
     ax1.plot_date(data.index, data['bt'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
     ax1.plot_date([start,start],[-500,500],'-k',label = 'true event',linewidth=1)                      
     ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
     ax1.plot_date([predstart,predstart],[-500,500],'-r',label = 'predicted event',linewidth=1)                      
     ax1.plot_date([predend,predend],[-500,500],'-r',linewidth=1)  
    
     plt.ylabel('B [nT]')
     plt.legend(loc=3,ncol=4,fontsize=8)
     
     ax1.set_ylim(-np.nanmax(data['bt'])-5,np.nanmax(data['bt'])+5)
     ax1.set_xlim(data.index[0],data.index[-1])
     

    
     
     plt.setp(ax1.get_xticklabels(), visible=False)

     plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
     ax2 = plt.subplot(412,sharex=ax1) 
     ax2.plot_date(data.index,data['vt'],'-k',label='V',linewidth=0.7)
    

     #plot vertical lines
     ax2.plot_date([start,start],[0,3000],'-k',label = 'true event',linewidth=1)                    
     ax2.plot_date([end,end],[0,3000],'-k',linewidth=1)            
     ax2.plot_date([predstart,predstart],[0,3000],'-r',label = 'predicted event',linewidth=1)                      
     ax2.plot_date([predend,predend],[0,3000],'-r',linewidth=1)  


     plt.ylabel('V [km/s]')
     
     if np.isnan(np.nanmin(data['vt']))==False:
         ax2.set_ylim(np.nanmin(data['vt'])-20,np.nanmax(data['vt'])+100)   
     
     
     plt.setp(ax2.get_xticklabels(), visible=False)


     ax3 = plt.subplot(413,sharex=ax1) 
     ax3.plot_date(data.index,data['np'],'-k',label='Np',linewidth=0.7)
     
     #plot vertical lines
     ax3.plot_date([start,start],[0,1000],'-k',label = 'true event',linewidth=1)                       
     ax3.plot_date([end,end],[0,1000],'-k',linewidth=1)            
     ax3.plot_date([predstart,predstart],[0,1000],'-r',label = 'predicted event',linewidth=1)                      
     ax3.plot_date([predend,predend],[0,1000],'-r',linewidth=1)  

     plt.ylabel('N [ccm-3]')
     
     if np.isnan(np.nanmin(data['np']))==False:
         ax3.set_ylim(0,np.nanmax(data['np'])+10)   
    
     
     plt.setp(ax3.get_xticklabels(), visible=False)


     ax4 = plt.subplot(414,sharex=ax1) 
     ax4.plot_date(data.index,data['tp'],'-k',label='Tp',linewidth=0.7)
    
     #plot vertical lines
     ax4.plot_date([start,start],[0,10],'-k',label = 'true event',linewidth=1)                        
     ax4.plot_date([end,end],[0,10],'-k',linewidth=1)            
     ax4.plot_date([predstart,predstart],[0,10*1e6],'-r',label = 'predicted event',linewidth=1)                      
     ax4.plot_date([predend,predend],[0,10*1e6],'-r',linewidth=1)  
     ax4.set_yscale('log')


     plt.ylabel('T [K]')
     
     if np.isnan(np.nanmin(data['tp']))==False:
         ax4.set_ylim(0,1e2)   

     
     
     plt.tight_layout()
     plt.show()