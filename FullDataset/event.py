import pandas as pds
import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pickle


class Event:

    def __init__(self, begin, end, param=None, origin = None):
        self.begin = begin
        self.end = end
        self.proba = None
        self.duration = self.end-self.begin
        self.origin = origin

    def __eq__(self, other):
        '''
        return True if other overlaps self during 65/100 of the time
        '''
        return overlap(self, other) > 0.65*self.duration

    def __str__(self):
        return "{} ---> {}".format(self.begin, self.end)

    def get_Proba(self, y):
        '''
        Give the mean probability of the event following the list
        of event predicted probability y
        '''
        self.proba = y[self.begin:self.end].mean()

    def get_data(self, df):
        self.param = df


    def iwfplot(self, data, delta, i, typ, predstart, predend):
        return plot_insitu_icmecat_mag_plasma(data, self.begin, self.end, delta, i, typ, predstart, predend)
    
    def iwfplotnopred(self, data, delta, typ):
        return plot_insitu_icmecat_mag_plasma_nopred(data, self.begin, self.end, delta, typ)#
    
    def plotall(self, data, delta, typ):
        return plotall(data, self.begin, self.end, delta, typ)#
    
    def plot_similarity(self, data, delta, i, typ, prediction):
        return plot_similarity(data, self.begin, self.end, delta, i, typ, prediction)
    
    def plot_similaritymap(self, data, delta, i, typ, prediction):
        return plot_similaritymap(data, self.begin, self.end, delta, i, typ, prediction)

    def getValue(self, df, feature):
        '''
        for a given df, return the mean of a given feature during the events
        '''
        return df[feature][self.begin:self.end].mean()
     
def clearempties(evtlist, data):
    
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
    at least one elt in event_list
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
    if event1 is None:
        return 0
    inter = overlap(event1, event2)
    return inter/(event1.duration+event2.duration-inter)

    
def read_cat(begin, end, iwinind, dateFormat="%Y/%m/%d %H:%M",
             sep=',', get_proba=False):
    
    '''
    get indices of events by different spacecraft
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


def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

        
def plot_similarity(data, start, end, delta, i, typ, prediction):

    sns.set_style('darkgrid')
    sns.set_context('paper')
    
        
    fig=plt.figure(figsize=(12,6), dpi=150) 
    
    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))

    
    #prediction.index = pds.to_datetime(prediction.index)
    

    data = data[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]
    
    prediction = prediction[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]

     #sharex means that zooming in works with all subplots
    ax1 = plt.subplot(211) 

    ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
    ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
    ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
    ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
    ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
    ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
    plt.ylabel('B [nT]')
    plt.legend(loc=3,ncol=4,fontsize=8)
     
#    try:
    ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5)
    ax1.set_xlim(data.index[0],data.index[-1])
#    except ValueError:  #raised if `y` is empty.
#        pass
     
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
    ax2 = plt.subplot(212)
    ax2.plot_date(prediction.index, prediction['pred'],'-r',label = 'Predicted label',linewidth=0.5)
    ax2.plot_date(prediction.index, prediction['true'],'-b',label = 'True label',linewidth=0.5)
    ax2.set_ylim(-0.5,1.5)
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
         #plot vertical lines
    ax2.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
    ax2.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
    plt.ylabel('Label')
    plt.legend(loc=3,ncol=4,fontsize=8)

    #forceAspect(ax2,aspect=9)
    
    ax1.get_shared_y_axes().join(ax1,ax2)

    plt.tight_layout()
    plt.show()
    
def plot_similaritymap(data, start, end, delta, i, typ, prediction):

    sns.set_style('darkgrid')
    sns.set_context('paper')
    
        
    fig=plt.figure(figsize=(12,6), dpi=150) 
    
    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))

    
    #prediction.index = pds.to_datetime(prediction.index)
    

    data = data[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]
    
    prediction = prediction[start-datetime.timedelta(hours=delta):
                end+datetime.timedelta(hours=delta)]

     #sharex means that zooming in works with all subplots
    ax1 = plt.subplot(311) 

    ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
    ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
    ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
    ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
    ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
    ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
    plt.ylabel('B [nT]')
    plt.legend(loc=3,ncol=4,fontsize=8)
     
#    try:
    ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5)
    ax1.set_xlim(data.index[0],data.index[-1])
#    except ValueError:  #raised if `y` is empty.
#        pass
     
#    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
    ax2 = plt.subplot(312,sharex = ax1)
    im = np.tile(prediction['pred'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax2.pcolormesh(X,Y,im,cmap='binary')
#    ax2.imshow(im,cmap='cividis')
    
#    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
         #plot vertical lines
    
    plt.ylabel('Predicted Label')
    
    ax3 = plt.subplot(313)
    im = np.tile(prediction['true'],(15,1))
    x = prediction.index
    y = np.arange(15)
    X,Y = np.meshgrid(x,y)
    ax3.pcolormesh(X,Y,im,cmap='binary')
    #ax3.imshow(im,cmap='cividis')
    
#    plt.setp(ax3.get_xticklabels(), visible=False)
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
        
     fig=plt.figure(figsize=(9,4), dpi=150)
    
     data = data[start-datetime.timedelta(hours=delta):
                 end+datetime.timedelta(hours=delta)]
     
     #sharex means that zooming in works with all subplots
     ax1 = plt.subplot(411) 

     ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
     ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
     ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
     ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
     ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
     ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
     plt.ylabel('B [nT]')
     plt.legend(loc=3,ncol=4,fontsize=8)
     
     ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5)   
     ax1.set_xlim(data.index[0],data.index[-1])
     
     plt.setp(ax1.get_xticklabels(), visible=False)

     plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
     ax2 = plt.subplot(412,sharex=ax1) 
     ax2.plot_date(data.index,data['V'],'-k',label='V',linewidth=0.7)
    

     #plot vertical lines
     ax2.plot_date([start,start],[0,3000],'-k',linewidth=1)                    
     ax2.plot_date([end,end],[0,3000],'-k',linewidth=1)


     plt.ylabel('V [km/s]')
     
     #check plasma data exists
     if np.isnan(np.nanmin(data['V']))==False:
         ax2.set_ylim(np.nanmin(data['V'])-20,np.nanmax(data['V'])+100)   
     
     
     plt.setp(ax2.get_xticklabels(), visible=False)


     ax3 = plt.subplot(413,sharex=ax1) 
     ax3.plot_date(data.index,data['Np'],'-k',label='Np',linewidth=0.7)
     
     #plot vertical lines
     ax3.plot_date([start,start],[-10,1000],'-k',linewidth=1)                       
     ax3.plot_date([end,end],[-10,1000],'-k',linewidth=1)

     plt.ylabel('N [ccm-3]')
     
     if np.isnan(np.nanmin(data['Np']))==False:
         ax3.set_ylim(-10,np.nanmax(data['Np'])+10)   
    
     
     plt.setp(ax3.get_xticklabels(), visible=False)  

     ax4 = plt.subplot(414,sharex=ax1) 
     ax4.plot_date(data.index,data['Beta'],'-k',label='Beta',linewidth=0.7)
     
     #plot vertical lines
     ax4.plot_date([start,start],[-10,1000],'-k',linewidth=1)                       
     ax4.plot_date([end,end],[-10,1000],'-k',linewidth=1)
#     ax4.set_yscale('log')
#     ax4.set_yticks([1e0, 0, 1e1])

     if np.isnan(np.nanmin(data['Beta']))==False:
         ax4.set_ylim(-0.5,np.nanmax(data['Beta'])+1) 


     plt.ylabel('Beta')
     
#     ax4.set_ylim(0,20)   
    
    
     plt.setp(ax4.get_xticklabels(), visible=False)  

     
     plt.tight_layout()
     plt.show()


     #plotfile=typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M")+'.png'
  
     #plt.savefig(plotfile)
     #print('saved as ',plotfile)
        
        
def plot_insitu_icmecat_mag_plasma(data, start, end, delta, i, typ, predstart, predend):
    
         
     sns.set_style('darkgrid')
     sns.set_context('paper')
        
     fig=plt.figure(figsize=(9,4), dpi=150)
    
     data = data[start-datetime.timedelta(hours=delta):
                 end+datetime.timedelta(hours=delta)]
     
     #sharex means that zooming in works with all subplots
     ax1 = plt.subplot(411) 

     ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
     ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
     ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
     ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
     ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
     ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
     ax1.plot_date([predstart,predstart],[-500,500],'-r',linewidth=1)                      
     ax1.plot_date([predend,predend],[-500,500],'-r',linewidth=1)  
    
     plt.ylabel('B [nT]')
     plt.legend(loc=3,ncol=4,fontsize=8)
     
     ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5)   
     ax1.set_xlim(data.index[0],data.index[-1])
     
     plt.setp(ax1.get_xticklabels(), visible=False)

     plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
     ax2 = plt.subplot(412,sharex=ax1) 
     ax2.plot_date(data.index,data['V'],'-k',label='V',linewidth=0.7)
    

     #plot vertical lines
     ax2.plot_date([start,start],[0,3000],'-k',linewidth=1)                    
     ax2.plot_date([end,end],[0,3000],'-k',linewidth=1)            
     ax2.plot_date([predstart,predstart],[0,3000],'-r',linewidth=1)                      
     ax2.plot_date([predend,predend],[0,3000],'-r',linewidth=1)  


     plt.ylabel('V [km/s]')
     
     #check plasma data exists
     if np.isnan(np.nanmin(data['V']))==False:
         ax2.set_ylim(np.nanmin(data['V'])-20,np.nanmax(data['V'])+100)   
     
     
     plt.setp(ax2.get_xticklabels(), visible=False)


     ax3 = plt.subplot(413,sharex=ax1) 
     ax3.plot_date(data.index,data['Np'],'-k',label='Np',linewidth=0.7)
     
     #plot vertical lines
     ax3.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax3.plot_date([end,end],[0,1000],'-k',linewidth=1)            
     ax3.plot_date([predstart,predstart],[0,1000],'-r',linewidth=1)                      
     ax3.plot_date([predend,predend],[0,1000],'-r',linewidth=1)  

     plt.ylabel('N [ccm-3]')
     
     if np.isnan(np.nanmin(data['Np']))==False:
         ax3.set_ylim(0,np.nanmax(data['Np'])+10)   
    
     
     plt.setp(ax3.get_xticklabels(), visible=False)
     
     ax4 = plt.subplot(414,sharex=ax1) 
     ax4.plot_date(data.index,data['Beta'],'-k',label='Beta',linewidth=0.7)
     
     #plot vertical lines
     ax4.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax4.plot_date([end,end],[0,1000],'-k',linewidth=1)
                    
     ax4.plot_date([predstart,predstart],[0,1000],'-r',linewidth=1)                      
     ax4.plot_date([predend,predend],[0,1000],'-r',linewidth=1)  
     ax4.set_yscale('log')
    
     ax4.set_yticks([1e0, 0, 1e1])
        
     plt.ylabel('Beta')
     
     ax4.set_ylim(0,20)     
    
    
     plt.setp(ax4.get_xticklabels(), visible=False)  

     
     plt.tight_layout()
     plt.show()
        
def read_csv(filename, get_origin=True,
             index_col=0, header=0, dateFormat="%Y/%m/%d %H:%M",
             sep=','):
    '''
    Consider a  list of events as csv file ( with at least begin and end)
    and return a list of events
    index_col and header allow the correct reading of the current fp lists
    '''
    df = pds.read_csv(filename, index_col=index_col, header=header, sep=sep)
    df['begin'] = pds.to_datetime(df['begin'], format=dateFormat)
    df['end'] = pds.to_datetime(df['end'], format=dateFormat)
    evtList = [Event(df['begin'][i], df['end'][i])
               for i in range(0, len(df))]
    if get_origin is True:
        for i, elt in enumerate(evtList):
            elt.origin = df['origin'][i]
    return evtList

def get_catevents():
    
    # load ICME catalog data
    [ic,header,parameters] = pickle.load(open('HELCATS_ICMECAT_v20_pandas.p', "rb" ))
    # extract important values
    isc = ic.loc[:,'sc_insitu'] 
    starttime = ic.loc[:,'icme_start_time']
    endtime = ic.loc[:,'mo_end_time']
    # Event indices
    iwinind = np.where(isc == 'Wind')[0]

    winbegin = starttime[iwinind]
    winend = endtime[iwinind]

    # get list of events

    evtListw = read_cat(winbegin, winend, iwinind)
    
    return evtListw   

def read_chinchilla():
    dateFormat="%Y %m/%d %H:%M"
    df = pds.read_csv('Nieves_Chinchilla.csv')
    df['MO2\nstart time'] = pds.to_datetime(df['MO2\nstart time'], format=dateFormat)
    df['MO/ICME\nend time'] = pds.to_datetime(df['MO/ICME\nend time'], format=dateFormat)
    evtList = [Event(df['MO2\nstart time'][i], df['MO/ICME\nend time'][i])
               for i in range(0, len(df))]
    
    return evtList

def read_chi():
    dateFormat="%Y-%m-%dT%H:%M:%S"
    df = pds.read_csv('Chi.csv')
    df['End of the Ejecta'] = pds.to_datetime(df['End of the Ejecta'], format=dateFormat)
    df['MC'] = pds.to_datetime(df['MC'], format=dateFormat)
    df['MC'] = pds.to_datetime(df['MC'], format=dateFormat)
    df = df[['End of the Ejecta','MC']]
    df.dropna(inplace = True)
    df.reset_index(inplace = True)
    evtList = [Event(df['End of the Ejecta'][i], df['MC'][i])
               for i in range(0, len(df))]
    
    return evtList

def read_lepping():
    df = pds.read_csv('lepping.csv', dtype=str)
    df['Hour.1'] = pds.to_numeric(df['Hour.1'])
    df['hour1'] = df['Hour.1'].astype(int)
    df['hour1'] = df['hour1'].astype(str)
    df['minute1'] = (df['Hour.1']*60%60).astype(int)
    df['minute1'] = df['minute1'].astype(str)

    df['Hour'] = pds.to_numeric(df['Hour'])
    df['hour'] = df['Hour'].astype(int)
    df['hour'] = df['hour'].astype(str)
    df['minute'] = (df['Hour']*60%60).astype(int)
    df['minute'] = df['minute'].astype(str)

    df['Start'] = pds.to_datetime(df['Year']+df['Month']+df['Day']+df['hour']+df['minute'],format = '%y%b%d%H%M')

    df['End'] = pds.to_datetime(df['Year']+df['Month.1']+df['Day.1']+df['hour1']+df['minute1'],format = '%y%b%d%H%M', errors='coerce')
    evtList = [Event(df['Start'][i], df['End'][i])
               for i in range(0, len(df))]
    
    return evtList


def read_richardson():
    df = pds.read_csv('richardson.csv')
    dateFormat="%Y/%m/%d %H%M"
    
    df['disturbancestart'] = pds.to_datetime(df['disturbancestart'], format=dateFormat)
    df['ICMEend'] = pds.to_datetime(df['ICMEend'], format=dateFormat)
    df = df[['disturbancestart','ICMEend']]
    df.dropna(inplace = True)
    df.reset_index(inplace = True)
    evtList = [Event(df['disturbancestart'][i], df['ICMEend'][i])
               for i in range(0, len(df))]
    
    return evtList


def plotall(data, start, end, delta, typ):
    
         
     sns.set_style('darkgrid')
     sns.set_context('paper')
        
     fig=plt.figure(figsize=(9,5), dpi=150)
    
     data = data[start-datetime.timedelta(hours=delta):
                 end+datetime.timedelta(hours=delta)]
     
     #sharex means that zooming in works with all subplots
     ax1 = plt.subplot(511) 

     ax1.plot_date(data.index, data['Bx'],'-r',label='Bx',linewidth=0.5)
     ax1.plot_date(data.index, data['By'],'-g',label='By',linewidth=0.5)
     ax1.plot_date(data.index, data['Bz'],'-b',label='Bz',linewidth=0.5)
     ax1.plot_date(data.index, data['B'],'-k',label='Btotal',lw=0.5)
    
     #plot vertical lines
     ax1.plot_date([start,start],[-500,500],'-k',linewidth=1)                      
     ax1.plot_date([end,end],[-500,500],'-k',linewidth=1)
    
     plt.ylabel('B [nT]')
     plt.legend(loc=3,ncol=4,fontsize=8)
     
     ax1.set_ylim(-np.nanmax(data['B'])-5,np.nanmax(data['B'])+5)   
     ax1.set_xlim(data.index[0],data.index[-1])
     
     plt.setp(ax1.get_xticklabels(), visible=False)

     plt.title(typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M"))
    
     ax2 = plt.subplot(512,sharex=ax1) 
     ax2.plot_date(data.index,data['V'],'-k',label='V',linewidth=0.7)
    

     #plot vertical lines
     ax2.plot_date([start,start],[0,3000],'-k',linewidth=1)                    
     ax2.plot_date([end,end],[0,3000],'-k',linewidth=1)


     plt.ylabel('V [km/s]')
     
     #check plasma data exists
     if np.isnan(np.nanmin(data['V']))==False:
         ax2.set_ylim(np.nanmin(data['V'])-20,np.nanmax(data['V'])+100)   
     
     
     plt.setp(ax2.get_xticklabels(), visible=False)


     ax3 = plt.subplot(513,sharex=ax1) 
     ax3.plot_date(data.index,data['Np'],'-k',label='Np',linewidth=0.7)
     
     #plot vertical lines
     ax3.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax3.plot_date([end,end],[0,1000],'-k',linewidth=1)

     plt.ylabel('N [ccm-3]')
     
     if np.isnan(np.nanmin(data['Np']))==False:
         ax3.set_ylim(0,np.nanmax(data['Np'])+10)   
    
     
     plt.setp(ax3.get_xticklabels(), visible=False)  
    
    
     ax4= plt.subplot(514,sharex=ax1) 
     ax4.plot_date(data.index,data['T'],'-k',label='Tp',linewidth=0.7)
     
     #plot vertical lines
     ax4.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax4.plot_date([end,end],[0,1000],'-k',linewidth=1)
        
     plt.ylabel('Tp [K]')
     
     #ax4.set_ylim(0,20)   
    
    
     plt.setp(ax4.get_xticklabels(), visible=False)

     ax5 = plt.subplot(515,sharex=ax1) 
     ax5.plot_date(data.index,data['Beta'],'-k',label='Beta',linewidth=0.7)
     
     #plot vertical lines
     ax5.plot_date([start,start],[0,1000],'-k',linewidth=1)                       
     ax5.plot_date([end,end],[0,1000],'-k',linewidth=1)
        
     plt.ylabel('Beta')
     
     #ax5.set_ylim(0,20)   
    
    
     plt.setp(ax5.get_xticklabels(), visible=False)  

     
     plt.tight_layout()
     plt.show()


     #plotfile=typ+'ICME'+' data, start: '+start.strftime("%Y-%b-%d %H:%M")+'  end: '+end.strftime("%Y-%b-%d %H:%M")+'.png'
  
     #plt.savefig(plotfile)
     #print('saved as ',plotfile)
        