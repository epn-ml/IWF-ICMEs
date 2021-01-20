
import pandas as pds
import datetime
import numpy as np
import time
import event as evt
import seaborn as sns
from joblib import Parallel, delayed
import numpy.random as random
from lmfit import models
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    
    if normalize:
        cm = cm.astype('float') / sum(sum(cm))

    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def get_label1(pred, test_clouds):
    x = np.size(pred)
    label = np.zeros(x)
    u = 0
    for p in np.arange(0, np.size(pred)):
            if (pred.index[p]>test_clouds[u].begin) and (pred.index[p]<test_clouds[u].end):
                label[p] = 1
                
                if (pred.index[p+1]>= test_clouds[u].end):
                    u = u + 1
                    
                    if u == len(test_clouds):
                        break
    return label


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


def gatherAsEventEnd(eventList, thres):
    '''
    gather neighboured events together to form the real events
    events are merged to their neighbour if distance between them is <thres
    (expressed in hours)
    '''
    i = 0
    while i < len(eventList)-1:
        if eventList[i+1].begin-eventList[i].end < datetime.timedelta(hours
                                                                      =thres):
            eventList[i] = evt.merge(eventList[i], eventList[i+1])
            eventList.pop(i+1)
        else:
            i += 1
    return eventList


def gatherAsEventBegin(eventList, thres):
        '''
        gather neighboured events together to form the real events
        events are merged to their neighbour if distance between beginning
        is <thres
        (expressed in hours)
        '''
        eventsToRemove = []
        for i in reversed(range(1, len(eventList))):
            if eventList[i].begin-eventList[i-1].begin < datetime.timedelta(hours=thres):
                eventsToRemove.append(eventList[i])
                eventList[i-1] = evt.merge(eventList[i-1], eventList[i])
        while len(eventsToRemove) > 0:
            eventList.remove(eventsToRemove[-1])
            eventsToRemove.remove(eventsToRemove[-1])
        return eventList


def removeCreepy(eventList, thres=2):
    '''
    For a given list, remove the element whose duration is under the threshold
    '''
    return [x for x in eventList if x.duration > datetime.timedelta(hours=thres)]


def turn_peaks_to_clouds(serie, thres, freq=10,
                         durationOfCreepies=2.5, n_jobs=1):
    '''
    Transforms the output serie of a pipeline into a complete list of events
    '''
    events = []
    pred = pds.Series(index=pds.date_range(serie.index[0],
                                           serie.index[-1],
                                           freq=(str(freq)+'T')),
                      data=np.nan)

    pred[serie.index[serie > thres]] = 1
    pred[serie.index[serie < thres]] = 0

    pred = pred.interpolate()
    
    intervals = makeEventList(pred, 1, freq)
    intervals = removeCreepy(intervals, durationOfCreepies)
    
    results = Parallel(n_jobs=n_jobs)(delayed(_turn_intervals_to_Event)(event, serie) for event in intervals)
    
    for fls in results:
        events.extend(fls)
    return events, pred


def _generate_model(spec):
    composite_model = None
    params = None
    x = spec['time']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def _turn_intervals_to_Event(event, serie):
    '''
    Find events in a temporal interval that contain one or several events
    '''
    spec = {
     'time': np.arange(0, len(serie[event.begin:event.end])),
     'y': serie[event.begin:event.end].values,
     'model': [
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'},
        {'type': 'GaussianModel'}
     ]
    }
    model, params = _generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['time'])
    fitted_integral = output.best_fit
    pos = find_peaks(fitted_integral)[0]
    width = peak_widths(fitted_integral, pos)

    ref_index = serie[event.begin:event.end].index
    clouds = [evt.Event(ref_index[int(width[2][x])], ref_index[int(width[3][x])]) for x in np.arange(0, len(width[0]))]
    return clouds
#    try:
#        model, params = _generate_model(spec)
#        output = model.fit(spec['y'], params, x=spec['time'])
#        fitted_integral = output.best_fit
#        pos = find_peaks(fitted_integral)[0]
#        width = peak_widths(fitted_integral, pos)
#
#        ref_index = serie[event.begin:event.end].index
#        clouds = [evt.Event(ref_index[int(width[2][x])], ref_index[int(width[3][x])]) for x in np.arange(0, len(width[0]))]
#        return clouds
#    except:
#        return []


def removeHoles(eventList, holes):
    return [x for x in eventList if max(evt.overlapWithList(x, holes))/x.duration<0.05]
