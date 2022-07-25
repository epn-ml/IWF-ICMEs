import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class UnetGen(Sequence):
    
    """Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
        targettype: indicates if the target is of type multi, similarity or regression
    # Returns
        A [Sequence](/utils/#sequence) instance.
    """   
    
    def __init__(self, data, targets, length,
                 stride = 1,
                 start_index = 0,
                 end_index = None,
                 targettype = 'multi', # 'multi', 'simil', 'regres'
                 shuffle=False,
                 batch_size=128):
        
       
        if len(data) != len(targets):
            print('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))
            print('Target is assumed to be similarity parameter')
            targettype = 'simil'

        self.data = data
        self.targets = targets
        self.length = length
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.targettype = targettype
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                        self.stride, self.end_index + 1), self.stride)
        if self.targettype == 'multi':
            samples = np.array([self.data[row-self.length:row]
                                for row in rows], dtype = 'float64')
            
            targets = np.array([self.targets[row-self.length:row]
                                for row in rows], dtype = 'float64')
                 
            samples = np.expand_dims(samples, 2) 
            targets = np.expand_dims(targets, 2)
        if self.targettype == 'simil':
            samples = np.array([self.data[row - self.length:row]
                                for row in rows])
            targets = np.array([self.targets[i] for i,row in enumerate(rows)])
            

        if self.targettype == 'regres':
            samples = np.array([self.data[row - self.length:row]
                            for row in rows])
            targets = np.array([self.targets[row] for row in rows])

        
        
        return samples, targets