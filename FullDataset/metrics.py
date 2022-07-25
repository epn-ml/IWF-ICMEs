import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f + y_pred_f)
    
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    #return (2 * intersection) / union 

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def true_skill_score(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    true_neg = K.sum((1-y_true_pos)*(1-y_pred_pos))
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    
    return ((true_pos+smooth)/(true_pos+false_neg + smooth)) - ((false_pos+smooth)/(false_pos+true_neg+smooth))
    
