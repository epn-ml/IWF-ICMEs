"""
Data Generator
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pandas as pds

def parse_image(img_path, image_shape):
    image_rgb = pds.read_csv(img_path,header = None,index_col=[0]).values
    image_rgb = np.reshape(image_rgb, image_shape)
    
    return image_rgb

def parse_mask(mask_path, image_shape):
    mask = pds.read_csv(mask_path,header = None, index_col=[0]).values
    mask = np.reshape(mask, (image_shape[0],1,1))
    weight = mask + 0.1
    return mask, weight

class DataGen(Sequence):
    def __init__(self, weight, image_shape, images_path, masks_path, batch_size=8):
        self.image_shape = image_shape
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.on_epoch_end()
        self.weight = weight

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index*self.batch_size

        images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
        masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        masks_batch = []
        weight_batch = []

        for i in range(len(images_path)):
            ## Read image and mask
            image = parse_image(images_path[i], self.image_shape)
            mask, weight = parse_mask(masks_path[i], self.image_shape)

            images_batch.append(image)
            masks_batch.append(mask)
            weight_batch.append(weight)

        if self.weight == True:
            return np.array(images_batch), np.array(masks_batch), np.array(weight_batch)
        else:
            return np.array(images_batch), np.array(masks_batch)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))
 