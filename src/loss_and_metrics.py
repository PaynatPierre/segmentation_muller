from re import L
import keras.backend as K
import tensorflow as tf
from params import *
import numpy as np

def IoULoss(targets, inputs, smooth=1e-6):
    # print(targets.shape)
    # print('------------------------------')
    # print(inputs.shape)
    #flatten label and prediction tensors

    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    result = 1 - IoU
    # print(result.numpy())
    return result

def ponderation_IoULoss(targets, inputs, smooth=1e-6):
    # print(targets.shape)
    # print('------------------------------')
    # print(inputs.shape)
    #flatten label and prediction tensors
    IoU = 0
    for i in range(NBR_CLASSES):
        IoU = IoU + IoULoss(targets[:,:,:,i],inputs[:,:,:,i])
    
    result=IoU/NBR_CLASSES
    
    return result

import tensorflow as tf
import keras.backend as K


def IoU_metric(targets, inputs, smooth=1e-6):
    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

def IoU_metric_regression(targets, inputs, smooth=1e-6):
    targets = tf.one_hot(tf.cast((targets*NBR_CLASSES), tf.int32), NBR_CLASSES, axis=-1)
    inputs = tf.one_hot(tf.cast((inputs*NBR_CLASSES), tf.int32), NBR_CLASSES, axis=-1)

    inputs = tf.expand_dims(K.flatten(inputs),1)
    targets = tf.expand_dims(K.flatten(targets),0)
    
    tmp=K.dot(targets, inputs)
    intersection = K.sum(tmp)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU