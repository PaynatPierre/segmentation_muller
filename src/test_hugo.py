import os 
import scipy.io
import preprocess as prep
import numpy as np 
from PIL import Image
import tensorflow 
from tensorflow import keras 
import generator as gen 
from sklearn.metrics import confusion_matrix
from params import *
#considérer argmax pour Y_pred
#flatten sur les Y_pred, Y_true 
def IoU_score(Y_true,Y_pred):
    #Y_pred is a matrix that contains all the prediction (for a batch for instance)
    #Y_real is the matrix of the real labels
    
    #Number of sample in Y_real
    nbr_sample = len(Y_true)
    #We will store all the result here
    result = np.zeros((NBR_CLASSES,NBR_CLASSES))
    #number total of pixel in the batch 
    total_pix = 0
    #create a list of weights
    list_weights = [0 for i in range(NBR_CLASSES)]

    #Flatten sur les Y_pred, Y_true 
    for i in range(nbr_sample):
        y_pred = Y_pred[i].flatten()
        y_true = Y_true[i].flatten()
        
        for j in range(NBR_CLASSES):
            list_weights[j] += np.count_nonzero(y_true == j)

        current = confusion_matrix(y_true, y_pred,labels=[i for i in range(NBR_CLASSES)])
        result = result + current 

        total_pix = total_pix + len(y_pred)
    
    list_weights = np.array(list_weights)/total_pix
    #print('weights',list_weights)
    intersection = np.diag(current)
    #print('intersection',intersection)
    #array which contains the sum over the columns
    ground_truth_set = current.sum(axis=1)
    #print('ground_truth_set',ground_truth_set)
    #aray which contains the sum over the rows
    predicted_set = current.sum(axis=0)
    #print('predicted_set', predicted_set)
    union = ground_truth_set + predicted_set - intersection
    #print('union',union)
    IoU = intersection / union.astype(np.float32)
    #drop na in IoU list
    indices = ~np.isnan(IoU)
    return np.average(IoU[indices],weights=list_weights[indices])
    
array_1 = np.random.randint(10, size=(3,6,4))
array_2 = array_1 


if __name__=='__main__':
    print(IoU_score(array_1,array_2))


