import scipy.io
import numpy as np
import h5py
import os

# data= h5py.File('../DataPola.h5','r')
# print(np.size(data['Im1']))
for path in os.listdir('./../data/'):
    f = scipy.io.loadmat('./../data/' + path)
    print(f['TPEFZ'])
    print('execution end')