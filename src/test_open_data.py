import scipy.io
import numpy as np
import h5py
import os
from PIL import Image

data= h5py.File('../DataPola.h5','r')
print(data['Im1/TPEF'].shape)

for i in range(1,12):
    im = Image.fromarray(np.array(data[f'Im{i}/TPEF'])*25)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(f"label_TPEF_{i}_nelson.jpeg")

    im = Image.fromarray(np.array(data[f'Im{i}/SHG'])*25)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(f"label_SHG_{i}_nelson.jpeg")


# for path in os.listdir('./../data/'):
#     f = scipy.io.loadmat('./../data/' + path)
#     print(f['TPEFZ'])
#     print('execution end')