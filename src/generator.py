from params import *
import tensorflow as tf
import os
import numpy as np
import scipy.io
from tensorflow import keras
from PIL import Image
import preprocess as prep


def create_generators2(data_path=DATASET_PATH, SHG = True):
    train_mat_file_paths = []
    test_mat_file_paths = []

    for file_name in os.listdir(data_path):
        if len(test_mat_file_paths) < TEST_DATASET_SIZE:
            test_mat_file_paths.append(os.path.join(data_path, file_name))
        else:
            train_mat_file_paths.append(os.path.join(data_path, file_name))
     
    train_data_generator = DataGeneratorClassifier2(train_mat_file_paths, SHG=SHG)
    test_data_generator = DataGeneratorClassifier2(test_mat_file_paths, image_size=TEST_IMAGE_SIZE, image_size_crop=TEST_IMAGE_SIZE_CROP, transform=False, SHG=SHG)
    return train_data_generator, test_data_generator


class DataGeneratorClassifier2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=BATCH_SIZE, image_size=TRAINING_IMAGE_SIZE, image_size_crop=TRAINING_IMAGE_SIZE_CROP, shuffle=SHUFFLE_DATA, transform=True, nbr_classes=NBR_CLASSES, SHG=True):
        'Initialisation'
        self.image_size = image_size
        self.image_size_crop = image_size_crop
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.SGH = SHG
        self.on_epoch_end()
        self.transform=transform
        self.nbr_classes = nbr_classes
        self.X_data = np.zeros((len(list_IDs), self.image_size_crop[0], self.image_size_crop[1], self.image_size_crop[2]))
        self.Y_data = np.zeros((len(list_IDs), self.image_size_crop[0], self.image_size_crop[1]))
        self.load_data()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs))/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X = np.zeros((self.batch_size, self.image_size_crop[0], self.image_size_crop[1], self.image_size_crop[2]))
        Y = np.zeros((self.batch_size, self.image_size_crop[0], self.image_size_crop[1]))

        for i, idx in enumerate(indexes):
            X[i,:,:,:] = self.X_data[idx]
            Y[i,:,:] = self.Y_data[idx]
        
        return X,Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def uniform_label_discretisation(self, labels):
        '''Retourne un set d'images (de référence) échelonnées sur 'classes' classes
        de manière uniforme et contenant autant d'images que le set data'''
        shape = np.shape(labels)
        n = shape[0]
        bins = [(x+1)/self.nbr_classes for x in range(self.nbr_classes-1)]
        discret_ = keras.layers.experimental.preprocessing.Discretization(bins = bins)
        data_discret = discret_(labels)
        data_discret = data_discret.numpy()

        return data_discret

    def load_data(self):

        Y_data_tmp = np.zeros((len(self.list_IDs), self.image_size[0], self.image_size[1]))

        nbr_pixel_to_crop_first_axe = self.image_size[0]%DIVISIBILITY_FACTOR
        nbr_pixel_to_crop_second_axe = self.image_size[1]%DIVISIBILITY_FACTOR

        nbr_pixel_to_crop_left = nbr_pixel_to_crop_first_axe//2
        nbr_pixel_to_crop_right = nbr_pixel_to_crop_first_axe - nbr_pixel_to_crop_left
        nbr_pixel_to_crop_top = nbr_pixel_to_crop_second_axe//2
        nbr_pixel_to_crop_bottom = nbr_pixel_to_crop_second_axe - nbr_pixel_to_crop_top

        for i, path in enumerate(self.list_IDs):
            data = scipy.io.loadmat(path)

            for j in range(data['Fin_MM_avgZ'].shape[2]):
                for l in range(data['Fin_MM_avgZ'].shape[3]):
                    self.X_data[i,:,:,data['Fin_MM_avgZ'].shape[2]*j+l] = prep.norm_data(prep.Grubbs_data(data['Fin_MM_avgZ'][:,:,j,l]))[nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]

            if self.SGH:
                Y_data_tmp[i,:,:] = prep.norm_data(prep.Grubbs_data(data['SHGZ'][:,:,0]))
            else:
                Y_data_tmp[i,:,:] = prep.norm_data(prep.Grubbs_data(data['TPEFZ'][:,:,0]))
        
        self.Y_data = self.uniform_label_discretisation(Y_data_tmp)[:,nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]

def show_data():
    train_gen, test_gen =create_generators2(SHG=True)

    x,y = train_gen.__getitem__(0)
    print('x shape is : ' + str(x.shape))
    print('x shape is : ' + str(y.shape))

    print(x)
    print(np.min(x))
    print(np.max(x))
    for j in range(16):
        im = Image.fromarray(x[0,:,:,j]*255)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(f"data_canal_{j}.jpeg")

    for i in range(len(y)):
        im = Image.fromarray(y[i,:,:]*25)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(f"label_{i}.jpeg")

if __name__ == "__main__":
    show_data()
