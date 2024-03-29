from params import *
import tensorflow as tf
import os
import numpy as np
import scipy.io
from tensorflow import keras
from PIL import Image
import preprocess as prep

'''
This function takes a path directory and SHG a boolean indicating whether the data is generated using Second Harmonic Generation (SHG).
returns train and test data
'''
def create_generators2(data_path=DATASET_PATH, SHG = True):
    # Initialize lists to store file paths for train and test data
    train_mat_file_paths = []
    test_mat_file_paths = []

    #Loop over all files in the data path directory to separate the train and test dataset depending on the number of test dataset wanted
    for file_name in os.listdir(data_path): 
        if len(test_mat_file_paths) < TEST_DATASET_SIZE:
            test_mat_file_paths.append(os.path.join(data_path, file_name))
        else:
            train_mat_file_paths.append(os.path.join(data_path, file_name))
    
    # Create a train data generator using the train file paths with data augmentation applied
    train_data_generator = DataGeneratorClassifier2(train_mat_file_paths, SHG=SHG, transform=TRAIN_AUGMENTATION) 
    # Create a test data generator using the test file paths with no data augmentation applied
    test_data_generator = DataGeneratorClassifier2(test_mat_file_paths, image_size=TEST_IMAGE_SIZE, image_size_crop=TEST_IMAGE_SIZE_CROP, transform=False, SHG=SHG, test=True)
    return train_data_generator, test_data_generator


'''
This class generates data for training and testing a classifier model.
It inherits from a Sequence structure which guarantees that the network will only train once on each sample per epoch
'''
class DataGeneratorClassifier2(tf.keras.utils.Sequence):
    '''
    This __init__ method takes the following arguments :
        list_IDs: a list of IDs which are the file paths to the data.
        batch_size: the number of samples per batch which is set to a constant BATCH_SIZE value.
        image_size: the size of the images in the dataset which are images of 500 x 500 with 3 channels.
        image_size_crop: the size of the cropped images.
        shuffle: a boolean indicating whether to shuffle the data after each epoch.
        transform: a boolean indicating whether to apply data augmentation or not.
        nbr_classes: the number of classes in the classification problem : here it is supposed to be 10 classes.
        SHG: a boolean indicating whether the data is generated using Second Harmonic Generation (SHG).
        test: a boolean indicating whether the data is for testing or not : the default value is set to False for training data.
    '''
    def __init__(self, list_IDs, batch_size=BATCH_SIZE, image_size=TRAINING_IMAGE_SIZE, image_size_crop=TRAINING_IMAGE_SIZE_CROP, shuffle=SHUFFLE_DATA, transform=False, nbr_classes=NBR_CLASSES, SHG=True, test = False):
        self.test = test
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

    '''
    This method denotes the number of batches per epoch taking into consideration whether data augmentation is applied or not
    '''
    def __len__(self):
        if self.transform:
            return int(np.floor(len(self.list_IDs))/self.batch_size) * DATA_AUGMENTATION_AMPLIFICATION
        else:
            return int(np.floor(len(self.list_IDs))/self.batch_size)

    '''
    This method takes an integer index
    return : a batch X and the associated batch of label Y
    '''
    def __getitem__(self, index):
        if self.transform: # to act as if there were 200 times more data in one epoch than reality in order to do less validation tests and waste less time than if we really validated the 4 data
            index = index % int(np.floor(len(self.list_IDs))/self.batch_size)
            if index == 0:
                self.on_epoch_end()

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # shufle indexes

        X = np.zeros((self.batch_size, self.image_size_crop[0], self.image_size_crop[1], self.image_size_crop[2])) # a data batch
        Y = np.zeros((self.batch_size, self.image_size_crop[0], self.image_size_crop[1])) # a label batch corresponding to the associated class for each pixel 
        ''' 
            self.batch_size : gives a batch of data
            self.image_size_crop[0] : size X of the image = 500 at first
            self.image_size_crop[1] : size Y of the image = 500 at first
            self.image_size_crop[2] : 16 chanels of one image
        '''

        for i, idx in enumerate(indexes): # Match the list of data to the chosen indexes
            X[i,:,:,:] = self.X_data[idx]
            Y[i,:,:] = self.Y_data[idx]
        
        if self.transform:
            X,Y = self.batch_augmentation(X,Y)
        

        # if not self.test:
        #     for i in range(X.shape[3]):
        #         im = Image.fromarray(np.array(X)[0,:,:,i]*255)
        #         if im.mode != 'RGB':
        #             im = im.convert('RGB')
        #         im.save(f"./tmp/train_data_canal_{i}.jpeg")

        #     tmp = tf.one_hot(Y.astype(np.int32), NBR_CLASSES, axis=-1)
        #     for i in range(tmp.shape[3]):
        #         im = Image.fromarray(np.array(tmp)[0,:,:,i]*255)
        #         if im.mode != 'RGB':
        #             im = im.convert('RGB')
        #         im.save(f"./tmp/train_label_classe_{i}.jpeg")

        return X,tf.one_hot(Y.astype(np.int32), NBR_CLASSES, axis=-1) #classification
        # return X,Y/NBR_CLASSES #regression


    '''
    This method takes a non augmented batch X and the associated batch of label Y
    applies the data augmentation for one batch
        - mirror effect following horizontal axis
        - mirror effect following vertical axis
        - rotation effect
        - gaussian noise
        - random crop horizontal
        - random crop vertical
    '''
    def batch_augmentation(self, X, Y):
        new_X = np.zeros(X.shape)
        new_Y = np.zeros(Y.shape)
        
        for i in range(len(X)):
            tmp_Xi = X[i,:,:,:]
            tmp_Yi = Y[i,:,:]

            # mirror effect following horizontal axis
            epsilon = np.random.rand()
            if epsilon > 0.5:
                tmp_Xi = np.flip(tmp_Xi,0)
                tmp_Yi = np.flip(tmp_Yi,0)

            # mirror effect following vertical axis
            epsilon = np.random.rand()
            if epsilon > 0.5:
                tmp_Xi = np.flip(tmp_Xi,1)
                tmp_Yi = np.flip(tmp_Yi,1)

            # rotation effect
            epsilon = np.random.randint(4)
            tmp_Xi = np.rot90(tmp_Xi,epsilon, (0,1))
            tmp_Yi = np.rot90(tmp_Yi,epsilon, (0,1))

            # #gaussian noise
            # for j in range(tmp_Xi.shape[-1]):
            #     std = (np.max(tmp_Xi[:,:,j]) - np.min(tmp_Xi[:,:,j]))*0.03*np.random.rand()
            #     noise = np.random.normal(0,std,tmp_Xi[:,:,j].shape)
            #     tmp_Xi[:,:,j] = tmp_Xi[:,:,j] + noise

            new_X[i,:,:,:] = tmp_Xi
            new_Y[i,:,:] = tmp_Yi

        # # random crop horizontal
        # epsilon = np.random.randint(((MAX_CROP_CONSERVATION_FACTOR)*self.image_size_crop[1])//DIVISIBILITY_FACTOR)
        # nbr_pixel_to_crop = epsilon*DIVISIBILITY_FACTOR

        # if nbr_pixel_to_crop != 0:
        #     nbr_pixel_to_crop_left = np.random.randint(nbr_pixel_to_crop + 1)
        #     nbr_pixel_to_crop_right = nbr_pixel_to_crop - nbr_pixel_to_crop_left

        #     new_X = new_X[:,:,nbr_pixel_to_crop_left:new_X.shape[2]-nbr_pixel_to_crop_right,:]
        #     new_Y = new_Y[:,:,nbr_pixel_to_crop_left:new_Y.shape[2]-nbr_pixel_to_crop_right]

        # # random crop vertical
        # epsilon = np.random.randint(((MAX_CROP_CONSERVATION_FACTOR)*self.image_size_crop[0])//DIVISIBILITY_FACTOR)
        # nbr_pixel_to_crop = epsilon*DIVISIBILITY_FACTOR

        # if nbr_pixel_to_crop != 0:
        #     nbr_pixel_to_crop_top = np.random.randint(nbr_pixel_to_crop + 1)
        #     nbr_pixel_to_crop_bottom = nbr_pixel_to_crop - nbr_pixel_to_crop_top

        #     new_X = new_X[:,nbr_pixel_to_crop_top:new_X.shape[1]-nbr_pixel_to_crop_bottom,:,:]
        #     new_Y = new_Y[:,nbr_pixel_to_crop_top:new_Y.shape[1]-nbr_pixel_to_crop_bottom,:]

        return new_X, new_Y

    '''
    This method updates indexes after each epoch in order to not always take the data in the same order at each epoch
    '''
    def on_epoch_end(self): 
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    '''
    This method takes a list of labels
    returns a set of (reference) images evenly scaled across classes "classes" and containing as many images as the defined data 
    '''
    def uniform_label_discretisation(self, labels):
        shape = np.shape(labels)
        n = shape[0]
        bins = [(x+1)/self.nbr_classes for x in range(self.nbr_classes-1)]
        discret_ = keras.layers.experimental.preprocessing.Discretization(bins = bins)
        data_discret = discret_(labels)
        data_discret = data_discret.numpy()
        return data_discret

    '''
    This method loads the data and applies to it the transformations before using them for the training
    '''
    def load_data(self):
        # Y_data_tmp = np.zeros((len(self.list_IDs), self.image_size[0], self.image_size[1]))
        Y_data_tmp = np.zeros((len(self.list_IDs), self.image_size_crop[0], self.image_size_crop[1]))

        nbr_pixel_to_crop_first_axe = self.image_size[0]%DIVISIBILITY_FACTOR
        nbr_pixel_to_crop_second_axe = self.image_size[1]%DIVISIBILITY_FACTOR

        nbr_pixel_to_crop_left = nbr_pixel_to_crop_first_axe//2
        nbr_pixel_to_crop_right = nbr_pixel_to_crop_first_axe - nbr_pixel_to_crop_left
        nbr_pixel_to_crop_top = nbr_pixel_to_crop_second_axe//2
        nbr_pixel_to_crop_bottom = nbr_pixel_to_crop_second_axe - nbr_pixel_to_crop_top

        for i, path in enumerate(self.list_IDs):
            data = scipy.io.loadmat(path)

            # for j in range(data['Fin_MM_avgZ'].shape[2]):
            #     for l in range(data['Fin_MM_avgZ'].shape[3]):
            #         self.X_data[i,:,:,data['Fin_MM_avgZ'].shape[2]*j+l] = prep.norm_data(prep.Grubbs_data(data['Fin_MM_avgZ'][:,:,j,l]))[nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]

            for j in range(data['Fin_MM_avgZ'].shape[2]):
                for l in range(data['Fin_MM_avgZ'].shape[3]):
                    if data['Fin_MM_avgZ'].shape[2]*j+l == 0:
                        self.X_data[i,:,:,0] = prep.norm_data(prep.Grubbs_data(data['Fin_MM_avgZ'][:,:,j,l]))[nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]
                    elif data['Fin_MM_avgZ'].shape[2]*j+l == 11:
                        self.X_data[i,:,:,1] = prep.norm_data(prep.Grubbs_data(data['Fin_MM_avgZ'][:,:,j,l]))[nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]
                    elif data['Fin_MM_avgZ'].shape[2]*j+l == 14:
                        self.X_data[i,:,:,2] = prep.norm_data(prep.Grubbs_data(data['Fin_MM_avgZ'][:,:,j,l]))[nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]

            if self.SGH:
                Y_data_tmp[i,:,:] = prep.norm_data(prep.Grubbs_data(data['SHGZ'][nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom),0]))
            else:
                Y_data_tmp[i,:,:] = prep.norm_data(prep.Grubbs_data(data['TPEFZ'][nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom),0]))
        
            # if self.SGH:
            #     Y_data_tmp[i,:,:] = data['SHGZ'][nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom),0]
            # else:
            #     Y_data_tmp[i,:,:] = data['TPEFZ'][nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom),0]
        

        self.Y_data = self.uniform_label_discretisation(Y_data_tmp)#[:,nbr_pixel_to_crop_left:(-nbr_pixel_to_crop_right),nbr_pixel_to_crop_top:(-nbr_pixel_to_crop_bottom)]


'''
This function enables us to see the results of our data augmentation
'''
def show_data():
    train_gen, test_gen = create_generators2(SHG=True)

    x,y = train_gen.__getitem__(0)
    print('x shape is : ' + str(x.shape))
    print('y shape is : ' + str(y.shape))

    print(x)
    print(np.min(x))
    print(np.max(x))
    for j in range(16):
        im = Image.fromarray(x[0,:,:,j]*255)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(f"data_canal_{j}.jpeg")

    for i in range(len(y)):
        im = Image.fromarray((np.argmax(y[i,:,:], axis=-1).astype(np.uint8))*25)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(f"label_{i}.jpeg")

if __name__ == "__main__":
    show_data()
