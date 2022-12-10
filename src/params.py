# training parametres
BATCH_SIZE = 1
SHUFFLE_DATA = True
DIVISIBILITY_FACTOR = 16
PRETRAINED_WEIGHTS = None
LEARNING_RATE = 0.001
NBR_EPOCH = 10
CHECKPOINT_PATH = './../models/try_1/my_checkpoint.ckpt'

# dataset parametres
DATASET_PATH = './../data/'
TEST_DATASET_SIZE = 7
FULL_DATASET_SIZE = 11
TRAINING_IMAGE_SIZE = (500,500,16)
TEST_IMAGE_SIZE = (500,500,16)
NBR_CLASSES = 10
TRAINING_IMAGE_SIZE_CROP = (TRAINING_IMAGE_SIZE[0] - TRAINING_IMAGE_SIZE[0]%DIVISIBILITY_FACTOR,TRAINING_IMAGE_SIZE[1] - TRAINING_IMAGE_SIZE[1]%DIVISIBILITY_FACTOR,TRAINING_IMAGE_SIZE[2])
TEST_IMAGE_SIZE_CROP = (TEST_IMAGE_SIZE[0] - TEST_IMAGE_SIZE[0]%DIVISIBILITY_FACTOR,TEST_IMAGE_SIZE[1] - TEST_IMAGE_SIZE[1]%DIVISIBILITY_FACTOR,TEST_IMAGE_SIZE[2])

# data augmentation parametres
TRAIN_AUGMENTATION = True
DATA_AUGMENTATION_AMPLIFICATION = 1000
MAX_CROP_CONSERVATION_FACTOR = 2/3