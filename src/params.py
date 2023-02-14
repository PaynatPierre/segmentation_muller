# training parametres
BATCH_SIZE = 2 ##on scanne toutes les données 2 par 2 , à la fin on passe à une autre epoch ?
#mettre de la gradient accumulation
#en général gros batch size pour bien moyenner les données
#il faudrait faire une retropopagation sur plusieurs batch 
SHUFFLE_DATA = True
DIVISIBILITY_FACTOR = 16
PRETRAINED_WEIGHTS = None
LEARNING_RATE = 0.1
NBR_EPOCH = 500
CHECKPOINT_PATH = './../models/try_11/my_checkpoint.ckpt'
SAVE_TRAIN_SAMPLE_INTERVAL = 100
GRADIANT_ACCUMULATION = 16

# dataset parametres
DATASET_PATH = './../data/'
TEST_DATASET_SIZE = 7
FULL_DATASET_SIZE = 11
TRAINING_IMAGE_SIZE = (500,500,3)
TEST_IMAGE_SIZE = (500,500,3)
NBR_CLASSES = 10
TRAINING_IMAGE_SIZE_CROP = (TRAINING_IMAGE_SIZE[0] - TRAINING_IMAGE_SIZE[0]%DIVISIBILITY_FACTOR,TRAINING_IMAGE_SIZE[1] - TRAINING_IMAGE_SIZE[1]%DIVISIBILITY_FACTOR,TRAINING_IMAGE_SIZE[2])
TEST_IMAGE_SIZE_CROP = (TEST_IMAGE_SIZE[0] - TEST_IMAGE_SIZE[0]%DIVISIBILITY_FACTOR,TEST_IMAGE_SIZE[1] - TEST_IMAGE_SIZE[1]%DIVISIBILITY_FACTOR,TEST_IMAGE_SIZE[2])

# data augmentation parametres
TRAIN_AUGMENTATION = True
DATA_AUGMENTATION_AMPLIFICATION = 200 # act as if there were 200 times more data than reality in order to do less validation tests and waste less time than if we really validated the 4 data
MAX_CROP_CONSERVATION_FACTOR = 2/3