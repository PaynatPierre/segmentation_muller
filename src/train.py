from params import *
from model import *
from generator import *
import tensorflow as tf


def train():

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)

    train_generator, test_generator = create_generators2()

    if TRAIN_AUGMENTATION:
        model = micro_unet(input_size = (None,None,TRAINING_IMAGE_SIZE_CROP[2]))
    else:
        model = micro_unet()
    model.fit(x=train_generator, epochs=NBR_EPOCH,
              validation_data=test_generator,
              class_weight=None,shuffle= SHUFFLE_DATA, callbacks=[model_checkpoint_callback])

    return model