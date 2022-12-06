from params import *
from model import *
from generator import *
import tensorflow as tf


def train():

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)

    train_generator, test_generator = create_generators2()
    model = micro_unet()
    model.fit(x=train_generator, epochs=NBR_EPOCH,
              validation_data=test_generator,
              class_weight=None,shuffle= SHUFFLE_DATA, callbacks=[model_checkpoint_callback])

    return model