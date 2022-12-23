from train import train
import tensorflow as tf


print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
with tf.device('GPU:0'):
    model = train()