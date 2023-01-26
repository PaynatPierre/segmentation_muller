from tensorflow.keras.callbacks import Callback
from PIL import Image
import numpy as np
from params import *

class SavePredictCallback(Callback):

    def __init__(self, xtrain, **kwargs):
        super().__init__(**kwargs)
        self.save_index = 0
        self.xtrain = xtrain


    def on_train_batch_end(self, batch, logs=None):
        pred = self.model(self.xtrain.__getitem__(batch)[0])

        for X in pred:
            if self.save_index % SAVE_TRAIN_SAMPLE_INTERVAL == 0:
                tmp = np.argmax(np.array(X), axis=-1).astype(float)*25
                print(tmp.shape)
                im = Image.fromarray(tmp)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(f"./tmp/train_result_idx_{self.save_index}.jpeg")
            self.save_index += 1