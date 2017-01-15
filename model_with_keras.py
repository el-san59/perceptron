from keras.engine import Input
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import json
import time
from matplotlib import pyplot as plt


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))


def create_model(input_params):
    input = Input(shape=(input_params, ))
    x = Dense(256, activation='relu')(input)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(input=input, output=output)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    with open('ig.csv', 'r') as f:
        features = np.array(json.load(f))
    ig = features[:, 1]

    df = pd.read_csv('learn.csv', index_col='id')
    mat = df.values
    x, y = mat[:, :-1], mat[:, -1]
    x = x[:, ig >= 0.985]

    with Profiler() as p:
        model = create_model(x.shape[1])
        a = model.fit(x, y, validation_split=.2, batch_size=32, nb_epoch=25, verbose=2,
                      callbacks=[EarlyStopping(patience=4)])

    test_errors = a.history['val_loss']
    train_errors = a.history['loss']
    plt.plot(range(len(test_errors)), test_errors)
    plt.plot(range(len(train_errors)), train_errors)
    plt.show()
    model.save('model2.h5')
