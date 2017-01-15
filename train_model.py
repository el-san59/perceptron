import numpy as np
import pandas as pd
import json
import pickle
from perceptron import Model, Layer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('learn.csv', index_col='id')
    mat = df.values

    with open('ig.csv', 'r') as f:
        features = np.array(json.load(f))
    ig = features[:, 1]

    x, y = mat[:, :-1], mat[:, -1]
    x = x[:, ig >= 0.985]
    y = y.reshape(-1, 1)
    print(x.shape[1])
    model = Model(Layer(256, x.shape[1], activation='relu'))
    model.add_layer(1, activation='sigmoid')
    test_error, train_errors, best_loss, best_epoch = model.fit(x, y, batch_size=32, nb_epoch=25)

    with open('model.pcl', 'wb') as f:
        pickle.dump(model, f)

    plt.plot(range(1, len(test_error)+1), test_error, label="test_loss")
    plt.plot(range(1, len(train_errors)+1), train_errors, label="train_loss")
    plt.plot(best_epoch, best_loss, '*', label="best_loss")
    plt.legend()
    plt.show()