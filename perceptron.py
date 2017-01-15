import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    z = sigmoid(x)
    return z * (1. - z)


def relu(x):
    return x * (x > 0.)


def relu_prime(x):
    return 1. * (x > 0.)


def logloss(yt, yp):  # Cross entropy loss
    yp = np.clip(yp, 1e-6, 1 - 1e-6)
    return -yt * np.log(yp) - (1 - yt) * np.log(1 - yp)


def logloss_prime(yt, yp):
    yp = np.clip(yp, 1e-9, 1 - 1e-9)
    return -yt / yp + (1 - yt) / (1 - yp)


def generate_batches(x, y, batch_size, noise=False):
    indices = np.arange(0, len(x))
    random.shuffle(indices)
    shuffled_x, shuffled_y = x[indices], y[indices]
    if noise:
        shuffled_x = shuffled_x.copy()
        shuffled_x += np.random.normal(0, .1, shuffled_x.shape)
    for i in range(0, len(x), batch_size):
        yield shuffled_x[i:i + batch_size], shuffled_y[i:i + batch_size]


class StdNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Model:
    def __init__(self, input):
        self.list_layers = [input]
        self.lr = 0.01
        self.normalizer = None

    def add_layer(self, n_count, activation):
        self.list_layers.append(Layer(n_count, pred_layer=self.list_layers[-1], activation=activation))

    def fit(self, x, y, validation_split=0.2, batch_size=32, nb_epoch=10):
        x_train, y_train, x_test, y_test = train_test_split(x, y, 1 - validation_split)
        if self.normalizer is None:
            self.normalizer = StdNormalizer()
            x_train = self.normalizer.fit_transform(x_train)
        else:
            x_train = self.normalizer.transform(x_train)
        x_test = self.normalizer.transform(x_test)

        best_weights = None
        best_loss = None
        test_errors, train_errors = [], []

        for i in range(nb_epoch):
            p_train = self.predict(x_train, False)
            p_test = self.predict(x_test, False)
            train_loss = np.mean(logloss(y_train, p_train))
            test_loss = np.mean(logloss(y_test, p_test))
            test_errors.append(test_loss)
            train_errors.append(train_loss)
            print("Epoch #{}, loss_train: {}, loss_test: {}".format(i + 1, train_loss, test_loss))
            if best_loss is None or best_loss > test_loss:
                best_epoch = i+1
                print('Best loss in ' + str(i+1) + ' epoch')
                best_weights = []
                for layer in self.list_layers:
                    best_weights.append(layer.W)
                    best_weights.append(layer.bias)
                best_loss = test_loss

            for batch_x, batch_y in generate_batches(x_train, y_train, batch_size, noise=True):#start learning
                self.fit_batch(batch_x, batch_y)

        for i, layer in enumerate(self.list_layers):
            layer.W = best_weights[2 * i]
            layer.bias = best_weights[2 * i + 1]

        print('Training done! Best loss: {}'.format(best_loss))
        return test_errors, train_errors, best_loss, best_epoch

    def fit_batch(self, batch, labels):
        nabla_b, nabla_w = self.backprop(batch, labels)
        for layer, db, dw in zip(self.list_layers, nabla_b, nabla_w):
            layer.W -= self.lr * dw
            layer.bias -= self.lr * db

    def backprop(self, x, y):
        outputs = [x]
        for layer in self.list_layers:
            x = layer.call(x)
            outputs.append(x)
        deltas = []

        layer = self.list_layers[-1]
        delta = logloss_prime(y, x) * layer.prime(layer.s)
        deltas.append(delta)

        for i in range(1, len(self.list_layers)):
            layer = self.list_layers[-i - 1]
            delta = np.dot(delta, self.list_layers[-i].W.T) * layer.prime(layer.s)
            deltas.append(delta)
        deltas = list(reversed(deltas))

        grads = []
        for i in range(len(self.list_layers)):
            grads.append(np.dot(outputs[i].T, deltas[i]))
            deltas[i] = np.sum(deltas[i], axis=0)

        return deltas, grads

    def predict(self, x, normalize=True):
        if normalize:
            x = self.normalizer.transform(x)
        for layer in self.list_layers:
            x = layer.call(x)
        return x


class Layer:
    __activations__ = {'relu': relu, 'sigmoid': sigmoid}
    __primes__ = {'relu': relu_prime, 'sigmoid': sigmoid_prime}

    def __init__(self, n, input_shape=None, pred_layer=None, activation='sigmoid'):
        if input_shape is None and pred_layer is None:
            raise RuntimeError()
        if input_shape is None:
            input_shape = pred_layer.W.shape[1]

        self.W = np.random.rand(input_shape, n) / 100
        self.bias = np.random.rand(1, n) / 100
        self.activation = Layer.__activations__[activation]
        self.prime = Layer.__primes__[activation]
        self.s = 0

    def call(self, x):
        self.s = np.dot(x, self.W) + self.bias
        return self.activation(self.s)


def train_test_split(x, y, p=0.2):
    probs = np.random.random(len(x))
    return x[probs <= p], y[probs <= p], x[probs > p], y[probs > p]