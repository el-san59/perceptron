import pandas as pd
import numpy as np
import json


def entropy(data, y):
    res = 0
    frequences = {}
    for line in data:
        if line[y] in frequences:
            frequences[line[y]] += 1
        else:
            frequences[line[y]] = 1
    for frequence in frequences.values():
        res += - frequence / len(data) * np.log2(frequence / len(data))
    return res


def gain(data, feature, y):
    subset_entropy = 0
    frequences = {}
    for line in data:
        if line[feature] in frequences:
            frequences[line[feature]] += 1
        else:
            frequences[line[feature]] = 1
    for value in frequences.keys():
        val_prob = frequences[value] / sum(frequences.values())
        data_subset = [line for line in data if line[feature] == value]
        subset_entropy += val_prob * entropy(data_subset, y)
    return entropy(data, y) - subset_entropy


if __name__ == "__main__":
    data = pd.DataFrame.from_csv('learn.csv')
    n = len(data.columns.values)
    gains = []
    for f in range(n - 1):
        g = gain(data.values, f, n - 1)
        gains.append((f, g))
        print(f, g)
    with open('ig.csv', 'w') as f:
        json.dump(gains, f)