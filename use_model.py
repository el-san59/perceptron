import numpy as np
import pandas as pd
import json
import pickle


if __name__ == "__main__":
    with open('model.pcl', 'rb') as f:
        model = pickle.load(f)

    with open('ig.csv', 'r') as f:
        features = np.array(json.load(f))
    ig = features[:, 1]

    df = pd.read_csv('test.csv', index_col='id')
    mat = df.values
    mat = mat[:, ig >= 0.985]
    labels = model.predict(mat)
    new_df = pd.DataFrame(data=labels, index=df.index.values, columns=['label'])
    new_df.to_csv('res.csv', index_label='id')