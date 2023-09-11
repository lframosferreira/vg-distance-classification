import pandas as pd
import numpy as np
from classifier import MyClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split

# obs: dados faltantes de labels => preenchendo com label 7 (sem doença)

results = pd.read_json("results/nnd_n_avgs0.json")
cols = results.columns

labels_df = pd.read_json("exams_labels.json", lines=True)
labels_df["exam_id"].to_numpy()

labels = np.zeros(cols.shape[0])
for i, col in enumerate(cols):
    if col in labels_df["exam_id"].to_numpy():
        labels[i] = labels_df.iloc[i]["label"]
    else:
        labels[i] = 7


dfs = np.array(
    [pd.read_csv(f"data/dists/saida{i}.csv", header=None).to_numpy() for i in range(8)]
)

train_len = int(0.7 * cols.shape[0])

clf = MyClassifier()
clf.fit(dfs[:, :train_len, :train_len], labels[:train_len])

y_pred = clf.predict(dfs[:, (train_len + 1):, :train_len])

ac = accuracy_score(labels[(train_len + 1):], y_pred, average=None)
prec = precision_score(labels[(train_len + 1):], y_pred, average=None)
rec = recall_score(labels[(train_len + 1):], y_pred, average=None)
f1 = f1_score(labels[(train_len + 1):], y_pred, average=None)

