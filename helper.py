#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from classifier import MyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# obs: dados faltantes de labels => preenchendo com label 7 (sem doença)


# In[32]:


results = pd.read_json("results/nnd_n_avgs17.json")
cols = results.columns


# In[41]:


labels_df = pd.read_json("exams_labels.json", lines=True)
labels_df["exam_id"].to_numpy()


# In[48]:


labels = np.zeros(cols.shape[0])
for i, col in enumerate(cols):
    if col in labels_df["exam_id"].to_numpy():
        labels[i] = labels_df.iloc[i]["label"]
    else:
        labels[i] = 7


# In[ ]:


dfs = np.array(
    [pd.read_csv(f"dists/saida{i}.csv", header=None).to_numpy() for i in range(8)]
)

X_train, X_test, y_train, y_test = train_test_split(dfs, labels, test_size=0.3)

clf = MyClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"f1-score: {f1_score(y_test, y_pred)}")

