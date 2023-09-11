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


# In[57]:


a = np.zeros((500, 500))
a.shape
a[200:].shape


# In[ ]:


dfs = np.array(
    [pd.read_csv(f"dists/saida{i}.csv", header=None).to_numpy() for i in range(8)]
)


clf = MyClassifier()
clf.fit(np.array(a[:4000][:, :4000] for a in dfs), labels[:4000])

y_pred = clf.predict(np.array([a[4001:] for a in dfs]))
print(f"f1-score: {f1_score(labels[4001:], y_pred)}")

