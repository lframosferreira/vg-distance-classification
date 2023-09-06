#!/usr/bin/env python

import pandas as pd
import numpy as np
df: pd.DataFrame = pd.read_json("results/nnd_n_avgs17.json")

""" dfs = [pd.read_json(f"nnd_n_avgs{i}.json") for i in range(18)]
all_data = pd.concat(dfs, axis=1)
all_data.to_json("nnd_n_avgs.json") """

cols = df.columns
df = df.to_numpy()


from classifier import *
from functions import *

# distance_matrix para cada lead
distance_matrix = np.zeros((8, df.shape[1], df.shape[1]))

for i in range(df.shape[1]):
    for j in range(df.shape[1]):
        if i == j:
            continue
        for k in range(8):
            if distance_matrix[k][i][j] != 0:
                continue
            nnd_G: float = df[k, i]["ndd"]
            nnd_H: float = df[k, j]["ndd"]
            averages_G: list[float] = df[k, i]["avgs"]
            averages_H: list[float] = df[k, j]["avgs"]
            aux: float = dissimilarity_measure_from_nnd_n_avgs(nnd_G, nnd_H, averages_G, averages_H)
            distance_matrix[k][i][j] = distance_matrix[k][j][i] = aux
    print(i)

np.savetxt("saida.csv", distance_matrix, delimiter=',')
