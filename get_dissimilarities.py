from functions import *
import h5py
import json
from ts2vg import NaturalVG
import scipy

d = {}
with h5py.File("data/exams_part17.hdf5", "r") as file:
    d["exam_id"] = int(file["exam_id"][0])
    tracings = file["tracings"][0]
    tracings = np.delete(tracings, [2, 3, 4, 5], axis=1)
    for i, lead in enumerate(tracings.T):
        graph = NaturalVG().build(lead).adjacency_matrix()
        nnd, avgs = network_node_dispersion(graph=graph)
        d[i] = {}
        d[i]["nnd"] = nnd
        d[i]["avgs"] = list(avgs)

with open("dists.json", "w") as f:
    json.dump(d, f, indent=4)
