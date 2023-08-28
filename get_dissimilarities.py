from functions import *
import h5py
import json
from ts2vg import NaturalVG
import scipy


for idx in range(18):
    d = {}
    with h5py.File(f"data/exams_part{idx}.hdf5", "r") as file:
        for i in range(file["exam_id"].shape[0]):
            d[int(file["exam_id"][i])] = {}
            tracings = file["tracings"][i]
            tracings = np.delete(tracings, [2, 3, 4, 5], axis=1)
            for j, lead in enumerate(tracings.T):
                graph = NaturalVG().build(lead).adjacency_matrix()
                nnd, avgs = network_node_dispersion(graph=graph)
                d[file["exam_id"][i]][j] = {}
                d[file["exam_id"][i]][j]["ndd"] = nnd
                d[file["exam_id"][i]][j]["avgs"] = avgs.tolist()

    with open(f"results/nnd_n_avgs{idx}.json", "w") as f:
        json.dump(d, f, indent=4)
    del d
