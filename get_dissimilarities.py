import numpy
import h5py
import json

dists = {}
for i in range(10):
    with h5py.File(f"exams_part{i}.hdf5", "a") as file:
        pass