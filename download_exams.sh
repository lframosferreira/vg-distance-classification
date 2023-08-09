#!/bin/bash

for i in {0..17}; do
    wget https://zenodo.org/record/4916206/files/exams_part$i.zip?download=1
done
