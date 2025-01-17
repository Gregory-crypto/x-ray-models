#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 15 10:26:24 2025

@author: grigorii
"""
import argparse
from datasets import Images
from viz import heatmap_cv, plot, boxplot_cv
from models import pca, tsne, umap, cross_v, dummy
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

args = parser.parse_args()

# LOAD DATA
path = "./data/"  # to the parent folder of the dataset
states = ["NORMAL", "PNEUMONIA"] # different states

image = Images(path)

data_scaled, labels = image.linear_data("train/", states) # to be honest in the future I will need to implement train and val sets

# Visualization of dimentional reduction
pca_data = pca(data_scaled = data_scaled)
tsne_data = tsne(data_scaled = data_scaled)
umap_data = umap(data_scaled = data_scaled)

plot(dim_data = pca_data, type = "PCA", labels = labels)
plot(dim_data = tsne_data, type = "tSNE", labels = labels)
plot(dim_data = umap_data, type = "UMAP", labels = labels)

# Nested CV with SVC
pivot_table_umap, umap_scores = cross_v(data = umap_data, labels = labels) # using umap coordinates
heatmap_cv(pivot_table = pivot_table_umap, mode = "umap")

pivot_table_raw, raw_scores = cross_v(data = data_scaled, labels = labels) # using full data (scaled)
heatmap_cv(pivot_table = pivot_table_raw, mode = "raw")

dummy_scores = dummy(data_scaled = data_scaled, labels = labels) # get estimations from naive model

score_table = np.vstack((dummy_scores, umap_scores, raw_scores)).T
score_table = pd.DataFrame(score_table)
score_table.columns = ["Dummy Scores", "UMAP Model Scores", "Raw Model Scores"]
score_table = score_table.melt(var_name = "Model Type", value_name = "Score")
boxplot_cv(score_table)