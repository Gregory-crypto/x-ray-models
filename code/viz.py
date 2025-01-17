#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 15 10:26:24 2025

@author: grigorii

VARIABLES
type - type of compactization between pca, tsne and umap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, KFold
import os

def plot(dim_data, labels, type):
    label_mapping = {0: 'Normal', 1: 'Pneumonia'}

    fig, ax = plt.subplots(figsize=(5, 5))

    # Create the scatterplot
    scatter = sns.scatterplot(
        x = dim_data[:, 0], 
        y = dim_data[:, 1], 
        hue = [label_mapping[label] for label in labels],  # Map numeric labels to strings
        alpha = 0.8, 
        palette = {'Normal': 'purple', 'Pneumonia': 'yellow'}, 
        ax = ax
    )

    # Customize the legend
    scatter.legend(title = 'Labels', loc = 'best')

    # Add title and labels
    plt.title(f"{type} of X-ray data", fontsize = 12)
    ax.set_xlabel(f"{type}_1", fontsize = 10)
    ax.set_ylabel(f"{type}_2", fontsize = 10)
    fig.savefig(f'./images/{type}.png')
    fig.show()

def imshow(dataloader):
    dataiter = iter(dataloader)
    idx = dataloader.batch_size - 1
    images, labels = next(dataiter)
    label = "Normal" if labels[idx] == 0 else "Pneumonia"

    img = images[idx] / 2 + 0.5
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f"Example of {label} X-ray")

    os.makedirs('./images', exist_ok=True)

    plt.savefig(f'./images/xray.png')
    plt.show()

def train_loss(losses):
    plt.plot(losses)
    plt.xlabel('Mini-Batch Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')

    os.makedirs('./images', exist_ok=True)

    plt.savefig(f'./images/cnn_loss.png')
    plt.show()

def heatmap_cv(pivot_table, mode):
    plt.figure(figsize = (8, 6))
    sns.heatmap(pivot_table, annot = True, 
                cmap = "YlGnBu", fmt=".4f",
                cbar_kws = {'label': 'Mean Test Score'})
    plt.title('Inner CV GridSearch SVC Raw Scores')
    plt.xlabel('C')
    plt.ylabel('Kernel')
    plt.savefig(f'./images/heatmap_grid_{mode}.png')
    plt.show()
    
def boxplot_cv(score_table):
    score_table.columns = ["Dummy Scores", "UMAP Model Scores", "Raw Model Scores"]
    score_table = score_table.melt(var_name = "Model Type", value_name = "Score")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data = score_table, x = "Model Type", y = "Score")
    sns.stripplot(data = score_table, x = "Model Type", y = "Score",
                color='red', alpha=0.5)
    plt.title('Outer CV SVC Scores')
    plt.ylabel('Score')
    plt.savefig('./images/boxplot_models_raw.png')
    plt.show()