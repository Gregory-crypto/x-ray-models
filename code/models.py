#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 15 10:26:24 2025

@author: grigorii
"""

from torch import flatten, no_grad, max
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.svm import SVC

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(120, 81)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(81, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
def train_model(n_epochs, model, dataloader, optimizer, criterion):
    losses = []
    epochs = n_epochs

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs) # forward
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')

    return losses


def eval_model(model, testset, testloader):
    correct = 0
    total = testset.__len__()

    with no_grad(): # torch.no_grad
        for data in testloader:
            images, labels = data

            outputs = model(images)

            _, predicted = max(outputs, 1) # torch.max
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 624 test images: {100 * correct // total} %')

def pca(data_scaled):
    pca = PCA(n_components = 2)
    return pca.fit_transform(data_scaled) # dataset with shape (n_samples, 2)

def tsne(data_scaled):
    tsne = TSNE()
    return tsne.fit_transform(data_scaled)

def umap(data_scaled):
    umap = UMAP(n_components=2)
    return umap.fit_transform(data_scaled)

def dummy(data_scaled, labels):
    outer_cv = KFold(n_splits = 5, shuffle = True)
    dummy_clf = DummyClassifier(strategy = "most_frequent")
    return cross_val_score(estimator = dummy_clf, X = data_scaled, y = labels,
                        cv = outer_cv, n_jobs = -1, scoring = 'roc_auc') # dummy scores

def cross_v(data, labels):
    # Declaration of grids
    inner_cv = KFold(n_splits = 5, shuffle = True)
    outer_cv = KFold(n_splits = 5, shuffle = True)

    # Setting models and parameters
    params = {"C": [1e-1, 1, 10], "kernel": ['linear', "poly"]}
    model = SVC()

    # Grid search
    search = GridSearchCV(estimator = model, param_grid = params, 
                        n_jobs = -1, cv = inner_cv, scoring = 'roc_auc') 
    scores = cross_val_score(search, X = data, y = labels,
                            cv = outer_cv, n_jobs = -1, scoring = 'roc_auc')
    
    # Preparation for visualization
    search.fit(data, labels)
    results = pd.DataFrame(search.cv_results_)

    print(f"Best C for inner CV: {search.best_params_['C']}")
    print(f"Best kernel for inner CV: {search.best_params_['kernel']}")
    print(f"Best score for outer CV: {scores.mean()}")

    pivot_table = results.pivot_table(
        values = 'mean_test_score', 
        index = 'param_kernel', 
        columns = 'param_C'
    )
    return pivot_table, scores
    