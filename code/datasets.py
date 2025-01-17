#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 15 10:26:24 2025

@author: grigorii
"""

"""
structure of input:
- test
    - NORMAL: images
    - PNEUMONIA: images
- train
    - NORMAL: images
    - PNEUMONIA: images
- val
    - NORMAL
    - PNEUMONIA

"""

"""
path - path to the data (with folders train, test and val)
state_path - path to the folders corresponing to the labels
img_path - path of a certain picture

"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


class Images():
    def __init__(self, path):
        self.path = path
        self.states = ["NORMAL", "PNEUMONIA"]
        
    def linear_data(self, mode, states):
        images = []
        labels = []
        path = self.path + mode

        for i, state in enumerate(states):
            state_path = path + state
            for filename in os.listdir(state_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(state_path, filename)
                    img = Image.open(img_path).convert('L')
                    img = img.resize((32, 32))
                    img_vector = np.array(img).flatten()
                    images.append(img_vector)
                    labels.append(i)
        
        dataset = list(zip(images, labels)) # combine into a single dataset

        random.shuffle(dataset)

        data, labels = zip(*dataset)
        
        scaler = StandardScaler() # scaling before visualization and applying models
        data_scaled = scaler.fit_transform(data) 

        return np.array(data_scaled), np.array(labels)

    def image_data(self, mode):
        path = self.path + mode

        # Define transformations for the images
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.Grayscale(num_output_channels = 1), # resize to one channel
            transforms.ToTensor(),          # Convert to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
        
        # Load the dataset
        dataset = datasets.ImageFolder(root = path, transform = transform)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)

        return dataset, dataloader
