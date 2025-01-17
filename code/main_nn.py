#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 15 10:26:24 2025

@author: grigorii
"""
import argparse
from datasets import Images
from viz import imshow, train_loss
from models import CNNet, train_model, eval_model
from torch import manual_seed, optim
from torch.nn import CrossEntropyLoss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 1, type=int)
parser.add_argument("--n_epochs", type = int,
                     default = 1)
parser.add_argument('--batch-size', default = 100, type=int)

args = parser.parse_args()
manual_seed(args.seed)

# LOAD DATA
path = "./data/"  # to the parent folder of the dataset

image = Images(path)
_, trainloader = image.image_data("train")
testset, testloader = image.image_data("test")

# Plot sample image
imshow(trainloader)

# Define model, criterion and optimizer
net = CNNet()
optimizer = optim.Adam(net.parameters())
criterion = CrossEntropyLoss()

# Train model

losses = train_model(n_epochs = args.n_epochs, model = net, 
            dataloader = trainloader, optimizer = optimizer, 
            criterion = criterion)

train_loss(losses)

# Evaluation model
eval_model(model = net, testset = testset, testloader = testloader)
