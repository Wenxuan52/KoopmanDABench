#!/usr/bin/env python3
import numpy as np

data = np.load('data/cylinder/cylinder_train_data.npy')

print(data.min())
print(data.max())

data = np.load('data/cylinder/cylinder_val_data.npy')

print(data.min())
print(data.max())
