from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


import itertools

import numpy as np

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn

from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem

from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

from schrodinger_system import gen_schro_dataset_fixed, gen_schro_dataset_sigma, gen_schro_dataset_x0, gen_schro_dataset_fixed_real
from models import model, model_matrix_batch

torch.set_default_dtype(torch.float32)

# Load dataset
num_train = 300
num_test = 100
lr = 0.001
epochs = 50000
#epochs = 1
#epochs = 500

m = 200
# Network
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1


net = dde.nn.DeepONet(
    #[m, 40, 40],
    #[dim_x, 40, 40],
    [m, 512, 512],
    [dim_x, 512, 512, 512],
    activation,
    initializer
    #num_outputs = m
    #multi_output_strategy="independent_energy"# For harmonic oscillator
    #ulti_output_strategy="split_both_energy"
    #multi_output_strategy="split_branch"
)       
    
#torch.set_default_dtype(torch.float64)
print(net)

# Generate dataset

X_train, y_train = gen_schro_dataset_fixed_real(num=500, sensors = m, x0=3.0, sigma=0.1, t0=0,tf=1)
X_test, y_test = gen_schro_dataset_fixed_real(num=500, sensors = m, x0=3.0, sigma=0.1, t0=0,tf=1)
#print(X_train,y_train)
print(f"X_train shape : {np.shape(X_train[0]), np.shape(X_train[1])}")
print(f"y_train shape {np.shape(y_train)}")
print("Dataset generated")
# Create dataloaders
output = model(X_train, net)
batch_size = 4



#y_pred = model_energy_v3_batch(X_train)

#y_pred = model(X_test)

#y_pred = model_matrix_batch(X_train)
# define optimizer 
# Custom backprop

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print("Initialized ")
print("Start training")

# Training the model


for epoch in range(epochs):
    optimizer.zero_grad()
    #y_pred = model(X_train)
    y_pred_energy = model(X_train,net)
    
    loss = loss_fn(y_pred_energy, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    #print(loss)
    optimizer.step()
    
    
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, loss {loss.item() :.8f} ")
        #print(y_pred_energy)
        #print(f"Without energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_orig[:,0]**2 + y_pred_orig[:,1]**2) - E)}")
        #print(f"With energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_energy[:,0]**2 + y_pred_energy[:,1]**2) - E)}")
        #print(f"Differnce = {y_pred_energy - y_pred_orig}")

    

print("Finished Training")

# Testing
y_pred = model(X_test,net)
#print(f"y_pred = {y_pred}")
#plt.plot(X_test[1], y_pred[:,0].detach().numpy(), label = 'prediction')
#plt.plot(X_test[1], y_test[:,0], label = 'ground truth')
#plt.legend()
#plt.savefig("deeponet/plots/harm_prediction.png")
#plt.show()
