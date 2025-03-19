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

import schrodinger_system

#from schrodinger_system import gen_schro_dataset_fixed, gen_schro_dataset_sigma, gen_schro_dataset_x0, gen_schro_dataset_fixed_real
#from models import model, model_matrix_batch

torch.set_default_dtype(torch.float32)


def model(X, net): # take in num_datax1
    
    x_func = net.branch(torch.tensor(X[0])).to(torch.float32) # output num_datax 2x2, num_datax 4
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]))).to(torch.float32)
    
    # Split x_func into respective outputs
    shift = 0
    size = x_loc.shape[1]
    xs = []
    for i in range(net.num_outputs):
        x_func_ = x_func[:, shift : shift + size]
        x = net.merge_branch_trunk(x_func_, x_loc, i)
        xs.append(x)
        shift += size
    
    result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result.to(torch.float32)

def model_matrix_batch(X, net): # take in num_datax1
    x_func = net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))

    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)
    
    #b =  torch.tensor([net.b[0], net.b[1]], dtype=torch.float64).unsqueeze(1)
    #b_batch = b.expand(x_func.shape[0], -1, -1).to(torch.float64)
    
    result = torch.bmm(x_func, x_loc).to(torch.float64) #+ b_batch
    #print(f"result: {result.shape}")
    return result.squeeze()

def model_schro(X, net):
    x_func = torch.tensor(X[0], dtype=torch.float32).permute(0, 2, 1)
    x_func = net.branch(x_func).to(torch.float32) # output num_datax 2x2, num_datax 4
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]))).to(torch.float32).squeeze(0)
    #x_loc = x_loc.unsqueeze(-1)
    
    
    result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result.permute(0, 2, 1)
    
# Load dataset
num_train = 500
num_test = 100
lr = 0.001
#epochs = 10000

nx = 20
nt = 20
#epochs = 1
#epochs = 500
epochs = 50000

m = 200
# Network
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1


X_train, y_train = schrodinger_system.gen_schro_dataset_x0_cart_real(num=num_train, sensors=nx, sigma=0.3, t0=0,tf=1)
X_test, y_test = schrodinger_system.gen_schro_dataset_x0_cart_real(num=num_test, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_train, y_train = schrodinger_system.gen_schro_dataset_x0_cart_complex(num=num_train, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_test, y_test = schrodinger_system.gen_schro_dataset_x0_cart_complex(num=num_test, sensors=nx, sigma=0.3, t0=0,tf=1)
#data = dde.data.TripleCartesianProd(X_train, y_train, X_test, y_test)
print(f"x_train : {np.shape(X_train[0])}, {np.shape(X_train[1])} \n y_train: {np.shape(y_train)}")

#net = dde.maps.DeepONetCartesianProd(
#    [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
#) 
 
net = dde.maps.DeepONetCartesianProd(
    [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
)  

#net = dde.maps.DeepONetCartesianProd(
#    [2, 512, 512], [2, 512, 512, 256], "relu", "Glorot normal", num_outputs=2, multi_output_strategy="split_branch" 
#)  
    
#torch.set_default_dtype(torch.float64)
print(net)

# Generate dataset

print(f"X_train shape : {np.shape(X_train[0]), np.shape(X_train[1])}")
print(f"y_train shape {np.shape(y_train)}")
print("Dataset generated")
# Create dataloaders

# define optimizer 
# Custom backprop

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

print("Initialized ")
print("Start training")

# Training the model
loss_record = []
err_record = []

y_pred = model_schro(X_test,net)
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    #y_pred_energy = model_schro(X_train,net)
    
    loss = loss_fn(y_pred_energy.squeeze(), torch.tensor(y_train, dtype=torch.float32))
    loss_record.append(loss.item())
    loss.backward()
    #print(loss)
    optimizer.step()
    
    with torch.no_grad():
        y_pred_test = model(X_test, net)
        err = loss_fn(y_pred_test.squeeze(), torch.tensor(y_test, dtype = torch.float32))
        err_record.append(err.item())

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, loss {loss.item() :.6f}, acc = {err.item():.6f}")
    

print("Finished Training")

# Testing
y_pred = model_schro(X_test,net)
plt.plot(np.array(loss_record), label="loss")
plt.plot(np.array(err_record), label = "error")
plt.title("Loss")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

# Plot 1
axes[0].plot_surface(X_test[1][0][:,0], X_test[1][0][:, 1], np.abs(y_test[0].reshape(20, 20, 2))**2, cmap='plasma')
axes[0].set_title('Ground Truth')

# Plot 2
axes[1].plot_surface(X_test[1][0][:,0], X_test[1][0][:, 1], y_pred[0].detach().numpy().reshape(20, 20), cmap='plasma')
axes[1].set_title('Prediction')