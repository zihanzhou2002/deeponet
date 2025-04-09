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
from matplotlib import cm 

import torch
import torch.optim as optim
from torch import nn

from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem

from scipy.fft import fft, ifft
import scipy
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test
from plot_utils import plot_schrodinger_3d, plot_schrodinger_prob
import schrodinger_system

#from schrodinger_system import gen_schro_dataset_fixed, gen_schro_dataset_sigma, gen_schro_dataset_x0, gen_schro_dataset_fixed_real
#from models import model, model_matrix_batch



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
    x_func = torch.tensor(X[0], dtype=torch.complex64)
    x_func = net.branch(x_func).to(torch.complex64) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex64).squeeze(0)
    #x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex64)
    x_loc = x_loc.unsqueeze(2).to(torch.complex64)
    
    result = torch.bmm(x_func, x_loc).to(torch.complex64) 
    #result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    #result = torch.einsum('bi, bi->b', x_func, x_loc).to(torch.complex64)
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result

def model_schro_multi(X, net):
    x_func = torch.tensor(X[0], dtype=torch.complex64)
    x_func = net.branch(x_func).to(torch.complex64) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex64).squeeze(0)
    #x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex64)
    x_loc = torch.transpose(x_loc, 0, 1)
    
    result = torch.matmul(x_func, x_loc).to(torch.complex64) 
    #result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    #result = torch.einsum('bi, bi->b', x_func, x_loc).to(torch.complex64)
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result

def model_schro_prob_simple(X, net, x_max = 10, fourier =True):
    # Apply branch and trunk network
    x_func = torch.tensor(X[0], dtype=torch.complex64)
    x_func = net.branch(x_func).to(torch.complex64) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex64).squeeze(0)
    #x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex64)
    x_loc = x_loc.unsqueeze(2).to(torch.complex64)
    
    # Probability preservation on
    nx = X[0].shape[1]
    dx = x_max / nx
    
    #E = nx / dx if fourier else 1 /dx
    E = torch.tensor([torch.sum(torch.abs(torch.abs(Xi)**2)) for Xi in torch.tensor(X[0], dtype=torch.complex64)])
    
    
    Q, R = torch.linalg.qr(x_func)
    alpha_tilde = torch.matmul(R, x_loc).to(torch.complex64)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    #alpha_scaled = alpha_tilde* np.sqrt(E) / norm_alpha_tilde
    alpha_scaled = alpha_tilde* torch.sqrt(E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
    
    result_prob = torch.bmm(Q, alpha_scaled).to(torch.complex64)
    
    
    return result_prob.squeeze()

def model_schro_prob_multi(X, net, x_max = 10, fourier =True):
    # Apply branch and trunk network
    x_func = torch.tensor(X[0], dtype=torch.complex64)
    x_func = net.branch(x_func).to(torch.complex64) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex64)
    #x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex64)
    x_loc = torch.transpose(x_loc, 0, 1)
    
    # Probability preservation on
    nx = X[0].shape[1]
    dx = x_max / nx
    
    E = nx / dx if fourier else 1 /dx
    
    Q, R = torch.linalg.qr(x_func)
    alpha_tilde = torch.matmul(R, x_loc).to(torch.complex64)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True) + 1e-08
    alpha_scaled = alpha_tilde* np.sqrt(E) / norm_alpha_tilde
    
    result_prob = torch.bmm(Q, alpha_scaled).to(torch.complex64)
    
    
    return result_prob


def complex_relu(x):
    return torch.relu(x.real) + 1j * torch.relu(x.imag)

def complex_mse_loss(pred, target):
    """
    Compute the MSE loss for complex-valued tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex64 or torch.complex128)
        target (torch.Tensor): Target complex tensor (same shape and dtype as pred)

    Returns:
        torch.Tensor: Scalar tensor (the mean squared error)
    """
    pred_real = pred.real
    pred_imag = pred.imag
    target_real = target.real
    target_imag = target.imag
    
    mse_real = torch.mean((pred_real - target_real)**2)
    mse_imag = torch.mean((pred_imag - target_imag)**2)

    # Total loss is sum of both parts
    return mse_real + mse_imag

def complex_l2_relative_error(pred, target):
    """
    Compute the L2 relative error for complex-valued tensors.

    Parameters:
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex64 or torch.complex128)
        target (torch.Tensor): Target complex tensor (same shape and dtype as pred)

    Returns:
        torch.Tensor: Scalar tensor (the relative error)
    """
    # Compute squared magnitude difference
    error = pred - target
    error_norm = torch.sum(error.real**2 + error.imag**2)

    # Compute squared magnitude of target
    target_norm = torch.sum(target.real**2 + target.imag**2)

    # Avoid divide-by-zero (in case the target is all zeros, though this is rare)
    if target_norm == 0:
        return torch.tensor(float('inf'), device=target.device)

    # Relative L2 error
    rel_error = torch.sqrt(error_norm / target_norm)
    return rel_error

# ==========================================================================================================
# ==========================================================================================================


# Load dataset
num_train = 500
num_test = 100
lr = 0.001
#epochs = 10000

nx = 20

#num_train = 500
#num_test = 100
num_train = 500
num_test= 100
#epochs = 1
#epochs = 500
#epochs = 50000
epochs = 10000
#potential = "quadratic"
potential = "zero"
# Network
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1


#X_train, y_train = schrodinger_system.gen_schro_dataset_x0_cart_real(num=num_train, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_test, y_test = schrodinger_system.gen_schro_dataset_x0_cart_real(num=num_test, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_train, y_train = schrodinger_system.gen_schro_dataset_x0_cart_complex(num=num_train, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_test, y_test = schrodinger_system.gen_schro_dataset_x0_cart_complex(num=num_test, sensors=nx, sigma=0.3, t0=0,tf=1)

#X_train, y_train = schrodinger_system.gen_schro_dataset_x0(num=num_train, sensors=nx, sigma=0.3, t0=0,tf=1)
#X_test, y_test = schrodinger_system.gen_schro_dataset_x0(num=num_test, sensors=nx, sigma=0.3, t0=0,tf=1)

X_train, y_train = schrodinger_system.gen_schro_fourier_rand(num=num_train, sensors=nx,potential="quadratic")
X_test, y_test = schrodinger_system.gen_schro_fourier_rand(num=num_test, sensors = nx, potential="quadratic")

#data = dde.data.TripleCartesianProd(X_train, y_train, X_test, y_test)
print(f"x_train : {np.shape(X_train[0])}, {np.shape(X_train[1])} \n y_train: {np.shape(y_train)}")


# Defining net

#net = dde.maps.DeepONetCartesianProd(
#    [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
#) 
 
#net = dde.maps.DeepONetCartesianProd(
#    [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
#)  

#net = dde.maps.DeepONetCartesianProd(
#    [2, 512, 512], [2, 512, 512, 256], "relu", "Glorot normal", num_outputs=2, multi_output_strategy="split_branch" 
#)  

            
            
#net = dde.nn.DeepONetComplex(
#    [nx, 512, 512], [1, 512, 512, 512], complex_relu, "Glorot normal"
#)  

net = dde.nn.DeepONetComplex(
    [nx, 400, 400], [1, 20, 20, 20], complex_relu, "Glorot normal"
)  


    
#torch.set_default_dtype(torch.float64)
print(net, "\n")

# Generate dataset

print(f"X_train shape : {np.shape(X_train[0]), np.shape(X_train[1])}")
print(f"y_train shape {np.shape(y_train)}")
print("Dataset generated")

# Prescribe initializer, model, and loss function

loss_fn = complex_mse_loss
err_fn = complex_mse_loss
model = model_schro_prob_simple
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)# , weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

print("Model:", model.__name__, "\n")
print("Optimizer:", optimizer, "\n")

print("Initialized ")
print("Start training")

# Training the model
loss_record = []
err_record = []





for epoch in range(epochs):
    optimizer.zero_grad()
    #y_pred = model(X_train)
    y_pred_train = model(X_train,net)
    
    loss = loss_fn(y_pred_train.squeeze(), torch.tensor(y_train, dtype=torch.complex64))
    
    with torch.no_grad():
        y_pred_test = model(X_test, net)
        err = err_fn(y_pred_test.squeeze(), torch.tensor(y_test, dtype = torch.complex64))

   
    if (epoch + 1) % 1000 == 0:
        loss_record.append(loss.item())
        err_record.append(err.item())
        print(f"Epoch {epoch + 1}, loss {loss.item() :.6f}, err = {err.item():.6f}")
    
    loss.backward()
    #print(loss)
    optimizer.step()
    #scheduler.step(loss)
    
    #if (epoch + 1) % 1000 == 0:
    #    current_lr = optimizer.param_groups[0]['lr']
    #    print(f"Learning Rate: {current_lr:.6e}")
    

print("Finished Training")
plt.figure()
plt.plot(np.arange(len(loss_record))*1000, np.array(loss_record), label=f"loss-{loss_fn.__name__}")
plt.plot(np.arange(len(loss_record))*1000, np.array(err_record), label = f"error-{loss_fn.__name__}")
plt.title(f"Training Loss, {epochs} epochs")
plt.legend()
plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\schro_train_{model.__name__}_{potential}-potential_epoch{epochs}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
plt.show()

# Testing on one initial condition
X_test_fixed, y_test_fixed = schrodinger_system.gen_schro_fourier_fixed(num = num_test, sensors = nx)
#X_test_fixed, y_test_fixed = schrodinger_system.gen_schro_fourier_fixed_multi(nu = 1, nx = nx, nt = 50, potential=potential)

with torch.no_grad():
    y_pred_fixed = model(X_test_fixed,net).squeeze()

err = err_fn(y_pred_fixed, torch.tensor(y_test_fixed, dtype = torch.complex64))
print(f"Final {err_fn.__name__} error rate : {err.item():.6f}")


y_pred_fixed_sol = ifft(y_pred_fixed.detach().numpy())
y_test_fixed_sol = ifft(y_test_fixed)

#y_pred_fixed_sol = ifft(torch.transpose(y_pred_fixed.squeeze(), 0, 1).detach().numpy())
#y_test_fixed_sol = ifft(np.transpose(y_test_fixed[0]))

plot_schrodinger_3d(y_pred_fixed_sol, y_test_fixed_sol, model,net, optimizer, x_max = 10, T = 1)
plot_schrodinger_prob(y_pred_fixed_sol, y_test_fixed_sol, model, net, optimizer, x_max = 10, T = 1)

print("Finished")

