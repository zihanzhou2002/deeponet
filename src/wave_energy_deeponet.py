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

from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import dft
import scipy
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test
from plot_utils import plot_schrodinger_3d, plot_schrodinger_prob
import wave_system_single

#from schrodinger_system import gen_schro_dataset_fixed, gen_schro_dataset_sigma, gen_schro_dataset_x0, gen_schro_dataset_fixed_real
#from models import model, model_matrix_batch




def model_matrix_batch(X, net, c, x_max): # take in num_datax1
    x_func = net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))

    x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
    x_loc = x_loc.unsqueeze(2).to(torch.float64)
    
    #b =  torch.tensor([net.b[0], net.b[1]], dtype=torch.float64).unsqueeze(1)
    #b_batch = b.expand(x_func.shape[0], -1, -1).to(torch.float64)
    
    result = torch.bmm(x_func, x_loc).to(torch.float64) #+ b_batch
    #print(f"result: {result.shape}")
    return result.squeeze()

def model_wave(X, net, c, x_max):
    """Original DeepONet model for wave equation.

    Args:
        X (_type_): _description_
        net (_type_): _description_
        c (_type_): _description_
        x_max (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_func = torch.tensor(X[0], dtype=torch.complex128)
    x_func = net.branch(x_func).to(torch.complex128) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex128).squeeze(0)
    x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex128)
    #x_loc = x_loc.view(x_loc.shape[0], -1, 2).to(torch.complex128)
    
    result = torch.bmm(x_func, x_loc).to(torch.complex128) 
    #result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    #result = torch.einsum('bi, bi->b', x_func, x_loc).to(torch.complex128)
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result

def model_wave_small(X, net, c, x_max):

    x_func = torch.tensor(X[0], dtype=torch.complex128).unsqueeze(-1)
    x_func = net.branch(x_func).to(torch.complex128) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex128).squeeze(0)
    x_loc = x_loc.unsqueeze(-1)
    
    #x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex128)
    #x_loc = x_loc.view(x_loc.shape[0], -1, 2).to(torch.complex128)
    
    result = torch.bmm(x_func, x_loc).to(torch.complex128) 
    #result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    #result = torch.einsum('bi, bi->b', x_func, x_loc).to(torch.complex128)
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result

def model_wave_multi(X, net, c, x_max ):
    x_func = torch.tensor(X[0], dtype=torch.complex128)
    x_func = net.branch(x_func).to(torch.complex128) 
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex128).squeeze(0)
    #x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex128)
    x_loc = torch.transpose(x_loc, 0, 1)
    
    result = torch.matmul(x_func, x_loc).to(torch.complex128) 
    #result  = torch.einsum('bij,kj->bik', x_func, x_loc)
    #result = torch.einsum('bi, bi->b', x_func, x_loc).to(torch.complex128)
    #result = net.concatenate_outputs(xs)
    #print(f"result = {result}")
    return result.permute(0, 2, 1).squeeze()

def model_wave_energy_simple(X, net, c, x_max , fourier =True):
    """ DeepoNet model for wave equation with energy preservation.

    Args:
        X (_type_): _description_
        net (_type_): _description_
        c (_type_): _description_
        x_max (_type_): _description_
        fourier (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Apply branch and trunk network
    x_func = torch.tensor(X[0], dtype=torch.complex128)
    x_func = net.branch(x_func).to(torch.complex128) 
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex128).squeeze(0)
    x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex128)
    
    # Energy preservation on
    nx = int(X[0].shape[1] / 2)
    dx = x_max / (nx - 1)
    
    # DFT matrix W
    W_np = dft(nx)
    W = torch.tensor(W_np, dtype=torch.complex128)
    W_inv_np = W_np.conj().T / nx
    W_inv = torch.tensor(W_inv_np, dtype=torch.complex128).contiguous()
    
    W_large = torch.kron(torch.eye(2, dtype=torch.complex128), W)
    W_large_inv = torch.kron(torch.eye(2, dtype=torch.complex128), W_inv)
    
    # Differentiation matrix D
    k_np = (2 * np.pi ) * fftfreq(nx, dx)
    k = torch.tensor(k_np, dtype=torch.complex128)
    D = torch.diag(1j * k)
    D_large = torch.block_diag(D, torch.eye(nx, dtype=torch.complex128))
    D_large_sqrt = torch.block_diag(torch.diag(torch.abs(k)), torch.eye(nx, dtype=torch.complex128))
    
    # Define relvant matrices
    #omega = torch.block_diag(c**2*torch.eye(nx, dtype=torch.complex128), torch.eye(nx, dtype=torch.complex128))
    omega_sqrt = torch.block_diag(c*torch.eye(nx, dtype=torch.complex128), torch.eye(nx, dtype=torch.complex128))
    
    
    #theta = D_large.conj().T@W_large_inv.conj().T@omega@W_large_inv@D_large
    #theta = omega@D_large.conj().T@D_large / nx
    theta_sqrt = omega_sqrt@D_large_sqrt /torch.sqrt(torch.tensor(nx,dtype=torch.complex128))
    #c_tensor = torch.tensor(c, dtype=torch.complex128)
    
    #init = torch.tensor(X[0], dtype=torch.complex128)
    init_ux_sol = torch.tensor(X[0], dtype=torch.complex128)@D_large.conj().T@W_large_inv.conj().T
    #E_alt = init@theta@init.conj().T
  
    # Caulate energy
    E = (
    torch.sum(c**2 * (torch.abs(init_ux_sol[:, :nx])**2), dim=1) +
    torch.sum(torch.abs(init_ux_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    

    # Decompose Q
    #theta_sqrt_batch = theta_sqrt.expand(x_func.shape[0], -1, -1)
    #theta_small = theta[1:, 1:]
    theta_small_sqrt = theta_sqrt[1:, 1:]
    theta_small_sqrt_inv = torch.diag(1 / torch.diagonal(theta_small_sqrt))
    
    b1s = x_func[:, 0, :]
    B_small = x_func[:, 1:, :]
    B_small_tilde = torch.einsum("pn, mnl ->mpl", theta_small_sqrt, B_small)
    
    
    Q_small_tilde, R = torch.linalg.qr(B_small_tilde)
    R_inv = torch.linalg.inv(R)
    
    
    
    Q_tilde = torch.zeros_like(x_func)
    
    # The first rows of each matrix
    Q_tilde[:, 0, :] = torch.matmul(b1s.unsqueeze(1), R_inv).squeeze(1)
    Q_tilde[:, 1:, :] = torch.einsum("pn, mnl -> mpl", theta_small_sqrt_inv, Q_small_tilde)
    
    
    alpha_tilde = torch.matmul(R, x_loc).to(torch.complex128)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    # Create a mask for zero values
    zero_mask = (norm_alpha_tilde == 0)

    # Replace zeros with ones in the denominator to avoid division by zero
    safe_norm = torch.where(zero_mask, torch.ones_like(norm_alpha_tilde), norm_alpha_tilde)
    #alpha_scaled = alpha_tilde* np.sqrt(E) / norm_alpha_tilde
    p = alpha_tilde.shape[1]
    alpha_scaled = alpha_tilde* torch.sqrt(E).unsqueeze(1).unsqueeze(2) / safe_norm
    
    result_energy = torch.bmm(Q_tilde, alpha_scaled).to(torch.complex128)
    
    
    return result_energy.squeeze()

def model_wave_energy_multi(X, net, c, x_max = 10, fourier =True):
    """ DeepoNet model for wave equation with energy preservation.

    Args:
        X (_type_): _description_
        net (_type_): _description_
        c (_type_): _description_
        x_max (_type_): _description_
        fourier (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Apply branch and trunk network
    x_func = torch.tensor(X[0], dtype=torch.complex128)
    x_func = net.branch(x_func).to(torch.complex128) 
    
    x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1]).unsqueeze(-1))).to(torch.complex128).squeeze(0)
    x_loc = x_loc.unsqueeze(-1)
    
    x_func = x_func.view(x_func.shape[0], np.shape(X[0])[1], -1).to(torch.complex128)
    
    # Energy preservation on
    nx = int(X[0].shape[1] / 2)
    dx = x_max / (nx - 1)
    
    # DFT matrix W
    W_np = dft(nx)
    W = torch.tensor(W_np, dtype=torch.complex128)
    W_inv_np = W_np.conj().T / nx
    W_inv = torch.tensor(W_inv_np, dtype=torch.complex128).contiguous()
    
    #W_large = torch.kron(torch.eye(2, dtype=torch.complex128), W)
    W_large_inv = torch.kron(torch.eye(2, dtype=torch.complex128), W_inv)
    
    # Differentiation matrix D
    k_np = (2 * np.pi ) * fftfreq(nx, dx)
    k = torch.tensor(k_np, dtype=torch.complex128)
    D = torch.diag(1j * k)
    D_large = torch.block_diag(D, torch.eye(nx, dtype=torch.complex128))
    D_large_sqrt = torch.block_diag(torch.diag(torch.abs(k)), torch.eye(nx, dtype=torch.complex128))
    
    # Define relvant matrices
    #omega = torch.block_diag(c**2*torch.eye(nx, dtype=torch.complex128), torch.eye(nx, dtype=torch.complex128))
    omega_sqrt = torch.block_diag(c*torch.eye(nx, dtype=torch.complex128), torch.eye(nx, dtype=torch.complex128))
    
    
    #theta = D_large.conj().T@W_large_inv.conj().T@omega@W_large_inv@D_large
    #theta = omega@D_large.conj().T@D_large / nx
    theta_sqrt = omega_sqrt@D_large_sqrt /torch.sqrt(torch.tensor(nx,dtype=torch.complex128))
    #c_tensor = torch.tensor(c, dtype=torch.complex128)
    
    #init = torch.tensor(X[0], dtype=torch.complex128)
    init_ux_sol = torch.tensor(X[0], dtype=torch.complex128)@D_large.conj().T@W_large_inv.conj().T
    #E_alt = init@theta@init.conj().T
  
    # Caulate energy
    E = (
    torch.sum(c**2 * (torch.abs(init_ux_sol[:, :nx])**2), dim=1) +
    torch.sum(torch.abs(init_ux_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    

    # Decompose Q
    #theta_sqrt_batch = theta_sqrt.expand(x_func.shape[0], -1, -1)
    #theta_small = theta[1:, 1:]
    theta_small_sqrt = theta_sqrt[1:, 1:]
    theta_small_sqrt_inv = torch.diag(1 / torch.diagonal(theta_small_sqrt))
    
    b1s = x_func[:, 0, :]
    B_small = x_func[:, 1:, :]
    B_small_tilde = torch.einsum("pn, mnl ->mpl", theta_small_sqrt, B_small)
    
    
    Q_small_tilde, R = torch.linalg.qr(B_small_tilde)
    R_inv = torch.linalg.inv(R)
    
    
    
    Q_tilde = torch.zeros_like(x_func)
    
    # The first rows of each matrix
    Q_tilde[:, 0, :] = torch.matmul(b1s.unsqueeze(1), R_inv).squeeze(1)
    Q_tilde[:, 1:, :] = torch.einsum("pn, mnl -> mpl", theta_small_sqrt_inv, Q_small_tilde)
    
    
    alpha_tilde = torch.matmul(R, x_loc.permute(1, 0, 2).squeeze()).to(torch.complex128)
    
    norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
    # Create a mask for zero values
    zero_mask = (norm_alpha_tilde == 0)

    # Replace zeros with ones in the denominator to avoid division by zero
    safe_norm = torch.where(zero_mask, torch.ones_like(norm_alpha_tilde), norm_alpha_tilde)
    #alpha_scaled = alpha_tilde* np.sqrt(E) / norm_alpha_tilde
    p = alpha_tilde.shape[1]
    alpha_scaled = alpha_tilde* torch.sqrt(E).unsqueeze(1).unsqueeze(2) / safe_norm
    
    result_energy = torch.bmm(Q_tilde, alpha_scaled).to(torch.complex128)
    
    
    return result_energy.permute(0, 2, 1).squeeze()


def complex_relu(x):
    return torch.relu(x.real) + 1j * torch.relu(x.imag)

def complex_tanh(x):
    return torch.tanh(x.real) + 1j* torch.tanh(x.imag)

def complex_mse_loss(pred, target):
    """
    Compute the MSE loss for complex-valued tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex128 or torch.complex128)
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
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex128 or torch.complex128)
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

def wave_energy_loss(y_pred, y_true):
    """
    Compute the MSE loss for complex-valued tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex128 or torch.complex128)
        target (torch.Tensor): Target complex tensor (same shape and dtype as pred)

    Returns:
        torch.Tensor: Scalar tensor (the mean squared error)
    """
    global x_max, c
    nx = int(y_pred.shape[1] / 2)
    dx = x_max / (nx - 1)
    
    # DFT matrix W
    W_np = dft(nx)
    W = torch.tensor(W_np, dtype=torch.complex128)
    W_inv_np = W_np.conj().T / nx
    W_inv = torch.tensor(W_inv_np, dtype=torch.complex128).contiguous()
    
    #W_large = torch.kron(torch.eye(2, dtype=torch.complex128), W)
    W_large_inv = torch.kron(torch.eye(2, dtype=torch.complex128), W_inv)
    
    # Differentiation matrix D
    k_np = (2 * np.pi ) * fftfreq(nx, dx)
    k = torch.tensor(k_np, dtype=torch.complex128)
    D = torch.diag(1j * k)
    D_large = torch.block_diag(D, torch.eye(nx, dtype=torch.complex128))
    
    
    y_pred_x_sol = y_pred@D_large.conj().T@W_large_inv.conj().T
    y_true_x_sol = y_true@D_large.conj().T@W_large_inv.conj().T
    
    energy_pred = dx*(torch.sum(c**2*torch.abs(y_pred_x_sol[:, 0: nx])**2,dim=1) + torch.sum(torch.abs(y_pred_x_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    energy_true = dx*(torch.sum(c**2*torch.abs(y_true_x_sol[:, 0: nx])**2,dim=1) + torch.sum(torch.abs(y_true_x_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    
    error = torch.mean(torch.abs(energy_pred - energy_true)**2)

    return error

def wave_energy_loss(y_pred, y_true):
    """
    Compute the MSE loss for complex-valued tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted complex tensor (dtype=torch.complex128 or torch.complex128)
        target (torch.Tensor): Target complex tensor (same shape and dtype as pred)

    Returns:
        torch.Tensor: Scalar tensor (the mean squared error)
    """
    global x_max, c
    nx = int(y_pred.shape[1] / 2)
    dx = x_max / (nx - 1)
    
    # DFT matrix W
    W_np = dft(nx)
    W = torch.tensor(W_np, dtype=torch.complex128)
    W_inv_np = W_np.conj().T / nx
    W_inv = torch.tensor(W_inv_np, dtype=torch.complex128).contiguous()
    
    #W_large = torch.kron(torch.eye(2, dtype=torch.complex128), W)
    W_large_inv = torch.kron(torch.eye(2, dtype=torch.complex128), W_inv)
    
    # Differentiation matrix D
    k_np = (2 * np.pi ) * fftfreq(nx, dx)
    k = torch.tensor(k_np, dtype=torch.complex128)
    D = torch.diag(1j * k)
    D_large = torch.block_diag(D, torch.eye(nx, dtype=torch.complex128))
    
    
    y_pred_x_sol = y_pred@D_large.conj().T@W_large_inv.conj().T
    y_true_x_sol = y_true@D_large.conj().T@W_large_inv.conj().T
    
    energy_pred = dx*(torch.sum(c**2*torch.abs(y_pred_x_sol[:, 0: nx])**2,dim=1) + torch.sum(torch.abs(y_pred_x_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    energy_true = dx*(torch.sum(c**2*torch.abs(y_true_x_sol[:, 0: nx])**2,dim=1) + torch.sum(torch.abs(y_true_x_sol[:, nx:])**2, dim=1)).to(torch.complex128)
    
    error = torch.mean(torch.abs(energy_pred - energy_true)**2)

    return error

def wave_mse_energy_loss(y_pred, y_true):
    return complex_mse_loss(y_pred, y_true) + wave_energy_loss(y_pred, y_true)
# ==========================================================================================================
# ==========================================================================================================


# Load dataset
num_train = 500
num_test = 100
lr = 0.001
#epochs = 10000


#num_train = 500
#num_test = 100
num_train = 500
num_test= 100
#epochs = 1
#epochs = 500
#epochs = 50000
epochs = 10000
nt = 50

# Network
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1

nx = 20
# Domain 
L = 10
T = 1
c = 1

#X_train, y_train = wave_system_single.gen_wave_fourier_rand_fixed_speed(num = num_train, Nx = nx, x_max= L, tf=T)
#X_test, y_test = wave_system_single.gen_wave_fourier_rand_fixed_speed(num=num_test, Nx = nx, x_max = L, tf = T)

X_train, y_train = wave_system_single.gen_wave_fourier_rand_GRF_fixed_speed_multi(Nu = num_train, Nx = nx,Nt = nt, x_max= L, tf=T)
X_test, y_test = wave_system_single.gen_wave_fourier_rand_GRF_fixed_speed_multi(Nu=num_test, Nx = nx, Nt=nt, x_max = L, tf = T)
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

#net = dde.nn.DeepONetComplex(
#    [2*nx, 800, 800], [1, 10, 10, 10], complex_relu, "Glorot normal"
#)  
p = nx


net = dde.nn.DeepONetComplex(
    [2*nx, 2*nx*p, 2*nx*p], [1, p, p, p], complex_relu, "Glorot normal"
)  

#net = dde.nn.DeepONetComplex(
#    [1, 128, 128], [1, 128, 128, 128],complex_relu,"Glorot normal"
#)  
#torch.set_default_dtype(torch.float64)
print(net, "\n")

# Generate dataset

print(f"X_train shape : {np.shape(X_train[0]), np.shape(X_train[1])}")
print(f"y_train shape {np.shape(y_train)}")
print("Dataset generated")

# Prescribe initializer, model, and loss function

#loss_fn = complex_mse_loss
loss_fn = complex_l2_relative_error
err_fn = complex_l2_relative_error
#model = model_wave_energy_simple
model = model_wave_energy_multi
#model = model_wave_multi
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)# , weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
#optimizer = torch.optim.LBFGS(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

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
    y_pred_train = model(X_train,net,c=c, x_max = L)
    
    loss = loss_fn(y_pred_train.squeeze(), torch.tensor(y_train, dtype=torch.complex128))
    
    with torch.no_grad():
        y_pred_test = model(X_test, net, c=c, x_max = L)
        err = err_fn(y_pred_test.squeeze(), torch.tensor(y_test, dtype = torch.complex128))

   
    if (epoch + 1) % 1000 == 0:

        print(f"Epoch {epoch + 1}, loss {loss.item() :.6f}, err = {err.item():.6f}")
        
    loss_record.append(loss.item())
    err_record.append(err.item())
    loss.backward()
    #print(loss)
    optimizer.step()
    scheduler.step(loss)
    
    if (epoch + 1) % 1000 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")
    

print("Finished Training")
plt.plot(np.arange(len(loss_record)), np.array(loss_record), label=f"loss-{loss_fn.__name__}")
plt.plot(np.arange(len(loss_record)), np.array(err_record), label = f"error-{loss_fn.__name__}")
plt.title(f"Training Loss, {epochs} epochs")
plt.grid(True)
plt.legend()
plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_train_{model.__name__}_epoch{epochs}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
plt.show()


# After your training loop where you print "Finished Training"
model_save_path = "C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\trained_nets"
os.makedirs(model_save_path, exist_ok=True)

# Create a descriptive filename
model_filename = f"{model.__name__}_epoch{epochs}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_loss-{loss_fn.__name__}.pt"
full_path = os.path.join(model_save_path, model_filename)

# Save the model
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'epoch': epochs,
    'loss_function': loss_fn.__name__, 
    'lr': optimizer.param_groups[0]['lr'],
    'train_losses': loss_record,
    'train_errors': err_record,
}, full_path)

print(f"Model saved to {full_path}")
# Testing on one initial condition
#X_test_fixed, y_test_fixed = wave_system_single.gen_wave_fourier_init_fixed_speed(num = 400, Nx = nx, x_max = L, tf = T)
#X_test_fixed, y_test_fixed = schrodinger_system.gen_schro_fourier_fixed_multi(nu = 1, nx = nx, nt = 50)

with torch.no_grad():
    y_pred_fixed = model(X_test,net,c,L).squeeze()

err = err_fn(y_pred_fixed, torch.tensor(y_test, dtype = torch.complex128))
print(f"Final {err_fn.__name__} error rate : {err.item():.6f}")


#y_pred_fixed_sol = ifft(y_pred_fixed.detach().numpy())
#y_test_fixed_sol = ifft(y_test_fixed)

#y_pred_fixed_sol = np.hstack((ifft(y_pred_fixed[:, 0: nx].detach().numpy(), axis = 1) , ifft(y_pred_fixed[:, nx: ].detach().numpy(), axis = 1)))
#y_test_fixed_sol = np.hstack((ifft(y_test_fixed[:, 0: nx], axis = 1), ifft(y_pred_fixed[:, nx:], axis=1)))

wave_system_single.plot_wave_2d(y_pred_fixed[0].detach().numpy(), y_test[0], model,net, optimizer, x_max = L, T = T)
wave_system_single.plot_wave_energy(y_pred_fixed[0].detach().numpy(), y_test[0], model, net, optimizer,c = c, x_max = L, T = T)

print("Finished")

