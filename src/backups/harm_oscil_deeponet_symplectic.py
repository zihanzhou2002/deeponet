from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn


import harm_oscil_system
import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test
from torch.autograd.functional import jacobian
from models import model_matrix_batch, model_energy_batch, model_energy_v2_batch, model_energy_v3_batch

def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    x = np.linspace(0, T, num=num)[:, None]
    #X_test = [np.tile(sensor_values.T, (num, 1)), x]
    X_test = (np.tile(sensor_values.T, (num, 1)), x)
    
    print("Testing the ODE")
    
    y_test = system.eval_s_func(u, x)
    
    #print(f"X_test \n{X_test}")
    #print(f"y_test \n{y_test}")
    if nn != "opnn":
        X_test = merge_values(X_test)
    
    #y_pred = model.predict(data.transform_inputs(X_test))
    y_pred = model.predict(X_test)
    
    
    #print("Shape of X_test", len(X_test))
    #print("Shape of y_test", len(y_test))
    
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    
    #print("Shape of y_pred", len(y_pred))
    
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))
    print("MSE error:", dde.metrics.mean_squared_error(y_test, y_pred))

def plot_energies(model, X_test,y_test, omega):
    
    time = X_test[1]
    y_pred = model(X_test).detach().numpy()
    energies_pred = 0.5*(y_pred[:,1]**2 + (omega*y_pred[:, 0])**2)
    energies_true = 0.5*(y_test[:,1]**2 + (omega*y_test[:, 0])**2)
    
    plt.plot(time, energies_pred, label='predicted energies')
    plt.plot(time, energies_true, label='actual energies')
    plt.ylim(-5, 5)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Total Energies over Time")
    plt.savefig("deeponet/plots/total_energies.png")
    plt.show()

def plot_prediction(model, X_test,y_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    
    X = X_test[0]
    time = X_test[1]
    y_pred = np.zeros(len(time))

    harm_pred = y_pred[:,0]
    plt.plot(harm_pred, marker='o')
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Predicted_harmonic_oscillator")
    plt.savefig(f"deeponet/plots/pred_{omega}.png")
    plt.show()


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
problem = "ode"

# harm oscil parameters
m = 2
T = 5
omega = 1
L = 2

# for testing
num_train = 300
num_test = 1
lr = 0.001
epochs = 50000
model = model_energy_v2_batch
#epochs = 1
#epochs = 500

# Network
activation = "relu"
initializer = "Glorot normal"  # "He normal" or "Glorot normal"
dim_x = 1 if problem in ["ode", "lt"] else 2


net = dde.nn.DeepONet(
    [m, 40, 40],
    [dim_x, 40, 20],
    activation,
    initializer,
    num_outputs = 2,
    #multi_output_strategy="independent_energy"# For harmonic oscillator
    #ulti_output_strategy="split_both_energy"
    multi_output_strategy="split_branch"
)       
    


# Generate dataset
#X_train, y_train = harm_oscil_system.gen_harm_dataset(1,1, omega, T, num_train) 
X_train, y_train = harm_oscil_system.gen_harm_dataset(omega, L, T, num_train) 
#X_train, y_train = harm_oscil_system.gen_harm_dataset_fixed(1.5,1, omega, T, num_train) 
X_test, y_test = harm_oscil_system.gen_harm_dataset_fixed(1,0, omega,T, num_test)
print(X_train,y_train)
print("Dataset generated")
# Create dataloaders
#output = model(X_train)
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

p0 = torch.tensor(X_train[0][:, 0])
q0 = torch.tensor(X_train[0][:, 1])
    
E = 0.5* (omega*p0**2 +q0**2)
# Training the model

for epoch in range(epochs):
    optimizer.zero_grad()
    #y_pred = model(X_train)
    y_pred_energy = model_energy_v2_batch(X_train,net,omega)
    
    loss = loss_fn(y_pred_energy, torch.tensor(y_train, dtype=torch.float64))
    loss.backward()
    optimizer.step()
    
    
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, loss {loss.item() :.8f} ")
        #print(y_pred_energy)
        #print(f"Without energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_orig[:,0]**2 + y_pred_orig[:,1]**2) - E)}")
        #print(f"With energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_energy[:,0]**2 + y_pred_energy[:,1]**2) - E)}")
        #print(f"Differnce = {y_pred_energy - y_pred_orig}")

    

print("Finished Training")

# Testing
J = torch.tensor([[0, -1], [1, 0]],dtype = torch.float32)

# Store the original x_loc
x_loc_original = net.activation_trunk(net.trunk(torch.tensor(X_test[1])))

# Define a closure that captures x_loc_original in its scope
def model_temp(X_func):
    
    return model((X_func, x_loc_original), net, omega)

N_dash = torch.autograd.functional.jacobian(model_temp, torch.tensor(X_test[0]), create_graph=True)
first = torch.matmul(N_dash.T, J)
res = torch.matmul(first.squeeze(), N_dash.squeeze())

print(res)

#y_pred = model(X_test)

#print(f"y_pred = {y_pred}")
#plt.plot(X_test[1], y_pred[:,0].detach().numpy(), label = 'prediction')
#plt.plot(X_test[1], y_test[:,0], label = 'ground truth')
#plt.legend()
#plt.savefig("deeponet/plots/harm_prediction.png")
#plt.show()

#safe_test(model, data, X_test, y_test)

#plot_energies(model, X_test, omega)
#plot_prediction(model, X_test, omega)
#plot_energies(model_energy_batch, X_test,y_test, omega)

