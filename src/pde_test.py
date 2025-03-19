from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import harm_oscil_system
import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test


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

def advd_system(T, npoints_output):
    """Advection-diffusion"""
    f = None
    g = None
    Nt = 100
    return ADVDSystem(f, g, T, Nt, npoints_output)

def test_u_advd(nn, system, T, m, model, data, u, fname):
    """Test Advection-diffusion"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((xt, y_test, y_pred)))

def plot_energies(model, X_test,y_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    y_pred = []
    time = X_test[1]
    y_pred = model.predict(X_test)
    energies_pred = 0.5*(y_pred[:,1]**2 + (omega*y_pred[:, 0])**2)
    energies_true = 0.5*(y_test[:,1]**2 + (omega*y_test[:, 0])**2)
    plt.plot(time, energies_pred, label='predicted energies')
    plt.plot(time, energies_true, label='actual energies')
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.ylim(-5,5)
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
    

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    problem = "ode"
    
    # harm oscil parameters
    m = 2
    T = 5
    omega = 1.5
    L = 2
    
    # for testing
    num_train = 300
    num_test = 100
    lr = 0.001
    epochs = 50000
    #epochs = 5
    #epochs = 500

    # Network
    nn = "opnn"
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
    
    if nn == "opnn":
        print("Using DeepONet")
        net = dde.nn.DeepONet(
            [m, 40, 40],
            [dim_x, 40, 20],
            activation,
            initializer,
            #multi_output_strategy="independent_energy"# For harmonic oscillator
            #ulti_output_strategy="split_both_energy"
            num_outputs = 2,
            multi_output_strategy="split_branch_matrix"
            #multi_output_strategy="split_branch_energy"
        )
        #net = dde.nn.DeepONetCartesianProd(
        #    [m, 40, 40],
        #    [dim_x, 40, 40],
        #    "relu",
        #    "Glorot normal"
    #X_train, y_train = harm_oscil_system.gen_harm_dataset_fixed(1,0,omega, T, num_train) 
    #X_test, y_test = harm_oscil_system.gen_harm_dataset_fixed(1,0, omega,T, num_test)
    #X_train, y_train = harm_oscil_system.gen_harm_dataset(omega, L, T, num_train)
    #X_test, y_test = harm_oscil_system.gen_harm_dataset_fixed(1.5,1, omega,T, num_test)
    
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    problem = "advd"
    npoints_output = 100
    system = advd_system(T, npoints_output)

    X_train, y_train = system.gen_operator_data(space, m, num_train)
    X_test, y_test = system.gen_operator_data(space, m, num_test)
    
    print("Test dataset generated")
        
    print(f"Length of X_train{ len(X_train)}")
    print(f"Shape of first element {X_train[0].shape}")
    print(f"Shape of second element {X_train[1].shape}")
    #print(f"X_train \n {X_train}")

    print("Shape of y_train", len(y_train))
    print(f"Shape of first element {y_train[0].shape}")
    #print(f"Initial conditions : {X_train[0]}")
    #print(f"Time location = {X_train[1]}")
    #print(f"ground truth = {y_train[0]}")

    data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        #data = dde.data.TripleCartesianProd(
        #    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        #)
    print("Triple Dataset created")
    print(f"Using {net}")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    #model.compile("adam", lr=lr, metrics=["mean l2 relative error"])
    
    checker = dde.callbacks.ModelCheckpoint(
        f"deeponet/checkpoints/{problem}", save_better_only=True, period=1000
    )
    #losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    losshistory, train_state = model.train(iterations=epochs, callbacks=[checker])
    
    #print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    total_params = np.sum([np.prod(p.size()) for p in model.net.parameters()])
    print("# Parameters:", total_params)
    
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    print("Plotting loss history")
    dde.utils.plot_loss_history(losshistory)
    plt.savefig("deeponet/plots/ode_harm_oscil_mse_err.png")
    plt.show()
    
    print("Restoring the trained models")
    #model.restore(f"deeponet/checkpoints/{problem}-" + str(train_state.best_step) + ".ckpt", verbose=1)
    print("Testing")
    

    features = space.random(10)
    sensors = np.linspace(0, 1, num=m)[:, None]
    u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
    for i in range(u.shape[0]):
        test_u_advd(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")
    
    #plot_energies(model, X_test, omega)
    #plot_prediction(model, X_test, omega)
    plot_energies(model, X_test,y_test, omega)
    
    

if __name__ == "__main__":
    main()
