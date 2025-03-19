from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test


def test_u_lt(nn, system, T, m, model, data, u, fname):
    """Test Legendre transform"""
    sensors = np.linspace(-1, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    ns = np.arange(system.npoints_output)[:, None]
    X_test = [np.tile(sensor_value, (system.npoints_output, 1)), ns]
    y_test = s
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((ns, y_test, y_pred)))


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


def test_u_dr(nn, system, T, m, model, data, u, fname):
    """Test Diffusion-reaction"""
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
    np.savetxt(fname, np.hstack((xt, y_test, y_pred)))


def test_u_cvc(nn, system, T, m, model, data, u, fname):
    """Test Advection"""
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


def lt_system(npoints_output):
    """Legendre transform"""
    return LTSystem(npoints_output)


def ode_system(T, system_params=None):
    """ODE"""
    omega, p0, q0 = system_params
    def g(s, u, x):
        
        
        # Antiderivative
        return u
        
        # Nonlinear ODE
        # return -s**2 + u
        
        # Gravity pendulum
        #k = 1
        #return [s[1], - k * np.sin(s[0]) + u]
        
        #Harmonic Oscillator
        #omega, p0, q0 = system_params
        
        #return [s[1], - omega**2*s[0]]

    s0 = [0]
    #s0 = [p0, q0]  # Gravity pendulum
    return ODESystem(g, s0, T)


def dr_system(T, npoints_output):
    """Diffusion-reaction"""
    D = 0.01
    k = 0.01
    Nt = 100
    return DRSystem(D, k, T, Nt, npoints_output)


def cvc_system(T, npoints_output):
    """Advection"""
    f = None
    g = None
    Nt = 100
    return CVCSystem(f, g, T, Nt, npoints_output)


def advd_system(T, npoints_output):
    """Advection-diffusion"""
    f = None
    g = None
    Nt = 100
    return ADVDSystem(f, g, T, Nt, npoints_output)

def plot_energies(model, X_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    y_pred = []
    X = X_test
    while is_nonempty(X):
        X_add, X = trim_to_65535(X)
        #y_pred.append(model.predict(data.transform_inputs(X_add)))
        y_pred.append(model.predict(X_add))
    y_pred = np.vstack(y_pred)
    energies = 0.5*(y_pred[:,1]**2 + (omega*y_pred[:, 0])**2)
    plt.plot(energies, marker='o', label='Total Energy Over Time')
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Total Energies over Time")
    plt.savefig("deeponet/plots/total_energies.png")
    plt.show()

def plot_prediction(model, X_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    y_pred = []
    X = X_test
    while is_nonempty(X):
        X_add, X = trim_to_65535(X)
        #y_pred.append(model.predict(data.transform_inputs(X_add)))
        y_pred.append(model.predict(X_add))
    y_pred = np.vstack(y_pred)
    harm_pred = y_pred[:,0]
    plt.plot(harm_pred, marker='o')
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Predicted_harmonic_oscillator")
    plt.savefig(f"deeponet/plots/pred_{omega}.png")
    plt.show()

def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")

    X_train, y_train = system.gen_operator_data(space, m, num_train)
    #plt.plot(X_train[0][0],label="X_train", marker = '.')
    #plt.plot(y_train[0], label = "y_train[0]", marker = '.')
    #plt.plot(y_train[1], label = "y_train[1]", marker = '.')
    #plt.legend()
    #plt.show()
    '''
    curr_path = os.getcwd()
    if 'deeponet' not in curr_path:
        curr_path = os.path.join(curr_path, 'deeponet')
    if 'data' not in curr_path:
        curr_path = os.path.join(curr_path, 'data')
    
    filename = f'{problem}_train{m}.npz'
    save_dir = os.path.join(curr_path, filename)
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)
    
    np.savez(save_dir, X=X_train, y=y_train, allow_pickle=True)
    '''
    #print(f"\nTrain dataset generated")
    
    X_test, y_test = system.gen_operator_data(space, m, num_test)
    print("Test dataset generated")
    if nn != "opnn":
        X_train = merge_values(X_train)
        X_test = merge_values(X_test)
        
    print(f"Length of X_train{ len(X_train)}")
    print(f"Shape of first element {X_train[0].shape}")
    print(f"Shape of second element {X_train[1].shape}")
    #print(f"X_train \n {X_train}")

    print("Shape of y_train", len(y_train))
    print(f"Shape of first element {y_train[0].shape}")
    #print(f"y_train \n{ y_train}")
    #print(f"X_test \n{X_test}")
    #print(f"y_test \n{y_test}")
   
    # np.savez_compressed("train.npz", X_train0=X_train[0], X_train1=X_train[1], y_train=y_train)
    # np.savez_compressed("test.npz", X_test0=X_test[0], X_test1=X_test[1], y_test=y_test)
    # return

    # d = np.load("train.npz")
    # X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    # d = np.load("test.npz")
    # X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    if nn == "opnn":
        data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
        #data = dde.data.TripleCartesianProd(
        #    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        #)
        print("Triple Dataset created")
    else:
        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
    
    print(f"Using {net}")
    model = dde.Model(data, net)
    #model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    
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
    plt.savefig("deeponet/plots/advd_l2rel_err.png")
    plt.show()
    
    print("Restoring the trained models")
    model.restore(f"deeponet/checkpoints/{problem}-" + str(train_state.best_step) + ".ckpt", verbose=1)
    print("Testing")
    
    safe_test(model, data, X_test, y_test)

    tests = [
        
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    for u, fname in tests:
        if problem == "lt":
            test_u_lt(nn, system, T, m, model, data, u, fname)
        elif problem == "ode":
            test_u_ode(nn, system, T, m, model, data, u, fname)
        elif problem == "dr":
            test_u_dr(nn, system, T, m, model, data, u, fname)
        elif problem == "cvc":
            test_u_cvc(nn, system, T, m, model, data, u, fname)
        elif problem == "advd":
            test_u_advd(nn, system, T, m, model, data, u, fname)

    if problem == "lt":
        features = space.random(10)
        sensors = np.linspace(0, 2, num=m)[:, None]
        u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_lt(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "cvc":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        # Case I Input: V(sin^2(pi*x))
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        # Case II Input: x*V(x)
        # u = sensors.T * space.eval_u(features, sensors)
        # Case III/IV Input: V(x)
        # u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_cvc(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "advd":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        for i in range(u.shape[0]):
            test_u_advd(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")


def main():
    # Problems:
    # - "lt": Legendre transform
    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    # - "dr": Diffusion-reaction
    # - "cvc": Advection
    # - "advd": Advection-diffusion
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    problem = "advd"
    T = 1
    
    # Harmonic Oscillator
    omega = 1
    p0 = 1
    q0 = 0
    harm_oscil_params = [omega, p0, q0]
    if problem == "lt":
        npoints_output = 20
        system = lt_system(npoints_output)
    elif problem == "ode":
        #system = ode_system(T, system_params = harm_oscil_params)
        system = ode_system(T, system_params = harm_oscil_params)
        print("ode_system created")
    elif problem == "dr":
        npoints_output = 100
        system = dr_system(T, npoints_output)
    elif problem == "cvc":
        npoints_output = 100
        system = cvc_system(T, npoints_output)
    elif problem == "advd":
        npoints_output = 100
        system = advd_system(T, npoints_output)

    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    m = 300
    #num_train = 10000
    #num_test = 100000
    num_train = 150 # Number of final time points we want to predict
    num_test = 300
    
    # for testing
    #m = 20
    #num_train = 300
    #num_test = 100
    lr = 0.001
    epochs = 50000
    #epochs = 1
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
            [dim_x, 40, 40],
            activation,
            initializer# For harmonic oscillator
        )
        #net = dde.nn.DeepONetCartesianProd(
        #    [m, 40, 40],
        #    [dim_x, 40, 40],
        #    "relu",
        #    "Glorot normal",
        #)

    elif nn == "fnn":
        print("Using FNN")
        net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    elif nn == "resnet":
        print("Using resnet")
        net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)

    run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)


if __name__ == "__main__":
    main()
