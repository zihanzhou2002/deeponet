import numpy as np
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt
from matplotlib import cm
# ============================================================================================
# Functions for solving wave equation
# ============================================================================================

def solve_wave_single(u0, v0, c, t_ind, Nt, Nx = 100, L = 10.0, T = 1.0):
   
   # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / Nt  # Time step
    t = np.linspace(0, T, Nt)  

    # Stability condition: CFL number
    CFL = c * dt / dx
    if CFL > 1:
        print("Warning: CFL condition not satisfied, decrease dt or increase dx.")

    # Initialize solution vectors
    #u = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
    #v = np.zeros_like(u)  # Initial velocity is zero
    u = u0
    v = v0

    # Time evolution using finite differences
    u_new = np.copy(u)
    v_new = np.copy(v)

    # Store results for visualization
    us = np.zeros((Nt, Nx))
    vs = np.zeros((Nt, Nx))

    for n in range(Nt):
        # Compute second spatial derivative using central difference
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2
        
        # Update equations
        v_new[1:-1] = v[1:-1] + dt * c**2 * u_xx[1:-1]
        u_new[1:-1] = u[1:-1] + dt * v[1:-1]

        # Enforce boundary conditions (e.g., Dirichlet u = 0)
        u_new[0] = u_new[-1] = 0
        v_new[0] = v_new[-1] = 0

        # Update values
        u, u_new = u_new, u  # Swap references
        v, v_new = v_new, v

        # Store solution at some time steps
        us[n] = np.copy(u)
        vs[n] = np.copy(v)

    
    return np.stack((us[t_ind], vs[t_ind]), axis = 1)


def solve_wave_whole(u0, v0,c, L = 10.0, T = 1.0, Nx = 100, Nt = 500):
   
   # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / Nt  # Time step
    t = np.linspace(0, T, Nt)
    

    # Stability condition: CFL number
    CFL = c * dt / dx
    if CFL > 1:
        print("Warning: CFL condition not satisfied, decrease dt or increase dx.")

    # Initialize solution vectors
    u = u0# Initial condition (Gaussian pulse)
    v = v0 # Initial velocity is zero

    # Time evolution using finite differences
    u_new = np.copy(u)
    v_new = np.copy(v)

    # Store results for visualization
    us = np.zeros((Nt, Nx))
    vs = np.zeros((Nt, Nx))

    for n in range(Nt):
        # Compute second spatial derivative using central difference
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2
        
        # Update equations
        v_new[1:-1] = v[1:-1] + dt * c**2 * u_xx[1:-1]
        u_new[1:-1] = u[1:-1] + dt * v[1:-1]

        # Enforce boundary conditions (e.g., Dirichlet u = 0)
        u_new[0] = u_new[-1] = 0
        v_new[0] = v_new[-1] = 0

        # Update values
        u, u_new = u_new, u  # Swap references
        v, v_new = v_new, v

        # Store solution at some time steps
        us[n] = np.copy(u)
        vs[n] = np.copy(v)

    
    result = np.stack((us, vs), axis=2)
    return result

   
# ============================================================================================
# Functions for generating triplet datasets
# ============================================================================================

def gen_wave_dataset_rand_fixed_speed(num = 200, Nx= 500, Nt = 800, x_max = 10, tf=1):
    """ Generate random initial conditions and their corresponding behaviour at different times.
    Here the wave speed c is fixed

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1(x, 0)], [u_2(x, 0)], ... [u_num(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1(x, t_1)], ..., [u_num(x, t_num)]]
    """
    # Specify constants
    dx = x_max / Nx
    x     = np.arange(0, x_max, dx)       # spatial grid points
    hbar = 1

    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 1.2
    
    x0_min = 2.0
    x0_max = 8.0
    
    # Wave speed
    c = 1.0
    
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    t_inds = np.random.choice(np.arange(Nt), size = num)

    t_points = t[t_inds]
    
    # Generating Datasets
    initial_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    y_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        t_ind = t_inds[i]
        u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        initial_data[i] = np.stack((u0, v0), axis=1)
        # Initial Wavefunction
        sol = solve_wave_single(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i] = sol
    
    
    return (initial_data, t_points), y_data




def gen_wave_fourier_rand_fixed_speed(num = 200, Nx= 500, Nt = 800, x_max = 10, tf=1):
    """ Generate random initial conditions and their corresponding behaviour at different times

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1(x, 0)], [u_2(x, 0)], ... [u_num(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1(x, t_1)], ..., [u_num(x, t_num)]]
    """
    # Specify constants
    dx = x_max / Nx
    x     = np.arange(0, x_max, dx)       # spatial grid points
    hbar = 1

    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 1.2
    
    x0_min = 2.0
    x0_max = 8.0
    
    # Wave speed
    c = 1.0
    
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    t_inds = np.random.choice(np.arange(Nt), size = num)

    t_points = t[t_inds]
    
    # Generating Datasets
    initial_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    y_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        t_ind = t_inds[i]
        u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        initial_data[i] = np.stack((fft(u0), fft(v0)), axis=1)
        # Initial Wavefunction
        sol = solve_wave_single(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i] = fft(sol, axis=0)
    
    
    return (initial_data, t_points), y_data



def gen_wave_dataset_init_fixed_speed(num = 200, sigma = 0.3, x0 = 5.0, c = 1, Nx= 500, x_max = 10, tf=1):
    """ Generate random initial conditions and their corresponding behaviour at different times

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1(x, 0)], [u_2(x, 0)], ... [u_num(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1(x, t_1)], ..., [u_num(x, t_num)]]
    """
    # Specify constants
    dx = x_max / Nx
    x     = np.linspace(0, x_max, Nx)       # spatial grid points

    
    Nt = num
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) 
    v0 = np.zeros_like(u0)  # Initial velocity is zero
    initial = np.stack((u0, v0), axis=1)
    initial_data = np.tile(initial, (num,1, 1))
    
    
    # Generating Datasets
    #initial_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    #y_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    
    # Initial Wavefunction
    sol = solve_wave_whole(u0 = u0, v0 = v0 ,c = c, L = x_max, T = tf, Nx = Nx, Nt = Nt)
    sol = sol.astype('complex64')
    
    
    return (initial_data, t), sol.astype('complex64')



def gen_wave_fourier_init_fixed_speed(num = 200, sigma = 0.3, x0 = 5.0, c = 1, Nx= 500, x_max = 10, tf=1):
    """ Generate random initial conditions and their corresponding behaviour at different times

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1(x, 0)], [u_2(x, 0)], ... [u_num(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1(x, t_1)], ..., [u_num(x, t_num)]]
    """
    # Specify constants
    dx = x_max / Nx
    x     = np.linspace(0, x_max, Nx)       # spatial grid points

    
    Nt = num
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) 

    v0 = np.zeros_like(u0)  # Initial velocity is zero
    initial = np.stack((u0, v0), axis=1)
    initial_data = np.tile(initial, (num,1, 1)).astype(np.complex64)
    
    # Generating Datasets
    #initial_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    #y_data = np.zeros((num, Nx, 2),dtype=np.complex64)
    
    # Initial Wavefunction
    sol = solve_wave_whole(u0 = u0, v0 = v0 ,c = c, L = x_max, T = tf, Nx = Nx, Nt = Nt)
    sol = sol.astype('complex64')
    
    
    return (fft(initial_data, axis=1), t), fft(sol, axis=1)



# ===========================================================================================



# ===========================================================================================


def plot_wave_3d(y_pred, y_true, model,net, optimizer, x_max = 10, T = 1):
    """ Plot the 3D plots of wave Equations. y_pred, y_true of the shape (nt, nx)

    Args:
        y_pred (nt, nx): predicted solution
        y_true (nt, nx): actual solution
        x_max (float): upper bound of spatial domain. Defaults to 10.
        T (float): upper bound of temporal domain. Defaults to 1.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    
    x = np.linspace(0, x_max, np.shape(y_true)[1])
    t = np.linspace(0, T, len(y_true))   
    x_grid, t_grid  = np.meshgrid(x, t)
    
    # Plot predicted solution
    ax1.plot_surface(x_grid, t_grid, y_pred[:, :, 0], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax1.set_title("Predicted Solution")
    ax2.plot_surface(x_grid, t_grid, y_pred[:, :, 0], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax2.set_title("Groundtruth Solution")

    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_pred_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
    
def plot_wave_energy(y_pred, y_true, model, net, optimizer, x_max = 10, T = 1):
    """ Plot the 3D plots of Schrodinger's Equations. y_pred, y_true of the shape (nt, nx)

    Args:
        y_pred (nt, nx): predicted solution
        y_true (nt, nx): actual solution
        x_max (float): upper bound of spatial domain. Defaults to 10.
        T (float): upper bound of temporal domain. Defaults to 1.
    """
    dx = x_max / np.shape(y_pred)[1]
    fig = plt.figure()
    
    t = np.linspace(0, T, len(y_true))   
    
    prob_pred = np.array([np.sum(np.abs(y)**2 * dx) for y in y_pred])
    prob_true = np.array([np.sum(np.abs(y)**2 * dx) for y in y_true])
    # Plot predicted solution
    plt.plot(t, prob_pred, label='predicted probabilities')
    plt.plot(t, prob_true, label='actual probabilities')
    plt.ylim(-2, 2)
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.title("Total Probability over Time")
    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\schro_prob_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()