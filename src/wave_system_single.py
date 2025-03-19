import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import dft
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



def solve_wave_single_fourier(u0, v0, c, t_ind, Nt, Nx = 100, L = 10.0, T = 1.0):
   
   # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / Nt  # Time step
    t = np.linspace(0, T, Nt)  

    # Stability condition: CFL number


    # Initialize solution vectors
    #u = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
    #v = np.zeros_like(u)  # Initial velocity is zero
    k = 2 * np.pi * fftfreq(Nx, d=dx) 

    # Initialize solution vectors
    u_hat = fft(u0)# Initial condition (Gaussian pulse)
    v_hat = fft(v0) # Initial velocity is zero
    
    # Store results for visualization
    us_hat = np.zeros((Nt, Nx), dtype=np.complex128)
    vs_hat = np.zeros((Nt, Nx), dtype=np.complex128)

    for i in range(Nt):
        # Compute second spatial derivative using central difference
        us_hat[i] = u_hat
        vs_hat[i] = v_hat
        v_hat = v_hat - dt * (c**2) * (k**2) * u_hat
        u_hat = u_hat + dt * v_hat

    
    result = np.stack((us_hat[t_ind], vs_hat[t_ind]), axis=1)

    
    return result



def solve_wave_whole_fourier(u0, v0,c, L = 10.0, T = 1.0, Nx = 100, Nt = 500):
   
   # Number of spatial points
    dx = L / Nx   # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / Nt  # Time step
    t = np.linspace(0, T, Nt)
    
    # Wave number
    k = 2 * np.pi * fftfreq(Nx, d=dx) 



    # Initialize solution vectors
    u_hat = fft(u0)# Initial condition (Gaussian pulse)
    v_hat = fft(v0) # Initial velocity is zero
    
    # Store results for visualization
    us_hat = np.zeros((Nt, Nx), dtype=np.complex128)
    vs_hat = np.zeros((Nt, Nx), dtype=np.complex128)

 
    for i in range(Nt):
        # Compute second spatial derivative using central difference
        us_hat[i] = u_hat
        vs_hat[i] = v_hat
        v_hat = v_hat - dt * (c**2) * (k**2) * u_hat
        u_hat = u_hat + dt * v_hat

    
    result = np.stack((us_hat, vs_hat), axis=2)
    return result

# ============================================================================================
# Functions for generating triplet datasets
# ============================================================================================

def gen_wave_dataset_rand_fixed_speed(num = 200, Nx= 500, Nt = 800, x_max = 10, tf=1, c=1.0):
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
    

    
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    t_inds = np.random.choice(np.arange(Nt), size = num)

    t_points = t[t_inds]
    
    # Generating Datasets
    initial_data = np.zeros((num, Nx*2),dtype=np.complex128)
    y_data = np.zeros((num, Nx*2),dtype=np.complex128)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        t_ind = t_inds[i]
        u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        initial_data[i][0: Nx] = u0
        initial_data[i][Nx: ] = v0
        # Initial Wavefunction
        sol = solve_wave_single(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i][0: Nx] = sol[:, 0]
        y_data[i][Nx: ] = sol[:, 1]
    
    
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
    initial_data = np.zeros((num, Nx*2),dtype=np.complex128)
    y_data = np.zeros((num, Nx*2),dtype=np.complex128)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        t_ind = t_inds[i]
        u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        initial_data[i][0: Nx] = fft(u0)
        initial_data[i][Nx: ] = fft(v0)
        # Initial Wavefunction
        sol = solve_wave_single_fourier(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i][0 : Nx] = sol[:, 0]
        y_data[i][Nx : ] = sol[:, 1]
    
    
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
    initial = np.hstack((u0, v0))
    initial_data = np.tile(initial, (num,1)).astype(np.complex128)
    y_data = np.zeros((num, Nx*2), dtype=np.complex128)
    
    # Generating Datasets
    #initial_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    #y_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    
    # Initial Wavefunction
    sol = solve_wave_whole(u0 = u0, v0 = v0 ,c = c, L = x_max, T = tf, Nx = Nx, Nt = Nt)
    for i in range(num):
        y_data[i][0: Nx] = sol[i, :, 0]
        y_data[i][Nx :] = sol[i, :, 1]
        
  
    
    
    return (initial_data, t), y_data



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
    
    # Wave numbers
    k = 2 * np.pi * fftfreq(Nx, d=dx)  
    
    # Generate sets of random sigmas, x0s, ts
    u0 = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) 

    v0 = np.zeros_like(u0)  # Initial velocity is zero
    initial = np.hstack((fft(u0), fft(v0)))
    initial_data = np.tile(initial, (num,1)).astype(np.complex128)
    y_data = np.zeros((num, Nx*2), dtype=np.complex128)
    
    # Generating Datasets
    #initial_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    #y_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    
    # Initial Wavefunction
    sol = solve_wave_whole_fourier(u0 = u0, v0 = v0 ,c = c, L = x_max, T = tf, Nx = Nx, Nt = Nt)
    for i in range (num):
        y_data[i][0: Nx] = sol[i, :, 0]
        y_data[i][Nx :] = sol[i, :, 1]
    
    
    return (initial_data, t), y_data


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
    nx = int(np.shape(y_true)[1] / 2)
    nt = len(y_true)
    x = np.linspace(0, x_max, nx)
    t = np.linspace(0, T, nt)   
    x_grid, t_grid  = np.meshgrid(x, t)
    
    W = dft(nx)
    W_inv = W.conj().T / nx
    
    W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    #print(f"W inv{W_large_inv.shape}")
    #print(f"y_pred {y_pred.shape}")
    y_pred_sol = y_pred@W_large_inv.conj().T
    y_true_sol = y_true@W_large_inv.conj().T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    
   
    
    # Plot predicted solution
    ax1.plot_surface(x_grid, t_grid, y_pred_sol[:, 0: nx], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax1.set_title("Predicted Solution")
    ax2.plot_surface(x_grid, t_grid, y_true_sol[:, 0 : nx], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax2.set_title("Groundtruth Solution")

    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_pred_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
    
def plot_wave_energy(y_pred, y_true, model, net, optimizer,c, x_max = 10, T = 1):
    """ Plot the 3D plots of Schrodinger's Equations. y_pred, y_true of the shape (nt, nx)

    Args:
        y_pred (nt, nx): predicted solution
        y_true (nt, nx): actual solution
        x_max (float): upper bound of spatial domain. Defaults to 10.
        T (float): upper bound of temporal domain. Defaults to 1.
    """
    
    nx = int(np.shape(y_pred)[1] / 2)
    dx = x_max / nx
    k = (2 * np.pi / x_max) * fftfreq(nx, dx)  # Correct physical frequencies

    
    W = dft(nx)
    W_inv = W.conj().T / nx
    
    W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    # Construct differentiation matrix
    D = np.diag(1j * k)
    D_large = np.block([[D, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.eye(nx)]])
    
    y_pred_x = y_pred@D_large.conj().T
    y_true_x = y_true@D_large.conj().T
    
    y_pred_x_sol = y_pred@D_large.conj().T@W_large_inv.conj().T
    y_true_x_sol = y_true@D_large.conj().T@W_large_inv.conj().T
    
    fig = plt.figure()
    
    t = np.linspace(0, T, len(y_true))   
    
    energy_pred = dx*np.array([np.sum(c**2*y[0: nx]**2) + np.sum(y[nx :]**2) for y in y_pred_x_sol])
    energy_true = dx*np.array([np.sum(c**2*y[0: nx]**2) + np.sum(y[nx :]**2) for y in y_true_x_sol])
    # Plot predicted solution
    plt.plot(t, energy_pred, label='predicted energies')
    plt.plot(t, energy_true, label='actual energies')
    plt.ylim(-20, 20)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Total Probability over Time")
    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_energy_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()