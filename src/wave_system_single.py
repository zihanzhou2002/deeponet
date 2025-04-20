import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import dft
import matplotlib.pyplot as plt
from matplotlib import cm

from spaces import FinitePowerSeries, FiniteChebyshev, GRF
# ============================================================================================
# Functions for solving wave equation, producing a single vector output [u(x_1), ..., u(x_N), v(x_1), ... , v(x_N)]
# ============================================================================================

def solve_wave_single(u0, v0, c, t_ind, Nt, Nx = 100, L = 10.0, T = 1.0):
    """_summary_

    Args:
        u0 (_type_): _description_
        v0 (_type_): _description_
        c (_type_): _description_
        t_ind (_type_): _description_
        Nt (_type_): _description_
        Nx (int, optional): _description_. Defaults to 100.
        L (float, optional): _description_. Defaults to 10.0.
        T (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    
    # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)  # Time step
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
    dt = T / (Nt - 1)  # Time step
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



def solve_wave_single_fourier_FE(u0, v0, c, t_ind, Nt, Nx = 100, L = 10.0, T = 1.0):
   
   # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)  # Time step
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

def solve_wave_single_fourier_exact(u0, v0, c, t_ind, Nt, Nx = 100, L = 10.0, T = 1.0):
   
   # Number of spatial points
    dx = L / (Nx - 1)  # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)  # Time step
    t = np.linspace(0, T, Nt)  

    # Stability condition: CFL number


    # Initialize solution vectors
    #u = np.exp(-((x - x0) ** 2 / (2.0 * sigma**2))) # Initial condition (Gaussian pulse)
    #v = np.zeros_like(u)  # Initial velocity is zero
    k = 2 * np.pi * fftfreq(Nx, d=dx) 

    # Initialize solution vectors
    u0_hat = fft(u0)# Initial condition (Gaussian pulse)
    v0_hat = fft(v0) # Initial velocity is zero
    
    # Store results for visualization
    us_hat = np.zeros(Nx, dtype=np.complex128)
    vs_hat = np.zeros(Nx, dtype=np.complex128)
    

    

    for i in range(Nx):
        # Compute second spatial derivative using central difference
        u0 = u0_hat[i]
        v0 = v0_hat[i]
        
        u_hat = u0 / 2 * (np.exp(1j*c*k[i] * t[t_ind]) + np.exp( -1j*c*k[i] * t[t_ind]))
        v_hat = u0 / 2 * (1j*c*k[i]* np.exp(1j*c*k[i] * t[t_ind]) - 1j*c* k[i] *np.exp( -1j*c*k[i] * t[t_ind]))
        us_hat[i] = u_hat
        vs_hat[i] = v_hat 

    
    result = np.stack((us_hat, vs_hat), axis=1)

    
    return result


def solve_wave_whole_fourier_FE(u0, v0,c, L = 10.0, T = 1.0, Nx = 100, Nt = 500):
   
   # Number of spatial points
    dx = L / (Nx - 1)   # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)  # Time step
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


def solve_wave_whole_fourier_exact(u0, v0,c, L = 10.0, T = 1.0, Nx = 100, Nt = 500):
   
   # Number of spatial points
    dx = L / (Nx  - 1)    # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)  # Time step
    t = np.linspace(0, T, Nt)
    
    # Wave number
    k = 2 * np.pi * fftfreq(Nx, d = dx) 

    # Initialize solution vectors
    u_hat_0 = fft(u0) # Initial condition (Gaussian pulse)
    v_hat_0 = fft(v0) # Initial velocity is zero
    
    # Store results for visualization
    us_hat = np.zeros((Nt, Nx), dtype=np.complex128)
    vs_hat = np.zeros((Nt, Nx), dtype=np.complex128)

    us_hat[0] = u_hat_0
    vs_hat[0] = v_hat_0
    
    
    for i in range(Nx):
        # Compute second spatial derivative using central difference
        
        u0 = us_hat[0, i]
        v0 = vs_hat[0, i]
        
        u_hat = u0 / 2 * (np.exp(1j*c*k[i] * t) + np.exp( -1j*c*k[i] * t))
        v_hat = u0 / 2 * (1j*c*k[i]* np.exp(1j*c*k[i] * t) - 1j*c* k[i] *np.exp( -1j*c*k[i] * t))
        us_hat[:, i] = u_hat
        vs_hat[:, i] = v_hat
    
    
    result = np.stack((us_hat, vs_hat), axis=2)
    return result

def solve_wave_whole_fourier_CK(u0, v0,c, L = 10.0, T = 1.0, Nx = 100, Nt = 500):
   
   # Number of spatial points
    dx = L / (Nx - 1)    # Spatial step
    x = np.linspace(0, L, Nx)  # Spatial grid

    # Number of time steps
    dt = T / (Nt - 1)   # Time step
    t = np.linspace(0, T, Nt)
    
    # Wave number
    k = 2 * np.pi * fftfreq(Nx, d = dx) 

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
        
        rk = (c*k)**2
        lk = (dt**2) / 4 * rk
    
        v_hat = ((1 - lk)*v_hat - dt * rk * u_hat)  / (1 + lk)
        u_hat = ((1 - lk)*u_hat + dt * v_hat)  / (1 + lk)

    
    result = np.stack((us_hat, vs_hat), axis=2)
    return result


def raised_cosine_edges(N, edge_ratio=0.1):
    """
    Create a window that is 1 in the middle and tapers to 0 at the edges.
    
    Args:
        N: Length of the window
        edge_ratio: Portion of the window (on each side) that tapers
    
    Returns:
        window: Array of length N
    """
    window = np.ones(N)
    edge_width = int(N * edge_ratio)
    
    # Left edge (taper from 0 to 1)
    for i in range(edge_width):
        window[i] = 0.5 * (1 - np.cos(np.pi * i / edge_width))
        
    # Right edge (taper from 1 to 0)
    for i in range(edge_width):
        window[N - i - 1] = 0.5 * (1 - np.cos(np.pi * i / edge_width))
        
    return window

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
    dx = x_max / (Nx - 1)
    x     = np.linspace(0, x_max, Nx)       # spatial grid points
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
    dx = x_max / (Nx - 1)
    x = np.linspace(0, x_max, Nx)       # spatial grid points

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
        sol = solve_wave_single_fourier_exact(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i][0 : Nx] = sol[:, 0]
        y_data[i][Nx : ] = sol[:, 1]
    
    
    return (initial_data, t_points), y_data



def gen_wave_fourier_rand_GRF_fixed_speed(num = 200, Nx= 500, Nt = 800, x_max = 10, tf=1):
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
    dx = x_max / (Nx - 1)
    x = np.linspace(0, x_max, Nx)   # spatial grid points
    c = 1.0
    # Range of randomized sigmas and x0s
    window = np.sin(np.pi * x / x_max)**2
    
    # Range of randomized sigmas and x0s
    space = GRF(x_max, kernel="RBF", length_scale=0.1, N=Nx, interp="cubic")
    u0s_raw = space.random(num)
    
    # Apply window to ensure zero boundary conditions
    u0s = np.array([u0_raw * window for u0_raw in u0s_raw])
    
    t = np.linspace(0, tf, Nt)
    
    # Generate sets of random sigmas, x0s, ts
    t_inds = np.random.choice(np.arange(Nt), size = num)

    t_points = t[t_inds]
    
    # Generating Datasets
    initial_data = np.zeros((num, Nx*2),dtype=np.complex128)
    y_data = np.zeros((num, Nx*2),dtype=np.complex128)
    
    for i in range(num):
        u0 = u0s[i]# Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        t_ind = t_inds[i]
        initial_data[i][0: Nx] = fft(u0)
        initial_data[i][Nx: ] = fft(v0)
        # Initial Wavefunction
        sol = solve_wave_single_fourier_exact(u0=u0, v0=v0, t_ind = t_ind, Nt=Nt, c= c, Nx = Nx, L=x_max, T = tf)

        y_data[i][0 : Nx] = sol[:, 0]
        y_data[i][Nx : ] = sol[:, 1]
    
    
    return (initial_data, t_points), y_data


def gen_wave_fourier_rand_GRF_fixed_speed_multi(Nu = 200, Nx= 500, Nt = 800, x_max = 10, tf=1):
    """ Generate random initial conditions for multiple times

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
    dx = x_max / (Nx - 1)
    x = np.linspace(0, x_max, Nx)   # spatial grid points
    c = 1.0
    # Range of randomized sigmas and x0s
    window = np.sin(np.pi * x / x_max)**2
    
    # Range of randomized sigmas and x0s
    space = GRF(x_max, kernel="RBF", length_scale=0.1, N=Nx, interp="cubic")
    u0s_raw = space.random(Nu)
    
    # Apply window to ensure zero boundary conditions
    u0s = np.array([u0_raw * window for u0_raw in u0s_raw])
    
    ts = np.linspace(0, tf, Nt)

    
    # Generating Datasets
    initial_data = np.zeros((Nu, 2*Nx),dtype=np.complex128)
    y_data = np.zeros((Nu, Nt, 2*Nx),dtype=np.complex128)
    
    for i in range(Nu):
        u0 = u0s[i]# Initial condition (Gaussian pulse)
        v0 = np.zeros_like(u0)  # Initial velocity is zero
        initial_data[i][0: Nx] = fft(u0)
        initial_data[i][Nx: ] = fft(v0)
        
        # Initial Wavefunction
        sol = solve_wave_whole_fourier_exact(u0=u0, v0=v0, Nt=Nt, c = c, Nx = Nx, L=x_max, T = tf)

        y_data[i,:, 0:Nx] = sol[:,:, 0]
        y_data[i, :, Nx:] = sol[:,:, 1]
    
    
    return (initial_data, ts), y_data



def gen_wave_dataset_init_fixed_speed(num = 200, sigma = 0.3, x0 = 5.0, c = 1, Nx= 500, x_max = 10, tf=1):
    """ Generate fixed initial conditions and their corresponding behaviour at different times

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
    dx = x_max / (Nx - 1)
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
    """ Generate fixed initial conditions and their corresponding behaviour at different times

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
    dx = x_max / (Nx - 1) 
    x     = np.linspace(0, x_max, Nx)       # spatial grid points

    
    Nt = num
    dt = tf / Nt
    t = np.arange(0, tf, dt)
    u0 = np.zeros(Nx, dtype=np.complex128)
    def init_fn(x, x0, sigma):
        val = np.exp(-((x - x0)**2)/sigma)
        if val<.001:
            return 0.0
        else:
            return val
    
    # Generate sets of random sigmas, x0s, ts
    for a in range(Nx):
        u0[a]=init_fn(x[a], x0, sigma)

    v0 = np.zeros_like(u0)  # Initial velocity is zero
    initial = np.hstack((fft(u0), fft(v0)))
    initial_data = np.tile(initial, (num,1)).astype(np.complex128)
    y_data = np.zeros((num, Nx*2), dtype=np.complex128)
    
    # Generating Datasets
    #initial_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    #y_data = np.zeros((num, Nx, 2),dtype=np.complex128)
    
    # Initial Wavefunction
    sol = solve_wave_whole_fourier_exact(u0 = u0, v0 = v0 ,c = c, L = x_max, T = tf, Nx = Nx, Nt = Nt)
    for i in range (num):
        y_data[i][0: Nx] = sol[i, :, 0]
        y_data[i][Nx :] = sol[i, :, 1]
    
    
    return (initial_data, t), y_data


# ===========================================================================================



# ===========================================================================================


def plot_wave_3d(y_pred, y_true, model,net, optimizer, loss_fn, x_max = 10, T = 1):
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
    
    #W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    #print(f"W inv{W_large_inv.shape}")
    #print(f"y_pred {y_pred.shape}")
    y_pred_sol = y_pred@W_large_inv.conj().T
    y_true_sol = y_true@W_large_inv.conj().T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    
   
    
    # Plot predicted solution
    ax1.plot_surface(x_grid, t_grid, y_pred_sol[:, 0: nx], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax1.set_title("Predicted Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u(x,t)")
    ax2.plot_surface(x_grid, t_grid, y_true_sol[:, 0 : nx], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax2.set_title("Groundtruth Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")

    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_pred_3d_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_loss-{loss_fn.__name__}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
def plot_wave_2d(y_pred, y_true, model,net, optimizer,loss_fn, x_max = 10, T = 1):
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
    
    #W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    #print(f"W inv{W_large_inv.shape}")
    #print(f"y_pred {y_pred.shape}")

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    y_pred_sol = y_pred@W_large_inv.conj().T
    y_true_sol = y_true@W_large_inv.conj().T
    max1 = np.max(np.abs(y_true_sol[:, 0:nx]))
    max2 = np.max(np.abs(y_pred_sol[:, 0:nx]))
    absmax = max(max1, max2)
    absmax = max(absmax, 1e-10)
    
    pcm = ax1.pcolormesh(x_grid, t_grid, y_true_sol[:, 0:nx].real, shading='auto', cmap= cm.coolwarm, vmin=-absmax, vmax=absmax)
    ax1.set_title("Groundtruth Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    pcm = ax2.pcolormesh(x_grid, t_grid, y_pred_sol[:, 0:nx].real, shading='auto', cmap= cm.coolwarm, vmin=-absmax, vmax=absmax)
    ax2.set_title("Predicted Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    fig.colorbar(pcm, ax=ax2)  # optional: shows the color scale
    
    error = y_pred_sol[:, 0:nx].real - y_true_sol[:, 0:nx].real

    absmax_err = np.max(np.abs(error))
    absmax_err = max(absmax_err, 1e-10)
    pcm = ax3.pcolormesh(x_grid, t_grid,error, shading='auto', cmap= cm.bwr, vmin=-absmax_err, vmax = absmax_err)
    ax3.set_title("Error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    fig.colorbar(pcm, ax=ax3)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_pred_2d_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_loss_fn-{loss_fn.__name__}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()

    
def plot_wave_energy(y_pred, y_true, model, net, optimizer, loss_fn, c, x_max = 10, T = 1):
    """ Plot the 3D plots of Schrodinger's Equations. y_pred, y_true of the shape (nt, nx)

    Args:
        y_pred (nt, nx): predicted solution
        y_true (nt, nx): actual solution
        x_max (float): upper bound of spatial domain. Defaults to 10.
        T (float): upper bound of temporal domain. Defaults to 1.
    """
    nx = int(np.shape(y_pred)[1] / 2)
    dx = x_max / (nx - 1)
    #k = (2 * np.pi / x_max) * fftfreq(nx, dx)  # Correct physical frequencies
    k = 2 * np.pi * fftfreq(nx, dx)
    
    W = dft(nx)
    W_inv = W.conj().T / nx
    
    W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    # Construct differentiation matrix
    D = np.diag(1j * k)
    D_large = np.block([[D, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.eye(nx)]])
    
    
    y_pred_x_sol = y_pred@D_large.conj().T@W_large_inv.conj().T
    y_true_x_sol = y_true@D_large.conj().T@W_large_inv.conj().T
    
    fig = plt.figure()
    
    t = np.linspace(0, T, len(y_true))   
    
    energy_pred = dx*np.array([np.sum(c**2*np.abs(y[0: nx])**2) + np.sum(np.abs(y[nx :])**2) for y in y_pred_x_sol])
    energy_true = dx*np.array([np.sum(c**2*np.abs(y[0: nx])**2) + np.sum(np.abs(y[nx :])**2) for y in y_true_x_sol])
    # Plot predicted solution
    energy_max = max(np.max(np.abs(energy_true)), np.max(np.abs(energy_pred)))
    
    
    plt.plot(t, energy_pred, label='predicted energy')
    plt.plot(t, energy_true, label='actual energy')
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylim(0, (energy_max // 10)*10 + 20)
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Total Energy over Time")
    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_energy_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_loss-{loss_fn.__name__}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
    
"""
def main():
    num = 201
    nx = 20
    L = 10
    T = 2.0
    c = 1
    nt = 41
    
    X_train, y_train = gen_wave_fourier_rand_GRF_fixed_speed_multi(Nu = num, Nx = nx, Nt = nt, x_max = L, tf = T)
    X_test_fixed, y_test_fixed = gen_wave_fourier_rand_GRF_fixed_speed_multi(Nu = 5, Nx = nx, Nt = nt, x_max = L, tf = T)
    nx = int(np.shape(y_test_fixed)[-1] / 2)
    
     # Correct physical frequencies
     
    
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)   
    
    k = (2 * np.pi) * fftfreq(nx, dx) 
    x_grid, t_grid  = np.meshgrid(x, t)
    
    i = 9
    
    plt.plot(x, X_test_fixed[0][0, :nx], label="initial condition")
    plt.plot(x, y_test_fixed[0, i,0:nx], label = "after certain time")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    W = dft(nx)
    W_inv = W.conj().T / nx
    
    W_large = np.kron(np.eye(2), W)
    W_large_inv = np.kron(np.eye(2), W_inv)
    #print(f"W inv{W_large_inv.shape}")
    #print(f"y_pred {y_pred.shape}")
    y_true_sol = y_test_fixed@W_large_inv.T # for single
    #y_true_sol = W_large_inv@y_test_fixed[0] # for whole
    j = 0
    
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.plot_surface(x_grid, t_grid, y_true_sol[j, :, 0:nx], rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    plt.show()
    plt.close()
    

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_grid, t_grid, y_true_sol[j,:, 0:nx].real, shading='auto', cmap= cm.coolwarm)
    fig.colorbar(pcm, ax=ax)  # optional: shows the color scale
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Solution Heatmap")
    plt.show()
    plt.close()
    # Construct differentiation matrix
    D = np.diag(1j * k)
    D_large = np.block([[D, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.eye(nx)]])

    
    y_true_x_sol = y_test_fixed[0]@D_large.conj().T@W_large_inv.conj().T
    #y_true_x_sol = W_large_inv@D_large@y_test_fixed[0]
    
    fig = plt.figure()

    x_grid, t_grid  = np.meshgrid(x, t)
    
    energy_true = dx*np.array([np.sum(c**2*np.abs(y[0:nx])**2) + np.sum(np.abs(y[nx:])**2) for y in y_true_x_sol])
    
    
    # Plot predicted solution
    plt.plot(t, energy_true, label='actual energies')
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Total Energy over Time")
    #plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\wave_energy_{model.__name__}_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
    print("Finished")

if __name__ == "__main__":
    main()
"""