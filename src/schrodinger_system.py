from scipy import integrate
from scipy import sparse
from scipy.fft import fft, ifft
import scipy
from spaces import *

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from IPython.display import HTML

import numpy as np

# =============================================================================================
#   Complex Data of triplet format (X_func, X_loc), y,
#   (The simpler data structure, focusing only on mapping from u_(x, 0) to u(x, t) )
# ---------------------------------------------------------------------------------------------
#   X_func (L x N): Initial conditions u_i(x, 0) (0 <= i <= L - 1),  evaluated at M spatial locations
#   x_loc (L,) : Time steps t_i (0 <= i <= L - 1)
#   y (L x N): Corresponding u_i(x, t_i) for each i
# =============================================================================================

def gen_schro_fourier_fixed(num= 200, sensors = 500, potential = "zero", x_max = 10, x0=3.0, sigma=0.3, t0=0,tf=1):
    """ Generate the entire dynamics of one Schrodinger equations with fixed initial conditions

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        x0 (float, optional): cetre of initial conditions. Defaults to 3.0.
        sigma (float, optional): Defaults to 0.3.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): repeated copies of identical initial conditions [[u(x, 0)], [u(x, 0)], ... [u(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u(x, t_1)], ..., [u(x, t_num)]]
    """
    dx = x_max / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass

    # For initial conditions
    A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

    # Initial Wavefunction
    psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
    

    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    
    # RHS of Schrodinger Equation
    hbar = 1
    # hbar = 1.0545718176461565e-34
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    dt = (tf - t0) / num
    t_eval = np.arange(t0, tf, dt)  # recorded time shots

# Solve the Initial Value Problem
    sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
    
    y_data = np.array(sol.y, dtype=np.complex64)
    y_data = np.transpose(y_data)
    
    initial_data = np.tile(y_data[0], (num, 1))
    
    return (initial_data, t_eval), y_data


def gen_schro_fourier_fixed(num= 200, potential = "zero", sensors = 500, x_max = 10, x0=3.0, sigma=0.3, t0=0,tf=1):
    """ Generate the Fourier coefficients of entire dynamics of one Schrodinger equations with fixed initial conditions

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        x0 (float, optional): cetre of initial conditions. Defaults to 3.0.
        sigma (float, optional): Defaults to 0.3.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): repeated copies of fourier coefficients of identical initial conditions [[u^hat(x, 0)], [u^hat(x, 0)], ... [u^hat(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u^hat(x, t) with [[u^hat(x, t_1)], ..., [u^hat(x, t_num)]]
    """
    dx = x_max / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass

    # For initial conditions
    A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

    # Initial Wavefunction
    psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
    

    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2

    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    
    # RHS of Schrodinger Equation
    hbar = 1
    # hbar = 1.0545718176461565e-34
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    dt = (tf - t0) / num
    t_eval = np.arange(t0, tf, dt)  # recorded time shots
    
    # Solve the Initial Value Problem
    sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
    
    y_data = np.array(sol.y, dtype=np.complex64)
    y_data = np.transpose(y_data)
    
    initial_data = np.tile(y_data[0], (num, 1))
    
    X_hat = fft(initial_data)
    y_hat = fft(y_data)
    
    return (X_hat, t_eval), y_hat


def gen_schro_dataset_rand(num = 200, sensors= 500, x_max = 10, potential = "zero", t0=0,tf=1):
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
    dx = x_max / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 2.0
    
    x0_min = 1.0
    x0_max = 9.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    # Define psi_t
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    ts = np.random.uniform(t0, tf, num)
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex64)
    y_data = np.zeros((num, sensors),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = np.array([ts[i]])
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = np.transpose(sol.y)[0]
    
    
    return (initial_data, ts), y_data


def gen_schro_fourier_rand(num = 200, sensors= 500, x_max = 10, potential = "zero", t0=0,tf=1):
    """ Generate random initial conditions and their DFT at different time steps

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1^hat(x, 0)], [u_2^hat(x, 0)], ... [u_num^hat(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1^hat(x, t_1)], ..., [u_num^hat(x, t_num)]]
    """
    # Specify constants
    dx = x_max / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 2.0
    
    x0_min = 1.0
    x0_max = 9.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    # Define psi_t
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    ts = np.random.uniform(t0, tf, num)
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex64)
    y_data = np.zeros((num, sensors),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = np.array([ts[i]])
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = np.transpose(sol.y)[0]
    
    
    X_hat = fft(initial_data)
    y_hat = fft(y_data)
    return (X_hat, ts), y_hat


def gen_schro_fourier_GRF(num = 200, sensors= 500, x_max = 10, potential = "zero", t0=0,tf=1):
    """ Generate random initial conditions and their DFT at different time steps

    Args:
        num (int, optional): Number of timesteps. Defaults to 200.
        sensors (int, optional): Number of satial locations. Defaults to 500.
        t0 (int, optional): lower bound of time domain. Defaults to 0.
        tf (int, optional): upper bound of time domain. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
            - X_func of shape (num, sensors): different initial conditions [[u_1^hat(x, 0)], [u_2^hat(x, 0)], ... [u_num^hat(x, 0)]]
            - X_loc of shape (num, ): time steps  t_1, ..., t_num to be evaluated on
            - y (num, sensors): the entire dynamic u(x, t) with [[u_1^hat(x, t_1)], ..., [u_num^hat(x, t_num)]]
    """
    # Specify constants
    dx = x_max / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 2.0
    
    x0_min = 1.0
    x0_max = 9.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    # Define psi_t
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    ts = np.random.uniform(t0, tf, num)
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex64)
    y_data = np.zeros((num, sensors),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = np.array([ts[i]])
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = np.transpose(sol.y)[0]
    
    
    X_hat = fft(initial_data)
    y_hat = fft(y_data)
    return (X_hat, ts), y_hat


# =============================================================================================
#   Complex Data of triplet format (X_func, X_loc), y,
#   (The slightly more complicated structure, studying the entire dynamics for each individual intial condition)
# ---------------------------------------------------------------------------------------------
#   X_func (L, N): DFT of initial conditions u_i(x, 0) (0 <= i <= L - 1),  evaluated at M spatial locations
#   x_loc (M,) : Time steps t_m (0 <= m <= M - 1)
#   y (L, N, M): DFT of each u_i(x_n, t_m) for each i, n, m
# =============================================================================================


def gen_schro_fourier_rand_multi(nu = 200, nx = 100, potential = "zero", nt= 50, t0=0,tf=1):
    
    # Specify constants
    dx = 10 / nx
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 2.0
    
    x0_min = 1.0
    x0_max = 9.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    # Define psi_t
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=nu)
    x0s = np.random.uniform(x0_min, x0_max, size= nu)
    ts = np.linspace(t0, tf, nt)
    
    if nu == 1:
        print(f"x0s = {x0s}")
        print(f"sigmas = {sigmas}")
    # Generating Datasets
    initial_data = np.zeros((nu, nx),dtype=np.complex64)
    y_data = np.zeros((nu, nx, nt),dtype=np.complex64)
    y_hat = np.zeros((nu, nx, nt),dtype=np.complex64)
    
    for i in range(nu):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = ts
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = sol.y
        
        y_hat[i] = np.transpose(fft(np.transpose(y_data[i])))
    
    
    X_hat = fft(initial_data)
    #y_hat = fft(y_data)
    return (X_hat, ts), y_hat


def gen_schro_fourier_fixed_multi(nu = 1, nx = 100, x0 = 5, sigma = 0.3, potential = "zero", nt= 50, t0=0,tf=1):
    
    # Specify constants
    dx = 10 / nx
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    # Range of randomized sigmas and x0s
    sigma_min = 0.2
    sigma_max = 2.0
    
    x0_min = 1.0
    x0_max = 9.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = x * 0
    
    if potential == "quadratic":
        V = 0.5 * k * (x - x_Vmin)**2
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    # Define psi_t
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    
    # Generate sets of random sigmas, x0s, ts
    #sigmas = np.random.uniform(sigma_min, sigma_max, size=nu)
    #x0s = np.random.uniform(x0_min, x0_max, size= nu)
    ts = np.linspace(t0, tf, nt)
   
    # Generating Datasets
    initial_data = np.zeros((nu, nx),dtype=np.complex64)
    y_data = np.zeros((nu, nx, nt),dtype=np.complex64)
    y_hat = np.zeros((nu, nx, nt),dtype=np.complex64)
    
    for i in range(nu):
        #sigma = sigmas[i]
        #x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = ts
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = sol.y
        
        y_hat[i] = np.transpose(fft(np.transpose(y_data[i])))
    
    
    X_hat = fft(initial_data)
    #y_hat = fft(y_data)
    return (X_hat, ts), y_hat

# =============================================================================================
# =============================================================================================









def gen_schro_dataset_fixed_real(num=500, sensors = 200, x0=3.0, sigma=0.1, t0=0,tf=1):
    dx = 10 / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass

    # For initial conditions
    A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

    # Initial Wavefunction
    psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
    

    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    
    # RHS of Schrodinger Equation
    hbar = 1
    # hbar = 1.0545718176461565e-34
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    dt = (tf - t0) / num
    t_eval = np.arange(t0, tf, dt)  # recorded time shots

# Solve the Initial Value Problem
    sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
    
    y_data = np.array(sol.y**2, dtype=np.float32)
    y_data = np.transpose(y_data)
    
    initial_data = np.tile(y_data[0], (num, 1))
    
    return (initial_data.astype(np.float32), np.array([t_eval]).T.astype(np.float32)), y_data.astype(np.float32)
    #return (np.array([y_data[0]]).astype(np.float32), np.array([t_eval]).T.astype(np.float32)), y_data.astype(np.float32)


def gen_schro_dataset_sigma(num=500, sensors=200, x0=3, t0=0,tf=1):
    dx = 10 / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    sigma_min = 0.1
    sigma_max = 2.0
    
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    ts = np.random.uniform(t0, tf, num)
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex128)
    y_data = np.zeros((num, sensors),dtype=np.complex128)
    
    for i in range(num):
        sigma = sigmas[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = np.array([ts[i]])
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = np.transpose(sol.y)[0]
    
    
    return (initial_data, ts), y_data
        
def gen_schro_dataset_x0(num=500, sensors=200, sigma=0.3, t0=0,tf=1):
    """_summary_

    Args:
        num (int, optional): Number of initial conditions. Defaults to 500.
        sensors (int, optional): Number of sensors. Defaults to 200.
        sigma (float, optional): _description_. Defaults to 0.3.
        t0 (int, optional): initial time. Defaults to 0.
        tf (int, optional): final time. Defaults to 1.

    Returns:
        (X_func, X_loc), y: 
    """
    dx = 10 / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    x0_min = 0.5
    x0_max = 9.5
    
    x0 = 3
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    x0s = np.random.uniform(x0_min, x0_max, size=num)
    ts = np.random.uniform(t0, tf, num)
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex64)
    y_data = np.zeros((num, sensors),dtype=np.complex64)
    
    for i in range(num):
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
    
        t_eval = np.array([ts[i]])
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
        initial_data[i] = psi0
        y_data[i] = np.transpose(sol.y)[0]
    
    
    return (initial_data, ts), y_data
        
        
def gen_schro_dataset_x0_cart_real(num=500, sensors=20, sigma=0.3, t0=0,tf=1):
    dx = 10 / sensors
    dt = (tf-t0) / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    x0_min = 0.5
    x0_max = 9.5
    
    x0 = 3
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    x0s = np.random.uniform(x0_min, x0_max, size=num)
    ts = np.arange(t0, tf, dt)
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.float32)
    y_data = np.zeros((num, sensors*sensors),dtype=np.float32)
    locs = np.zeros((sensors**2,2),dtype=np.float32)
    locs[:, 0] = np.tile(ts, (sensors, 1)).flatten()
    locs[:, 1] = np.tile(x[:, np.newaxis], (1, sensors)).flatten()
    
    for i in range(num):
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = ts, method="RK23")
        initial_data[i] = psi0**2
        y_data[i] = np.array(sol.y**2, dtype=np.float32).flatten()
    
    
    return (initial_data, locs), y_data
        

def gen_schro_dataset_x0_cart_complex(num=500, sensors=20, sigma=0.3, t0=0,tf=1):
    dx = 10 / sensors
    dt = (tf-t0) / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    x0_min = 0.5
    x0_max = 9.5
    
    x0 = 3
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    x0s = np.random.uniform(x0_min, x0_max, size=num)
    ts = np.arange(t0, tf, dt)
    # Generating Datasets
    initial_data = np.zeros((num, sensors, 2),dtype=np.float32)
    y_data = np.zeros((num, sensors*sensors, 2),dtype=np.float32)
    locs = np.zeros((sensors**2,2),dtype=np.float32)
    locs[:, 0] = np.tile(ts, (sensors, 1)).flatten()
    locs[:, 1] = np.tile(x[:, np.newaxis], (1, sensors)).flatten()
    
    for i in range(num):
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = ts, method="RK23")
        psi0_real = [psi.real for psi in psi0]
        psi0_imag = [psi.imag for psi in psi0]
        
        for j in range(sensors):
            initial_data[i, j, 0] = np.real(psi0[j])
            initial_data[i, j, 1] = np.imag(psi0[j])

        
        y_flat = np.array(sol.y).flatten()
        for j in range(sensors*sensors):
            y_data[i, j, 0] = np.real(y_flat[j])
            y_data[i, j, 1] = np.imag(y_flat[j])
    
    
    return (initial_data, np.expand_dims(locs, axis=0)), y_data
        
def gen_schro_dataset_x0_cart_complex_sig(num=500, sensors=20, sigma=0.3, t0=0,tf=1):
    dx = 10 / sensors
    dt = (tf-t0) / sensors
    x     = np.arange(0, 10, dx)       # spatial grid points
    kx    = 0.1                        # wave number
    m     = 0.5                          # mass
    hbar = 1
    
    x0_min = 0.5
    x0_max = 9.5
    
    x0 = 3
    
    # Potential V(x)
    x_Vmin = 5         # center of V(x)
    T      = 1           # peroid of SHO 

    # For RHS
    omega = 2 * np.pi / T
    k = omega**2 * m
    V = 0.5 * k * (x - x_Vmin)**2
    
    
    
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    
    def psi_t(t, psi):
        return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)
    
    x0s = np.random.uniform(x0_min, x0_max, size=num)
    ts = np.arange(t0, tf, dt)
    # Generating Datasets
    initial_data = np.zeros((num, sensors, 2),dtype=np.float32)
    y_data = np.zeros((num, sensors*sensors, 2),dtype=np.float32)
    locs = np.zeros((sensors**2,2),dtype=np.float32)
    locs[:, 0] = np.tile(ts, (sensors, 1)).flatten()
    locs[:, 1] = np.tile(x[:, np.newaxis], (1, sensors)).flatten()
    
    for i in range(num):
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
        sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = ts, method="RK23")
        psi0_real = [psi.real for psi in psi0]
        psi0_imag = [psi.imag for psi in psi0]
        
        for j in range(sensors):
            initial_data[i, j, 0] = np.real(psi0[j])
            initial_data[i, j, 1] = np.imag(psi0[j])

        
        y_flat = np.array(sol.y).flatten()
        for j in range(sensors*sensors):
            y_data[i, j, 0] = np.real(y_flat[j])
            y_data[i, j, 1] = np.imag(y_flat[j])
    
    
    return (initial_data, locs), y_data
        