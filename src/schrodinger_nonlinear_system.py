from scipy import integrate
from scipy import sparse
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import dft
from scipy.sparse.linalg import spsolve
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

def solve_nonlinear_schro_CK(psi_initial, Nx, Nt, V, L = 10, T=1, ):
    """
    Solve the nonlinear Schrödinger equation using Crank-Nicolson method.
    
    Parameters:
    - psi_initial: Initial wavefunction
    - x_points: Spatial grid points
    - t_points: Number of time steps
    - dx: Spatial step size
    - dt: Time step size
    - potential_func: Function for potential V(x)
    - nonlinearity: Whether to include nonlinear term |ψ|²

    Returns:
    - psi_history: Evolution of wavefunction over time
    """
    
    dx = L / (Nx - 1)
    x = np.linsapce(0, L, Nx)
    
    dt = T / (Nt - 1)  # Time step size
    t = np.linspace(0, T, Nt)
    
    
    psi = psi_initial.copy()
    psi_sol = np.zeros((Nt, Nx), dtype=complex)
    psi_sol[0] = psi
    
    # Laplacian operator (second derivative) using finite difference
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
    

    # Potential term
    
    # Identity matrix
    I = sparse.eye(Nx, format="csr")
    
    # Coefficient for the Laplacian term
    coef = 0.5j * dt / 2
    
    for t in range(1, Nt):
        # Calculate nonlinear term if applicable
        nonlinear_term = 0.5 * (np.abs(psi)**2)

        
        # Build matrices for implicit scheme
        # Left side: (I + i*dt/2 * (0.5*∇² - V - nonlinear))
        A = I + coef * (0.5 * D2 - sparse.diags(V + nonlinear_term))
        
        # Right side: (I - i*dt/2 * (0.5*∇² - V - nonlinear))
        B = I - coef * (0.5 * D2 - sparse.diags(V + nonlinear_term))
        
        # Solve the system: A * psi_next = B * psi
        psi = spsolve(A, B @ psi)
        
        # Update nonlinear term based on new psi
        for _ in range(3):  # A few iterations are usually sufficient
            nonlinear_term = 0.5 * (np.abs(psi)**2)
            A = I + coef * (0.5 * D2 - sparse.diags(V + nonlinear_term))
            psi = spsolve(A, B @ psi)
        
        psi_sol[t] = psi
        
    return psi_sol

def gen_schro_nonlinear_fourier_fixed(num= 200, potential = "zero", sensors = 500, x_max = 10, x0=3.0, sigma=0.3, t0=0,tf=1):
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
    dx = x_max / (sensors - 1)
    x     = np.linspace(0, x_max, sensors)       # spatial grid points
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

    # hbar = 1.0545718176461565e-34
    
    dt = (tf - t0) / (num - 1)
    t_eval = np.linspace(t0, tf, num)  # recorded time shots
    
    # Solve the Initial Value Problem
    sol = solve_nonlinear_schro_CK(psi0, sensors, num, V=V, L=x_max, T=tf)
    
    y_data = np.array(sol, dtype=np.complex128)
    
    initial_data = np.tile(y_data[0], (num, 1))
    
    X_hat = fft(initial_data)
    y_hat = fft(y_data)
    
    return (X_hat, t_eval), y_hat

def gen_schro_nonlinear_fourier_rand(num = 200, sensors= 500, Nt = 800, x_max = 10, potential = "zero", t0=0,tf=1):
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
    dx = x_max / (sensors - 1)
    x  = np.linspace(0, x_max, sensors)   
    dt = (tf - t0) / (Nt - 1)    # spatial grid points
    t = np.linspace(t0, tf, Nt)  # recorded time shots
    kx = 0.1                        # wave number
    m = 0.5                          # mass
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
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=num)
    x0s = np.random.uniform(x0_min, x0_max, size= num)
    t_inds = np.random.choice(np.arange(Nt), size = num)

    t_points = t[t_inds]
    
    # Generating Datasets
    initial_data = np.zeros((num, sensors),dtype=np.complex64)
    y_data = np.zeros((num, sensors),dtype=np.complex64)
    
    for i in range(num):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
        t_ind = t_inds[i]
        sol = solve_nonlinear_schro_CK(psi0, sensors, Nt, V=V, L=x_max, T=tf)
        initial_data[i] = psi0
        y_data[i] = sol[t_ind]
    
    
    X_hat = fft(initial_data)
    y_hat = fft(y_data)
    return (X_hat, t_points), y_hat

def gen_schro_nonlinear_fourier_rand_multi(nu = 200, nx = 100, potential = "zero", nt= 50,x_max = 10, t0=0,tf=1):
    
    # Specify constants
    dx = x_max / (nx - 1)
    dt = (tf - t0) / (nt -1)   # spatial grid points
    
    x = np.linspace(0, x_max, nx)       # spatial grid points
    t = np.linspace(t0, tf, nt)  # recorded time shots
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
    
    
    # Generate sets of random sigmas, x0s, ts
    sigmas = np.random.uniform(sigma_min, sigma_max, size=nu)
    x0s = np.random.uniform(x0_min, x0_max, size= nu)
    
    # Generating Datasets
    initial_data = np.zeros((nu, nx),dtype=np.complex128)
    y_data = np.zeros((nu, nt, nx),dtype=np.complex128)
    y_hat = np.zeros((nu, nt, nx),dtype=np.complex128)
    
    for i in range(nu):
        sigma = sigmas[i]
        x0 = x0s[i]
        A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

        # Initial Wavefunction
        psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
        
        sol = solve_nonlinear_schro_CK(psi0, nx, nt, V=V, L=x_max, T=tf)
        initial_data[i] = psi0
        y_data[i, :, :] = sol
        
        y_hat[i, :, :] = fft(sol)
    
    
    X_hat = fft(initial_data)
    #y_hat = fft(y_data)
    return (X_hat, t), y_hat

def main():
    # Setup parameters
    L = 10  # Domain size
    nx = 512  # Number of spatial points
    nt = 40  # Number of time steps
    nu = 200
    
    dx = L / (nx - 1)  # Spatial step size
    T = 1.0
    dt = T / (nt - 1)  # Time step size
    
    x = np.linspace(0,L, nx)
    t = np.linspace(0, T, nt)
    
    # Initial wavefunction (Gaussian wave packet)
    sigma = 1.0
    k0 = 2.0
    x0 = 3.0
    kx    = 0.1   
    A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

    # Initial Wavefunction
    psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)

    # Solve using Crank-Nicolson
    #psi_sol = solve_nonlinear_schro_CK(psi0, nx, nt, potential="zero", L=L, T=T)
    
    (initial_data_hat, t), psi_hat = gen_schro_nonlinear_fourier_rand_multi(nu=nu, nx=nx, nt=nt, x_max=L, t0=0, tf=T)
    i = 20
    psi_sol = ifft(psi_hat[i])
    #psi_hat = fft(psi_sol)
    k = (2* np.pi) * fftfreq(nx, dx)
    D = np.diag(1j * k)
    W = dft(nx)
    W_inv = W.conj().T / nx
    psi_x_sol = ifft(psi_hat[i]@D.T)
    #Omega = W@W_inv.conj().T + D.conj().T@W@W_inv.conj().T@D / nx**2
    
    energy = dx*np.array([np.sum(np.abs(psi_sol[i])**2)*0.25 + np.sum(np.abs(psi_x_sol[i])**2) *0.25 for i in range(nt)])
    

    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot initial, middle, and final probability densities
    plt.plot(x, np.abs(ifft(initial_data_hat[0]))**2, label='Initial')
    #plt.plot(x, np.abs(psi_sol[nt//2])**2, label=f'Middle (t={nt*dt/2:.1f})')
    #plt.plot(x, np.abs(psi_sol[-1])**2, label=f'Final (t={nt*dt:.1f})')
    plt.plot(x, np.abs(psi_sol[0])**2, label=f'Final (t={nt*dt:.1f})')
    
    plt.title('Schrödinger Equation: Probability Density Evolution')
    plt.xlabel('Position')
    plt.ylabel('|ψ|²')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close() 
    

    prob_true = np.array([np.sum(np.abs(y)**2 * dx) for y in psi_sol])
    # Plot predicted solution
    plt.plot(t, prob_true, label='actual probabilities')
    plt.plot(t, energy, label='energy')
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    plt.title("Total Probability over Time")
    plt.show()

if __name__ == "__main__":
    main()