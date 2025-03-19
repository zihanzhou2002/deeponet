import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm 

def plot_schrodinger_3d(y_pred, y_true, model,net, optimizer, potential="zero", x_max = 10, T = 1):
    """ Plot the 3D plots of Schrodinger's Equations. y_pred, y_true of the shape (nt, nx)

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
    ax1.plot_surface(x_grid, t_grid, np.abs(y_pred)**2, rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax1.set_title("Predicted Solution")
    ax2.plot_surface(x_grid, t_grid, np.abs(y_true)**2, rstride=1, cstride=1,cmap = cm.coolwarm, edgecolor="none")
    ax2.set_title("Groundtruth Solution")

    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\schro_pred_{model.__name__}_potential-{potential}net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()
    
    
def plot_schrodinger_prob(y_pred, y_true, model, net, optimizer, potential="zero", x_max = 10, T = 1):
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
    plt.savefig(f"C:\\Users\\zzh\\Desktop\\Oxford\\dissertation\\deeponet\\plots\\schro_prob_{model.__name__}_{potential}-potential_net-{net.branch.linears[-1].out_features}-{net.trunk.linears[-1].out_features}_l2-{optimizer.param_groups[0]["weight_decay"]}")
    plt.show()