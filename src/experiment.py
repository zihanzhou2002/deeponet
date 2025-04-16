import numpy as np
import scipy
from scipy.fft import fft, ifft, fftfreq
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from matplotlib import pyplot as plt
"""
N = 100
M = 40
space = GRF(1, kernel="RBF", length_scale=0.1, N=N, interp="cubic")
u0s = space.random(40)

print(u0s.shape)
for i in range(8):
    plt.plot(space.x,u0s[i])
    
plt.show()

ans = 0
"""

x = 8
k = fftfreq(4)
for i in range(4):
    for j in range(4):
        print(f"i={i}, j={j}")
        res = np.exp(1j* k[i]*2* np.pi * x) * np.exp(1j* (-k[j])*2 * np.pi * x)
        print(res) if res.real > 1e-10 or res.imag > 1e-10 else print(0)
        