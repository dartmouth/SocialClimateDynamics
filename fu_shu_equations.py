# Luke Wisniewski, Jan 20, 2025

# Recreating figures of Fu Paper

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Parameters
gamma = 0.2  # Disturbance rate
g0 = 2  # Maximum growth rate
Topt = 28  # Optimal temperature for vegetation growth
beta = 10  # Half-width of the growth vs temperature curve
a = 5  # Temperature difference between bare soil and forest
fmax = 5  # Maximum warming cost
w = 3  # Nonlinearity of warming cost
tp = 10  # Time period for temperature projection
tf = 15  # Time frame for temperature forecast
alpha = 1  # Cost of mitigation
delta = 1  # Social norm strength
Tc = 2.5  # Critical warming temperature
delay = 5  # Time delay for vegetation and mitigation dynamics

# Vegetation growth function
def vegetation_growth(v, g):
    return g * v * (1 - v)

# Temperature function
def temperature(v, Tv):
    return Tv + (1 - v) * a

# Growth rate function
def growth_rate(T, x):
    return g0 * (1 - ((T - Topt) / beta) ** 2) * (0.2 + 0.4 * x)

# Warming cost function
def warming_cost(Tf):
    return fmax / (1 + np.exp(-w * (Tf - Tc)))

# Replicator dynamics function
def dx_dt(x, Tf):
    return x * (1 - x) * (delta * (2 * x - 1) - alpha + warming_cost(Tf))

# Define the DDE system
def coupled_system(Y, t, Tv):
    v, x = Y(t)
    v_tau, x_tau = Y(t - delay) if t > delay else Y(0)
    
    T = temperature(v_tau, Tv)
    g = growth_rate(T, x_tau)
    dvdt = g * v_tau * (1 - v_tau) - gamma * v_tau
    Tf = tf / tp * (T - Topt)
    dxdt = dx_dt(x_tau, Tf)
    
    return np.array([dvdt, dxdt])

# Initial conditions
def history(t):
    return np.array([0.5, 0.5])  # Initial vegetation and mitigation levels

# Time span
T = np.linspace(0, 200, 1000)
Tv = 25  # Ambient temperature

# Solve the DDE
solution = ddeint(coupled_system, history, T, fargs=(Tv,))

# Extract results
v = solution[:, 0]  # Vegetation coverage over time
x = solution[:, 1]  # Proportion of mitigators over time

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(T, v, label='Vegetation Coverage (v)', color='green')
plt.plot(T, x, label='Proportion of Mitigators (x)', color='blue')
plt.xlabel('Time (years)')
plt.ylabel('Proportion')
plt.title('Coupled Social-Climate Dynamics with Delay')
plt.legend()
plt.grid()
plt.show()
