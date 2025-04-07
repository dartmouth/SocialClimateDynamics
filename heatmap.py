#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
import seaborn as sns


delay = 10

# Define the delay differential equations
def model(Y, t, d, Tv, Tc):
    v, x = Y(t)
    v_lag, x_lag = Y(t-d)

    # Equation for dv/dt
    dv_dt = 2 * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * (0.2 + 0.4 * x) * v * (1 - v) - 0.2 * v

    # Equation for dx/dt
    dx_dt = x * (1 - x) * (2 * x - 2 + 5 / (1 + np.exp(-3 * (-7.5 * v + 7.5 * v_lag - Tc))))

    return [dv_dt, dx_dt]

gamma = 0.2
g0=2
Topt = 28
beta=10
a=5


def uncoupled_model(v,t,gamma,Tv):
    dvdt = v*(2*(1-v)*(1-((Tv+a*(1-v)-Topt)/beta)**2)-gamma)
    return dvdt


def history_function(v0, x0, Tv):
    time = np.linspace(0, delay,10000)
    v=odeint(uncoupled_model, v0, time,args=(gamma, Tv))
    v = v.flatten()
    x0_array = np.full_like(v, x0)
    output = np.column_stack((v, x0_array))
    def history(t):
        idx = np.searchsorted(time, t)
        if idx == 0:
            return output[0]
        else:
            print("Time=", time[idx-1], output[idx-1])
            return output[idx - 1]
    return history



Tv_values = np.linspace(10, 35, 25)
Tc_values = np.linspace(0, 3, 25)
v0, x0 = 0.1, 0.9
tspan = np.linspace(delay, 200, 1000)

    
def classify_outcome(v_final, x_final):
    if v_final < 0.05 and x_final < 0.05:
        return v_final
    elif x_final > 0.95:
        return v_final + x_final
    elif v_final > 0.65:
        return 2 * (v_final + x_final)
    else:
        return 3 * (v_final + x_final)

# Initialize the grid for classifications
outcome_grid = np.empty((len(Tc_values), len(Tv_values)))

for i, Tc in enumerate(Tc_values):
    for j, Tv in enumerate(Tv_values):
        hist = history_function(v0, x0, Tv)
        sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
        v_final_val, x_final_val = sol[-1]
        outcome_grid[i, j] = classify_outcome(v_final_val, x_final_val)
      

sns.heatmap(data=outcome_grid)
plt.title('v0=0.1, x0=0.9')

plt.show()


