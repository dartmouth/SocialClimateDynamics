#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
import seaborn as sns
from dde_figs import model, history_function
#from numba import njit


delay = 10
steps = 25



Tv_values = np.linspace(10, 35, steps)
Tc_values = np.linspace(0, 3, steps)
v0, x0 = 0.1, 0.9
tspan = np.linspace(delay, 200, 1000)

    
def classify_outcome(v_final, x_final):
    return v_final + x_final


def create_heatmap_temp(y_vals, x_vals):
    outcome_grid = np.empty((len(y_vals), len(x_vals)))
    for i, Tc in enumerate(y_vals):
        for j, Tv in enumerate(x_vals):
            hist = history_function(v0, x0, Tv)
            sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
            v_final_val, x_final_val = sol[-1]
            outcome_grid[i, j] = classify_outcome(v_final_val, x_final_val)
    return outcome_grid
      

sns.heatmap(data=create_heatmap_temp(Tc_values, Tv_values))
plt.title('v0=0.1, x0=0.9')
plt.show()

def create_heatmap_conditional(y_vals, x_vals, type='temp', v0=0.1, x0=0.9):
    outcome_grid = np.empty((len(y_vals), len(x_vals)))
    if type == 'warm': # Option for warming cost parameterization
        Tv = 30
        Tc = 2
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                hist = history_function(v0,x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, y, x))
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val)
        return outcome_grid
    elif type == 'social': # Option for social dynamics parameterization
        Tv = 30
        Tc = 2
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                hist = history_function(v0,x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, 5, 3, y, x))
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val)
        return outcome_grid
    else:
        for i, Tc in enumerate(y_vals): # Temperature parameterization (Fu Paper Replication)
            for j, Tv in enumerate(x_vals):
                hist = history_function(v0, x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i, j] = classify_outcome(v_final_val, x_final_val)
        return outcome_grid

# sns.heatmap(data=create_heatmap_conditional(Tc_values, Tv_values, type='temp', v0=0.2, x0=0.8))
# #plt.title(f"v0={v0}, x0={x0}")
# plt.xticks(np.arange(0,25,step=1), labels = [10 + i for i in range(0, 25)])
# plt.yticks(np.arange(0,25, step=1), labels = [i/8 for i in range(0, 25)])
# plt.show()

fmax_vals = np.linspace(0,10,steps)
w_vals = np.linspace(0,10,steps)


#sns.heatmap(data=create_heatmap_conditional(fmax_vals, w_vals, type='warm'))
#plt.xticks(np.arange(0,25,step=1), labels = [i/2.5 for i in range(0, 25)])
#plt.yticks(np.arange(0,25, step=1), labels = [i/2.5 for i in range(0, 25)])
#plt.show()
            
# Tv = 30
# Tc = 2

# def create_heatmap_warmcost(fmax_vals, w_vals):
#     outcome_grid = np.empty((len(fmax_vals), len(w_vals)))
#     for i, fmax in enumerate(fmax_vals):
#         for j, w in enumerate(w_vals):
#             hist = history_function(v0,x0, Tv)
#             sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, fmax, w))
#             v_final_val, x_final_val = sol[-1]
#             outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val)
#     return outcome_grid

# fmax_values = np.linspace(0,10,25)
# w_values = np.linspace(0,10,25)
# sns.heatmap(data=create_heatmap_warmcost(fmax_values, w_values))
# plt.title('Tv = 30, Tc = 2, v0 = 0.1, x0 = 0.9')
# plt.xticks(ticks = np.arange(0, 10,10))
# plt.yticks(ticks = np.arange(0, 10, 10))
# plt.show()

# def create_heatmap_socialdynamics(social_vals, cost_vals):
#     outcome_grid = np.empty((len(social_vals), len(cost_vals)))
#     for i, delta in enumerate(social_vals):
#         for j, alpha in enumerate(cost_vals):
    #         hist = history_function(v0,x0, Tv)
    #         sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, 5, 3, delta, alpha))
    #         v_final_val, x_final_val = sol[-1]
    #         outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val)
    # return outcome_grid

# delta_values = np.linspace(0,2,25)
# alpha_values = np.linspace(0,2,25)
# sns.heatmap(data=create_heatmap_socialdynamics(delta_values, alpha_values))
# plt.title('Tv = 30, Tc = 2, v0 = 0.1, x0 = 0.9, fmax = 5, w = 3')
# plt.xticks(ticks = np.arange(0,2,10))
# plt.yticks(ticks = np.arange(0,2,10))
# plt.show()


def reverse_parameter_sweep(Tv, Tc):
    v_vals = np.linspace(0, 1, 25)
    x_vals = np.linspace(0, 1, 25)
    outcome_grid = np.empty((len(v_vals), len(x_vals)))
    for i, v in enumerate(v_vals):
        for j, x in enumerate(x_vals):
            hist = history_function(v, x, Tv)
            sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
            v_final_val, x_final_val = sol[-1]
            outcome_grid[i, j] = classify_outcome(v_final_val, x_final_val)
    return outcome_grid


#sns.heatmap(data=reverse_parameter_sweep(30, 2))
#plt.show()