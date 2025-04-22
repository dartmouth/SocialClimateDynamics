#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
import seaborn as sns
from dde_figs import model, history_function
from heterogeneous_pop import heterogeneous_pop_model, doub_history_function
#from numba import njit

max_ticks = 6

delay = 10
steps = 25

Tv_values = np.linspace(10, 35, steps)
Tc_values = np.linspace(0, 3, steps)
v0, x0 = 0.1, 0.9
tspan = np.linspace(delay, 200, 1000)

    
def classify_outcome(v_final, x_final, v_avg, x_avg):
    if v_avg > 0.1 and v_avg <0.9 and x_avg > 0.1 and x_avg < 0.9:
        return 2 + (v_final + x_final)/1.8
    else:
        return v_final + x_final


def create_heatmap_temp(y_vals, x_vals):
    outcome_grid = np.empty((len(y_vals), len(x_vals)))
    for i, Tc in enumerate(y_vals):
        for j, Tv in enumerate(x_vals):
            hist = history_function(v0, x0, Tv)
            sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
            column_0_vals = [sol_val[0] for sol_val in sol]
            column_1_vals = [sol_val[1] for sol_val in sol]
            avg_column_0 = sum(column_0_vals) / len(column_0_vals)
            avg_column_1 = sum(column_1_vals) / len(column_1_vals)
            v_final_val, x_final_val = sol[-1]
            outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val, avg_column_0, avg_column_1)
    return outcome_grid
      

# sns.heatmap(data=create_heatmap_temp(Tc_values, Tv_values), cmap="viridis")
# plt.title('v0=0.1, x0=0.9')
# plt.show()

def create_heatmap_conditional(y_vals, x_vals, type='temp', v0=0.1, x0=0.9):
    outcome_grid = np.empty((len(y_vals), len(x_vals)))
    if type == 'warm': # Option for warming cost parameterization
        Tv = 30
        Tc = 2
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                hist = history_function(v0,x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, y, x))
                column_0_vals = [sol_val[0] for sol_val in sol]
                column_1_vals = [sol_val[1] for sol_val in sol]
                avg_column_0 = sum(column_0_vals) / len(column_0_vals)
                avg_column_1 = sum(column_1_vals) / len(column_1_vals)
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val, avg_column_0, avg_column_1)
        return outcome_grid
    elif type == 'social': # Option for social dynamics parameterization
        Tv = 30
        Tc = 2
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                hist = history_function(v0,x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc, 5, 3, y, x))
                column_0_vals = [sol_val[0] for sol_val in sol]
                column_1_vals = [sol_val[1] for sol_val in sol]
                avg_column_0 = sum(column_0_vals) / len(column_0_vals)
                avg_column_1 = sum(column_1_vals) / len(column_1_vals)
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val, avg_column_0, avg_column_1)
        return outcome_grid
    else:
        for i, Tc in enumerate(y_vals): # Temperature parameterization (Fu Paper Replication)
            for j, Tv in enumerate(x_vals):
                hist = history_function(v0, x0, Tv)
                sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
                column_0_vals = [sol_val[0] for sol_val in sol]
                column_1_vals = [sol_val[1] for sol_val in sol]
                avg_column_0 = sum(column_0_vals) / len(column_0_vals)
                avg_column_1 = sum(column_1_vals) / len(column_1_vals)
                v_final_val, x_final_val = sol[-1]
                outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val, avg_column_0, avg_column_1)
        return outcome_grid

# sns.heatmap(data=create_heatmap_conditional(Tc_values, Tv_values, type='temp', v0=0.2, x0=0.8), cmap='YlGnBu_r')
# plt.title(f"v0={v0}, x0={x0}")
# plt.xticks(np.arange(0,steps,step=1), labels = np.round(Tv_values, 1))
# plt.yticks(np.arange(0,steps, step=1), labels = np.round(Tc_values, 1))
# plt.xlabel('Tv')
# plt.ylabel('Tc')
# plt.show()

fmax_vals = np.linspace(0,10,steps)
w_vals = np.linspace(0,10,steps)


# sns.heatmap(data=create_heatmap_conditional(fmax_vals, w_vals, type='warm'), cmap='YlGnBu_r')
# plt.xticks(np.arange(0,steps,step=1), labels = np.round(fmax_vals, 1))
# plt.yticks(np.arange(0,steps, step=1), labels = np.round(w_vals, 1))
# plt.xlabel('fmax')
# plt.ylabel('w')
# plt.show()


def reverse_parameter_sweep(Tv, Tc):
    v_vals = np.linspace(0, 1, 25)
    x_vals = np.linspace(0, 1, 25)
    outcome_grid = np.empty((len(v_vals), len(x_vals)))
    for i, v in enumerate(v_vals):
        for j, x in enumerate(x_vals):
            hist = history_function(v, x, Tv)
            sol = ddeint(model, hist, tspan, fargs=(delay, Tv, Tc))
            column_0_vals = [sol_val[0] for sol_val in sol]
            column_1_vals = [sol_val[1] for sol_val in sol]
            avg_column_0 = sum(column_0_vals) / len(column_0_vals)
            avg_column_1 = sum(column_1_vals) / len(column_1_vals)
            v_final_val, x_final_val = sol[-1]
            outcome_grid[i,j] = classify_outcome(v_final_val, x_final_val, avg_column_0, avg_column_1)
    return outcome_grid


# sns.heatmap(data=reverse_parameter_sweep(30, 2))
# plt.show()




## HETEROGENEOUS POPULATION MODEL
def h_classify(v_final, xp_final, xr_final, avg_v, avg_xp, avg_xr, rho):
    x_avg = rho * avg_xr + (1-rho) * avg_xp
    x_fin = rho * xr_final+ (1-rho) * xp_final
    if avg_v > 0.1 and avg_v <0.9 and x_avg > 0.1 and x_avg < 0.9:
        return 2 + (v_final + x_fin)/1.8
    else:
        return v_final + x_fin
    
def veg_classify(v_final):
    return v_final

def heterogeneous_param_sweep(Tv, Tc, v0=0.2):
    xr_vals = np.linspace(0,1,25)
    xp_vals = np.linspace(0,1,25)
    outcome_grid = np.empty((len(xr_vals), len(xp_vals)))
    for i, xr in enumerate(xr_vals):
        for j, xp in enumerate(xp_vals):
            hist = doub_history_function(v0, Tv, xr, xp)
            sol = ddeint(heterogeneous_pop_model, hist, tspan, fargs=(delay, Tv, Tc))
            v_raw = [sol_val[0] for sol_val in sol]
            xp_raw = [sol_val[1] for sol_val in sol]
            xr_raw = [sol_val[2] for sol_val in sol]
            avg_v = sum(v_raw) / len(v_raw)
            avg_xp = sum(xp_raw) / len(xp_raw)
            avg_xr = sum(xr_raw) / len(xr_raw)
            v_final_val, xp_final_val, xr_final_val = sol[-1]
            #outcome_grid[i,j] = h_classify(v_final_val, xp_final_val, xr_final_val, avg_v, avg_xp, avg_xr, 0.2)
            outcome_grid[i,j] = veg_classify(v_final_val)
    return outcome_grid

proportion_vals = np.linspace(0,1,25)
sns.heatmap(data=heterogeneous_param_sweep(30, 2), cmap='YlGnBu_r')
plt.xticks(np.arange(0,25,step=1), labels = np.round(proportion_vals, 2))
plt.yticks(np.arange(0,25, step=1), labels = np.round(proportion_vals, 2))
plt.title("V0=0.2, Tv=30, Tc=2")
plt.xlabel('xp0')
plt.ylabel('xr0')
plt.show()


def heterogeneous_heat_sweep(Tc_vals, Tv_vals, v0, xr0, xp0):
    outcome_grid = np.empty((len(Tc_vals), len(Tv_vals)))
    for i, Tc in enumerate(Tc_vals):
        for j, Tv in enumerate(Tv_vals):
            hist = doub_history_function(v0, Tv, xr0, xp0)
            sol = ddeint(heterogeneous_pop_model, hist, tspan, fargs=(delay, Tv, Tc))
            v_raw = [sol_val[0] for sol_val in sol]
            xp_raw = [sol_val[1] for sol_val in sol]
            xr_raw = [sol_val[2] for sol_val in sol]
            avg_v = sum(v_raw) / len(v_raw)
            avg_xp = sum(xp_raw) / len(xp_raw)
            avg_xr = sum(xr_raw) / len(xr_raw)
            v_final_val, xp_final_val, xr_final_val = sol[-1]
            outcome_grid[i,j] = h_classify(v_final_val, xp_final_val, xr_final_val, avg_v, avg_xp, avg_xr, 0.2)
            outcome_grid[i,j] = veg_classify(v_final_val)
    return outcome_grid

sns.heatmap(data=heterogeneous_heat_sweep(Tc_values, Tv_values, 0.1, 0.9, 0.9), cmap='YlGnBu_r')
plt.xticks(np.arange(0,25,step=1), labels = np.round(Tv_values, 2))
plt.yticks(np.arange(0,25, step=1), labels = np.round(Tc_values, 2))
plt.title("V0=0.1, Xr0 = 0.9, Xp0 = 0.9")
plt.xlabel('Tv')
plt.ylabel('Tc')
plt.show()
