#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint

delay = 10

# Define the delay differential equations
def model(Y, t, Tv, Tc, d):
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
    dvdt = v*(2*(1-v)*(1-((Tv-a*(1-v)-Topt)/beta)**2)-gamma)
    return dvdt


def history_function(v0, x0, Tv):
    time = np.linspace(0,10,10000)
    v=odeint(uncoupled_model, v0, time,args=(gamma, Tv))
    v = v.flatten()
    x0_array = np.full_like(v, x0)
    output = np.column_stack((v, x0_array))  # Combine v and x0 into pairs
    def history(t):
        idx = np.searchsorted(time, t)
        if idx == 0:
            return output[0]
        else:
            print("Time=", time[idx-1], output[idx-1])
            return output[idx - 1]
    return history


# Time grid
tspan = np.linspace(delay, 200, 10000)


# First solution
sol1 = ddeint(model, history_function(0.1,0.9,31.5), tspan, fargs=(delay, 31.5, 2))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(tspan, sol1[:, 0], label="Vegetation")
plt.plot(tspan, sol1[:, 1], label="Mitigation")
plt.xlabel("Time (years)")
plt.ylabel("Proportions")
plt.title("V0=0.1, X0=0.9, Tv=31.5, Tc=2")
plt.legend()
plt.grid(True)
plt.show()

sol2 = ddeint(model, history_function(0.1,0.9,32.5), tspan, fargs=(delay, 32.5, 2.5))
plt.figure(figsize=(10, 6))
plt.plot(tspan, sol2[:, 0], label="Vegetation")
plt.plot(tspan, sol2[:, 1], label="Mitigation")
plt.xlabel("Time (years)")
plt.ylabel("Proportions")
plt.title("V0=0.1, X0=0.9, Tv=32.5, Tc=2.5")
plt.legend()
plt.grid(True)
plt.show()


# Figure 9


sol1_9 = ddeint(model, history_function(0.5,0.5,15), tspan, fargs=(delay, 15, 2))
plt.figure(figsize=(10, 6))
plt.plot(tspan, sol1_9[:, 0], label="Vegetation")
plt.plot(tspan, sol1_9[:, 1], label="Mitigation")
plt.xlabel("Time (years)")
plt.ylabel("Proportions")
plt.title("V0=0.5, X0=0.5, Tv=15, Tc=2")
plt.legend()
plt.grid(True)
plt.show()

# # def model(Y, t, d):
# #     x, y = Y(t)
# #     xd, yd = Y(t - d)
# #     return array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])


# # g = lambda t: array([1, 3])
# # tt = linspace(2, 30, 20000)

# # fig, ax = subplots(1, figsize=(4, 4))

# # for d in [0, 0.2]:
# #     print("Computing for d=%.02f" % d)
# #     yy = ddeint(model, g, tt, fargs=(d,))
# #     # WE PLOT X AGAINST Y
# #     ax.plot(yy[:, 0], yy[:, 1], lw=2, label="delay = %.01f" % d)
# #     plt.show()



# # # Define system parameters
# # delay = 10  # Delay in years

# g0= 2
# beta=10
# Topt = 28
# gamma=0.2

# def uncoupledgrowthrate(T):
#     return g0 * (1 - ((T - Topt) / beta) ** 2)

# # Define two cases as in Figure 8
# cases = [
#     {"Tv": 31.5, "Tc": 2.0, "label": "Tv = 31.5°C, Tc = 2.0°C"},
#     {"Tv": 32.5, "Tc": 2.5, "label": "Tv = 32.5°C, Tc = 2.5°C"}
# ]

# def model(Y, t, d, Tv, Tc):
#     v, x = Y(t)
#     v_delayed = Y(t-d)[0] if t-d >=10 else Y(0)[0]
#     dvdt = 2 * (1-0.01*(Tv-23-5*v)**2)*(0.2+0.4*x)*v*(1-v)-0.2*v
#     dxdt = x * (1 - x) * (2 * x - 2 + 5 / (1 + np.exp(-3 * (-7.5 * v + 7.5 * v_delayed - Tc))))
#     return np.array([dvdt, dxdt])

# g=np.array([0.1,0.9])

# tt = np.linspace(10,200,1000)

# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# solution1 = ddeint(model(d=10, Tv=31.5, Tc=2.0), g, tt)

# plot(tt, solution1)


# # # Define the DDE system
# # def dde_system(Y, t, history, Tv, Tc):
# #     v, x = Y(t)  # Current values
# #     v_delayed = history(t - delay)[0] if t - delay >= 10 else history(10)[0]  # Handle delay from t=10

# #     # Define dv/dt
# #     dvdt = 2 * (1-0.01*(Tv-23-5*v)**2)*(0.2+0.4*x)*v*(1-v)-0.2*v
# #     #dvdt = 2 * (1 - v) * (0.2 + 0.4 * x) * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * v - 0.2 * v
    
# #     # Define dx/dt with delay term
# #     dxdt = x * (1 - x) * (2 * x - 2 + 5 / (1 + np.exp(-3 * (-7.5 * v + 7.5 * v_delayed - Tc))))

# #     return np.array([dvdt, dxdt])

# # # Define initial history function (valid for t < 10)
# # def initial_history(t):
# #     return np.array([0.9, 0.9])  # Initial conditions v(0)=0.1, x(0)=0.9

# # # Time range for simulation (starting from year 10)
# # t_values = np.linspace(10, 200, 1000)  # Solve from year 10 to 200

# # # Create subplots for the two cases
# # fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # for i, case in enumerate(cases):
# #     Tv, Tc, label = case["Tv"], case["Tc"], case["label"]
    
# #     # Solve the DDE system (delayed simulation starting from t=10)
# #     solution = ddeint(dde_system(Y, t, initial_history, Tv, Tc), initial_history, t_values)

# #     # Extract v and x solutions
# #     v_values, x_values = solution.T

# #     # Plot the results
# #     axes[i].plot(t_values, v_values, label="Vegetation Coverage (v)", color='g')
# #     axes[i].plot(t_values, x_values, label="Mitigation Level (x)", color='b')

# #     # Set y-axis limits to match figure 8
# #     axes[i].set_ylim(0, 1)  # Full range 0 to 1 for both variables
# #     axes[i].set_yticks(np.arange(0, 1.1, 0.2))  # Tick marks for readability

# #     # Labels and title
# #     axes[i].set_title(f"Evolution of v and x for {label}")
# #     axes[i].set_ylabel("State Variables")
# #     axes[i].legend()
# #     axes[i].grid(True)

# # axes[-1].set_xlabel("Time (Years)")

# # # Adjust layout and show plot
# # plt.tight_layout()
# # plt.show()
