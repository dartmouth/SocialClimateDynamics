#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
from scipy.special import expit

delay = 10
fmax = 5 # Maximum warming cost
w = 3 # Non-linearity of warming cost
delta = 1 # strength of social norms
alpha = 1 # cost of mitigation

def model(Y, t, d, Tv, Tc, fmax=5, w=3, delta=1, alpha=1):
    v, x = Y(t)
    v_lag, x_lag = Y(t-d)
    #v = np.clip(v, 0, 1)
    #x = np.clip(x, 0, 1)
    #v_lag = np.clip(v_lag, 0, 1)
    #x_lag = np.clip(x_lag, 0, 1)
    #z = w * (7.5 * v - 7.5 * v_lag + Tc)
    #fTf = fmax*expit(z)
    fTf = fmax/(1+np.exp(-1 * w * (-7.5 * v + 7.5 * v_lag - Tc)))

    dv_dt = 2 * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * (0.2 + 0.4 * x) * v * (1 - v) - 0.2 * v

    dx_dt = x * (1 - x) * (delta * (2 * x - 1) - alpha + fTf)

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
    output = np.column_stack((v, x0_array))  # Combine v and x0 into pairs
    def history(t):
        idx = np.searchsorted(time, t)
        if idx == 0:
            return output[0]
        else:
            #print("Time=", time[idx-1], output[idx-1])
            return output[idx - 1]
    return history



if __name__ == "__main__":
    tspan = np.linspace(delay, 200, 10000)


# First solution
    sol1 = ddeint(model, history_function(0.1,0.9,31.5), tspan, fargs=(delay, 31.5, 2))

# Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol1[:, 0], label="Vegetation")
    plt.plot(tspan, sol1[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
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
    plt.ylim(0,1)
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
    plt.ylim(0,1)
    plt.title("V0=0.5, X0=0.5, Tv=15, Tc=2")
    plt.legend()
    plt.grid(True)
    plt.show()

    sol9b = ddeint(model, history_function(0.9,0.9,31.5), tspan, fargs=(delay, 31.5, 2))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol9b[:, 0], label="Vegetation")
    plt.plot(tspan, sol9b[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title("V0=0.9, X0=0.9, Tv=31.5, Tc=2")
    plt.legend()
    plt.grid(True)
    plt.show()

    sol10a = ddeint(model, history_function(0.1,0.1,34), tspan, fargs=(delay, 34, 0.5))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol10a[:, 0], label="Vegetation")
    plt.plot(tspan, sol10a[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title("V0=0.1, X0=0.1, Tv=34, Tc=0.5")
    plt.legend()
    plt.grid(True)
    plt.show()

    sol10b = ddeint(model, history_function(0.9,0.1,13), tspan, fargs=(delay, 13, 1.5))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol10b[:, 0], label="Vegetation")
    plt.plot(tspan, sol10b[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title("V0=0.9, X0=0.1, Tv=13, Tc=1.5")
    plt.legend()
    plt.grid(True)
    plt.show()

    sol11a = ddeint(model, history_function(0.1,0.1,25), tspan, fargs=(delay, 25, 1.5))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol11a[:, 0], label="Vegetation")
    plt.plot(tspan, sol11a[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title("V0=0.1, X0=0.1, Tv=25, Tc=1.5")
    plt.legend()
    plt.grid(True)
    plt.show()

    sol11b = ddeint(model, history_function(0.1,0.9,25), tspan, fargs=(delay, 25, 1))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, sol11b[:, 0], label="Vegetation")
    plt.plot(tspan, sol11b[:, 1], label="Mitigation")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title("V0=0.1, X0=0.9, Tv=25, Tc=1")
    plt.legend()
    plt.grid(True)
    plt.show()
