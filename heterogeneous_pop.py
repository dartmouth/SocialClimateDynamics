#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
import seaborn as sns
from dde_figs import uncoupled_model
#from numba import njit

gamma = 0.2

fmax = 5 # Between 4 and 6
alpha_r = 0.5 # Between 0.45 and 0.55
delta = 1 # Between 0.5 and 1.5
kappa = 0.05 # Between 0.02 and 0.2
alpha_p0 = 1 # Between 0.9 and 1.1
xr0 = 0.1
xp0 = 0.5
ir0 = 5 # Between 4.5 and 5.5
ip0 = 3.5 # Between 3.15 and 3.85
rho = 0.25 # Size of rich group, between 
cr = 0.4 # Between 
cp = 0.85
kr = 1
kp = 1.5
h = 0.5 # Homophily parameter, between 0 and 1
d = 5
w = 3
dc = 1.5

def heterogeneous_pop_model(Y, t, delay, Tv, Tc):
    v, xp, xr = Y(t)
    v_lag, xp_lag, xr_lag = Y(t-delay)
    gr = 1- (cr/(1+np.exp(-1*kr*(Tv-2))))
    gp = 1- (cp/(1+np.exp(-1*kp*(Tv-2))))
    ir = ir0 * max(gr, 0)
    ip = ip0 * max(gp, 0)
    x = rho * xr + (1-rho) * xp
    alpha_p = alpha_p0 + d/(1+np.exp(-1 * w*((1-xr)/(1-xr0))*(ir/ip)*(ip0/ir0)-dc))

    dv_dt = 2 * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * (0.2 + 0.4 * x) * v * (1 - v) - 0.2 * v

    fTf = 0.5 * fmax/(1+np.exp(-1 * w * (-7.5 * v + 7.5 * v_lag - Tc)))
    ep_m = -1 * alpha_p + fTf + delta * (xp + (1-h)*xr)
    ep_n = -1 * fTf + delta *((1-xp)+(1-h)*(1-xr))
    er_m = -1 * alpha_r + fTf + delta * ((1-h)*xp+xr)
    er_n = -1 * fTf + delta * ((1-h)*(1-xp)+(1-xr))

    dxp_dt = kappa * xp *(1-xp)*(ep_m - ep_n) + (1-h)*kappa*(xr*max(er_m-ep_n, 0)*(1-xp)-(1-xr)*max(er_n-ep_m, 0)*xp)

    dxr_dt = kappa * xr *(1-xr)*(er_m - er_n) + (1-h)*kappa*(xp*max(ep_m-er_n, 0)*(1-xr)-(1-xp)*max(ep_n-er_m, 0)*xr)

    return [dv_dt, dxp_dt, dxr_dt]

time_delay = 10

def doub_history_function(v0, xr0, xp0, Tv):
    time = np.linspace(0, time_delay,10000)
    v=odeint(uncoupled_model, v0, time,args=(gamma, Tv))
    v = v.flatten()
    xr0_array = np.full_like(v, xr0)
    xp0_array = np.full_like(v, xp0)
    output = np.column_stack((v, xr0_array, xp0_array))
    def history(t):
        idx = np.searchsorted(time, t)
        if idx == 0:
            return output[0]
        else:
            print("Time=", time[idx-1], output[idx-1])
            return output[idx - 1]
    return history

tspan = np.linspace(time_delay, 200, 10000)

test = ddeint(heterogeneous_pop_model, doub_history_function(0.5, 0.5, 0.5, 30), tspan, fargs = (time_delay, 30, 2))

plt.figure(figsize=(10, 6))
plt.plot(tspan, test[:, 0], label="Vegetation")
plt.plot(tspan, test[:, 1], label="Mitigation: Rich")
plt.plot(tspan, test[:, 2], label = "Mitigation: Poor")
plt.xlabel("Time (years)")
plt.ylabel("Proportions")
plt.ylim(0,1)
plt.title("V0=0.5, Xr0=0.5, Xp0=0.5, Tv=30, Tc=2")
plt.legend()
plt.grid(True)
plt.show()
