#Venv: fu_lab_work 

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import odeint
import seaborn as sns
from dde_figs import uncoupled_model
#from numba import njit

gamma = 0.2 # Disturbance rate
fmax = 5 # Maximum warming cost, between 4 and 6
alpha_r = 0.45 # Cost of mitigation for rich group, between 0.45 and 0.55
delta = 2 # Strength of social norms, between 0.5 and 1.5
kappa = 0.05 # Social learning rate, between 0.02 and 0.2
alpha_p0 = 0.9 # Cost of mitigation for poor group, between 0.9 and 1.1
xr0 = 0.9 # Initial mitigation rate in the rich subpopulation
xp0 = 0.9 # Initial mitigation rate in the poor subpopulation
ir0 = 4.5 # Initial income of rich group, between 4.5 and 5.5
ip0 = 3.85 # Initial income of poor group, between 3.15 and 3.85
rho = 0.25 # Size of rich group relative to poor group, between 0 and 0.5
cr = 0.4 # Maximum of income cost function for rich subpopulation, between 0.36 and 0.44
cp = 0.85 # Maximum of income cost function for poor subpopulation, between 0.765 and 0.935
kr = 1.1 # Nonlinearity of income impact function for rich subpopulation, between 0.9 and 1.1
kp = 1.35 # Nonlinearity of income impact function for poor subpopulation, between 1.35 and 1.65
h = 0 # Homophily parameter, between 0 and 1
d = 5 # Maximum cost of dissatisfaction, between 0 and 10
w = 3 # Nonlinearity of warming cost, between 2.7 and 3.3
omega = 3 # Nonlinearity of dissatisfaction cost, between 2.7 and 3.3
dc = 1.5 # Critical value of dissatisfaction cost, between 1.35 and 1.65
s = 1.5 # Effectiveness of mitigative action scaling, >= 1

time_delay = 10
tspan = np.linspace(time_delay, 200, 10000)

def heterogeneous_pop_model(Y, t, delay, Tv, Tc):
    v, xp, xr = Y(t)
    v_lag, xp_lag, xr_lag = Y(t-delay)
    gr = 1- (cr/(1+np.exp(-1*kr*(Tv-5*(1-v)-Tc))))
    gp = 1- (cp/(1+np.exp(-1*kp*(Tv-5*(1-v)-Tc))))
    yp = 1-xp
    yr = 1-xr
    ir = ir0 * max(gr, 0)
    ip = ip0 * max(gp, 0)
    x = s * rho * xr + (1-rho) * xp
    alpha_p = alpha_p0 + d/(1+np.exp(-1 * omega *((1-xr)/(1-xr0))*(ir/ip)*(ip0/ir0)-dc))
    eta_x = 0.2 + 0.4 * x

    dv_dt = 2 * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * eta_x * v * (1 - v) - 0.2 * v

    fTf = fmax/(1+np.exp(-1 * w * (-7.5 * v + 7.5 * v_lag - Tc)))
    ep_m = -1 * alpha_p + 0.5 * fTf + delta * (xp + (1-h)*xr)
    ep_n = -0.5 * fTf + delta *(yp+(1-h)*yr)
    er_m = -1 * alpha_r + 0.5 * fTf + delta * ((1-h)*xp+xr)
    er_n = -0.5 * fTf + delta * ((1-h)*yp+yr)

    dxp_dt = kappa * xp *(1-xp)*(ep_m - ep_n) + (1-h)*kappa*(xr*max(er_m-ep_n, 0)*(1-xp)-(1-xr)*max(er_n-ep_m, 0)*xp)

    dxr_dt = kappa * xr *(1-xr)*(er_m - er_n) + (1-h)*kappa*(xp*max(ep_m-er_n, 0)*(1-xr)-(1-xp)*max(ep_n-er_m, 0)*xr)

    return [dv_dt, dxp_dt, dxr_dt]



def doub_history_function(v0, Tv, init_xr, init_xp):
    time = np.linspace(0, time_delay,10000)
    v=odeint(uncoupled_model, v0, time,args=(gamma, Tv))
    v = v.flatten()
    xr0_array = np.full_like(v, init_xr)
    xp0_array = np.full_like(v, init_xp)
    output = np.column_stack((v, xr0_array, xp0_array))
    def history(t):
        idx = np.searchsorted(time, t)
        if idx == 0:
            return output[0]
        else:
            #print("Time=", time[idx-1], output[idx-1])
            return output[idx - 1]
    return history
if __name__ == "__main__":
    tv_vals = np.linspace(22, 35, 5)
    tc_vals = np.linspace(1, 3, 5)

    test = ddeint(heterogeneous_pop_model, doub_history_function(0.1, 25, xr0, xp0), tspan, fargs=(time_delay, 25, 1.5))
    plt.figure(figsize=(10, 6))
    plt.plot(tspan, test[:, 0], label="Vegetation")
    plt.plot(tspan, test[:, 1], label="Mitigation: Rich")
    plt.plot(tspan, test[:, 2], label = "Mitigation: Poor")
    plt.xlabel("Time (years)")
    plt.ylabel("Proportions")
    plt.ylim(0,1)
    plt.title(f"v0=0.1, Xr0={xr0}, Xp0={xp0}, Tv=25, Tc=1.5")
    plt.legend()
    plt.grid(True)
    plt.show()

    # for i, Tv in enumerate(tv_vals):
    #     for j, Tc in enumerate(tc_vals):
    #         test = ddeint(heterogeneous_pop_model, doub_history_function(0.3, Tv, xr0, xp0), tspan, fargs=(time_delay, Tv, Tc))

    #         plt.figure(figsize=(10, 6))
    #         plt.plot(tspan, test[:, 0], label="Vegetation")
    #         plt.plot(tspan, test[:, 1], label="Mitigation: Rich")
    #         plt.plot(tspan, test[:, 2], label = "Mitigation: Poor")
    #         plt.xlabel("Time (years)")
    #         plt.ylabel("Proportions")
    #         plt.ylim(0,1)
    #         plt.title(f"v0=0.3, Xr0={xr0}, Xp0={xp0}, Tv={Tv}, Tc={Tc}")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
