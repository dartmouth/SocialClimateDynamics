import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Define system parameters
delay = 10  # Delay in years

# Define two cases as in Figure 8
cases = [
    {"Tv": 31.5, "Tc": 2.0, "label": "Tv = 31.5°C, Tc = 2.0°C"},
    {"Tv": 32.5, "Tc": 2.5, "label": "Tv = 32.5°C, Tc = 2.5°C"}
]

# Define the DDE system
def dde_system(Y, t, history, Tv, Tc):
    v, x = Y(t)  # Current values
    v_delayed = history(t - delay)[0] if t - delay >= 10 else history(10)[0]  # Handle delay from t=10

    # Define dv/dt
    dvdt = 2 * (1-0.01*(Tv-23-5*v)**2)*(0.2+0.4*x)*v*(1-v)-0.2*v
    #dvdt = 2 * (1 - v) * (0.2 + 0.4 * x) * (1 - 0.01 * (Tv - 23 - 5 * v)**2) * v - 0.2 * v
    
    # Define dx/dt with delay term
    dxdt = x * (1 - x) * (2 * x - 2 + 5 / (1 + np.exp(-3 * (-7.5 * v + 7.5 * v_delayed - Tc))))

    return np.array([dvdt, dxdt])

# Define initial history function (valid for t < 10)
def initial_history(t):
    return np.array([0.1, 0.9])  # Initial conditions v(0)=0.1, x(0)=0.9

# Time range for simulation (starting from year 10)
t_values = np.linspace(10, 200, 1000)  # Solve from year 10 to 200

# Create subplots for the two cases
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i, case in enumerate(cases):
    Tv, Tc, label = case["Tv"], case["Tc"], case["label"]
    
    # Solve the DDE system (delayed simulation starting from t=10)
    solution = ddeint(lambda Y, t: dde_system(Y, t, initial_history, Tv, Tc), 
                      initial_history, t_values)

    # Extract v and x solutions
    v_values, x_values = solution.T

    # Plot the results
    axes[i].plot(t_values, v_values, label="Vegetation Coverage (v)", color='g')
    axes[i].plot(t_values, x_values, label="Mitigation Level (x)", color='b')

    # Set y-axis limits to match figure 8
    axes[i].set_ylim(0, 1)  # Full range 0 to 1 for both variables
    axes[i].set_yticks(np.arange(0, 1.1, 0.2))  # Tick marks for readability

    # Labels and title
    axes[i].set_title(f"Evolution of v and x for {label}")
    axes[i].set_ylabel("State Variables")
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Time (Years)")

# Adjust layout and show plot
plt.tight_layout()
plt.show()
