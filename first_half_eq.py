# Luke Wisniewski, Feb 22, 2025


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ddeint import ddeint

gamma = 0.2  # Disturbance rate
g0= 2
beta=10
Topt = 28
a=5


def uncoupledgrowthrate(T):
    return g0 * (1 - ((T - Topt) / beta) ** 2)


def temperature(v, Tv):
    return Tv + (1 - v) * a


# Figure 1 (subfigures 1 and 2)

fig_1_g_values = [0.3, 1.2]  # Two different values of g

# Define the range of v values (0 to 1)
v_values = np.linspace(0, 1, 100)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, g in enumerate(fig_1_g_values):
    ax = axes[i]  # Select the subplot
    
    # Compute function values
    gv_1_v = g * v_values * (1 - v_values)  # gv(1-v)
    gamma_v = gamma * v_values              # gamma*v

    # Plot both functions
    ax.plot(v_values, gv_1_v, label=r'$g v (1 - v)$', color='b')
    ax.plot(v_values, gamma_v, label=r'$\gamma v$', color='r', linestyle='dashed')

    # Highlight intersection (equilibrium)
    equilibrium_v = 1- gamma / g
    if 0 <= equilibrium_v <= 1:
        ax.scatter(equilibrium_v, g * equilibrium_v * (1 - equilibrium_v), color='black', zorder=3, label='Equilibrium')

    # Labels and title
    ax.set_xlabel('v')
    ax.set_title(f'g = {g}')
    ax.legend()
    ax.grid(True)

# Set shared y-axis label
axes[0].set_ylabel('Function Value')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Figure 1c
# Define the range of g values (avoid g=0 to prevent division by zero)
c_g_values = np.linspace(0.2, 4, 400)  # g from 0.1 to 2

# Compute v values
c_v_values = 1 - gamma / c_g_values

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(c_g_values, c_v_values, color='b', label=r'$v = 1 - \frac{\gamma}{g}$')

# Labels and title
plt.xlabel('g')
plt.ylabel('v')
plt.title('$v = 1 - \gamma/g$ as g changes')
plt.axhline(0, color='gray', linestyle='dashed')  # Horizontal reference line
plt.legend()
plt.grid(True)

# Show plot
plt.show()

# Figure 2
def fig2eq(v, Tv):
    return (1 - v) * (1 - 0.01 * (Tv - 23 - 5 * v) ** 2) - 0.1

fig2_Tv_values = np.linspace(13, 36, 100)  # 100 points from 13 to 36
fig2_v_solutions = []

for Tv in fig2_Tv_values:
    # Solve numerically for v
    v_guess = 0.5  # Initial guess
    fig2_v_solution = fsolve(fig2eq, v_guess, args=(Tv,))
    
    # Ensure we capture only valid solutions (between 0 and 1)
    valid_v = [v for v in fig2_v_solution if 0 <= v <= 1]
    
    # Store solutions along with the trivial v=0
    fig2_v_solutions.append([0] + valid_v)  # Include v=0 always

# Convert to an array for plotting
fig2_Tv_solutions = []
fig2_v_values = []

for i, Tv in enumerate(fig2_Tv_values):
    for v in fig2_v_solutions[i]:
        fig2_Tv_solutions.append(Tv)
        fig2_v_values.append(v)

# Plot results of figure 2
plt.figure(figsize=(8, 5))
plt.scatter(fig2_Tv_solutions, fig2_v_values, color='b', s=10, label="Solutions for v")
plt.xlabel("Tv")
plt.ylabel("v")
plt.title("Possible values of v as Tv increases from 13 to 36")
plt.legend()
plt.grid(True)
plt.show()


# Figure 3
def fig3_equation(v, Tv, x):
    return 2 * (1 - v) * (0.2 + 0.4 * x) * (1 - 0.01 * (Tv - 23 - 5 * v)**2) - 0.2

# Define the range of Tv values
Tv_values = np.linspace(13, 35, 200)

# Function to compute v solutions for a given x
def compute_v_solutions(x):
    fig3_v_solutions = []
    for Tv in Tv_values:
        v_guess = 0.5  # Initial guess
        fig3_v_solution = fsolve(fig3_equation, v_guess, args=(Tv, x))
        
        # Ensure valid solutions within the range [0,1]
        valid_v = [v for v in fig3_v_solution if 0 <= v <= 1]
        fig3_v_solutions.append(valid_v[0] if valid_v else np.nan)  # Use NaN if no valid solution
    return np.array(fig3_v_solutions)

# Compute v solutions for x=0 and x=1
v_x0 = compute_v_solutions(x=0)
v_x1 = compute_v_solutions(x=1)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for x=0
axes[0].plot(Tv_values, v_x0, color='b', label='x = 0')
axes[0].set_title("x = 0")
axes[0].set_xlabel("Tv")
axes[0].set_ylabel("v")
axes[0].set_ylim(0, 0.5)
axes[0].grid(True)
axes[0].legend()

# Plot for x=1
axes[1].plot(Tv_values, v_x1, color='r', label='x = 1')
axes[1].set_title("x = 1")
axes[1].set_xlabel("Tv")
axes[1].set_ylim(0, 1.0)
axes[1].grid(True)
axes[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# Figure 4
x_values = np.linspace(0, 1, 100)

# List of Tv values to analyze
fig4_Tv_list = [15, 25, 30, 31, 32, 33]

# Function to compute v solutions for a given Tv across x values
def compute_v_solutions(Tv):
    fig4_v_solutions = []
    for x in x_values:
        v_guess = 0.5  # Initial guess
        v_solution = fsolve(fig3_equation, v_guess, args=(Tv, x))
        
        # Ensure valid solutions within the range [0,1]
        valid_v = [v for v in v_solution if 0 <= v <= 1]
        fig4_v_solutions.append(valid_v[0] if valid_v else np.nan)  # Use NaN if no valid solution
    return np.array(fig4_v_solutions)

# Create subplots (one for each Tv value)
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

# Loop through each Tv value and plot the corresponding graph
for i, Tv in enumerate(fig4_Tv_list):
    row, col = divmod(i, 3)  # Determine subplot position

    fig4_v_solutions = compute_v_solutions(Tv)  # Solve for v

    # Plot v vs x for the given Tv
    axes[row, col].plot(x_values, fig4_v_solutions, color='b', label=f'Tv = {Tv}')
    axes[row, col].set_title(f'Tv = {Tv}')
    axes[row, col].set_xlabel("x")
    axes[row, col].set_ylabel("v")
    axes[row, col].grid(True)
    axes[row, col].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


#Figure 5
def x_equilibrium(Tc):
    return 1 - 2.5 / (1 + np.exp(3 * Tc))

# Define the range of Tc values
fig5_Tc_values = np.linspace(0, 4, 100)

# Compute x* values
fig5_x_star_values = x_equilibrium(fig5_Tc_values)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(fig5_Tc_values, fig5_x_star_values, color='b', label=r'$x^*(T_c) = 1 - \frac{2.5}{1 + e^{3T_c}}$')

# Add labels and title
plt.xlabel(r'Critical Warming $T_c$')
plt.ylabel(r'Equilibrium Mitigation Level $x^*$')
plt.title(r'Equilibrium Mitigation Level as a Function of $T_c$')

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
