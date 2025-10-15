# Disrespect Others, Respect the Climate? Applying Social Dynamics with Inequality to Forest Climate Models
This repository contains the source code for the simulations and plots in the
 ``Disrespect Others, Respect the Climate? Applying Social Dynamics
 with Inequality to Forest Climate Models'' manuscript.
The preprint is available on arXiv.

## Layout
- requirements.txt: list of python requirements
- Plotly_plot.ipynb: Jupyter notebook that generates static and interactive phase portraits (Figure 1)
- outcomesovertime.ipynb: Jupyter notebook that generates mitigation proportion trajectories (Figures 3, 5)
- heatmaps.ipynb: Jupyter notebook that generates mitigation proportion trajectories (Figures 2, 4, 6-8)
- dde_figs.py: Script that shows mitigation proportion trajectories for multiple parameter sets
- heterogenous_pop_.py: Script that shows a particular mitigation proportion trajectory

## Running
To run the simulations, create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
and install the dependencies
```bash
python3 -m pip install -r requirements.txt
```
### Jupyter notebooks
The jupyter notebooks can be viewed by launching jupyter
```bash
jupyter notebook
```
and selecting the appropriate `.ipynb` file

### Scripts
The scripts can be run using python directly:
```bash
python3 <script name>.py
```
