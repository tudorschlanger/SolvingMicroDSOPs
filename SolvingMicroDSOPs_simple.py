# %%
from copy import copy
import time

# First import all necessary libraries.
import numpy as np  # Import Python's numeric library as an easier-to-use abbreviation, "np"
import matplotlib.pyplot as plt  # Import Python's plotting library as the abbreviation "plt"
import scipy.stats as stats  # Import Scientific Python's statistics library
from numpy import log, exp  # for cleaner-looking code below
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq  # Import the brentq root-finder
from scipy.optimize import minimize

# Custom made functions
from Code.Python.gothic_class import Gothic 
from Code.Python.resources import (
    Utility,
    DiscreteApproximation,
    DiscreteApproximationTwoIndependentDistribs,
)

## notice that resources is stored in the subfolder Code/Python.
## It can be imported only if there is an __init__.py file in that folder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
## the userwarning is surpressed for a cleaner presentation.

# %%
### for comparison purposes
## import some modules from HARK libraries
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
)

# %% [markdown]
# ## 0. Define Parameters, Grids, and the Utility Function 
# Set up general parameters, as well as the first two major class instances: the utility function and the discrete approximation.

# %%
# Set up general parameters:

rho = 2.0  ### CRRA coefficient
beta = 0.96  ### discount factor
gamma = Gamma = np.array([1.0])  # permanent income growth factor (later, we can impose an expected growth path based on the empirical lifetime income path)
# A one-element "time series" array
# (the array structure needed for gothic class below)
R = 1.02  ## Risk free interest factor
T = 10   ## Number of periods in agent's life   

# Define utility:
u = Utility(rho)

# Construct a discrete distribution of the log normal distribution of income (\theta)
theta_sigma = 0.5
theta_mu    = -0.5 * (theta_sigma**2)
theta_dist  = stats.lognorm(
                scale = np.exp(theta_mu), # mean
                s = theta_sigma ) # std dev 
theta_grid_N = 10 # grid points to approximate distribution

theta = DiscreteApproximation(
    theta_grid_N,
    theta_dist.cdf,
    theta_dist.pdf,
    theta_dist.ppf
)

# Self-imposed lower bounds(for period T)
# the minimum asset (the maximum debt) is the discounted value of lowest possible transitory income shock
## the so called "natural borrowing constraint"

self_a_min = -min(theta.X) * gamma[0] / R  # self-imposed minimum a

### set the a and m grid 
a_min, a_max, a_size = self_a_min, 5, 10  # cash on hand 
m_min, m_max, m_size = 0.5, 4.0, 5       # why do we need these bounds? 

a_grid = np.linspace(a_min, a_max, a_size)
m_grid = np.linspace(m_min, m_max, m_size)
m_grid_fine = np.linspace(m_min, m_max, 100)


# %%
# np.array([1.0])
print(Gamma)
print(theta.X)
print(a_grid)

# %% [markdown]
# # Plot figure 2 using the Gothic object. 
# Rewrite and simplify the code provided to plot the first period consumption function, c(m), as done in Figure 15

# %%
## Solve the consumption-saving choice problem in the second-to-the-last period

# create a gothic instance
gothic = Gothic(
    u, beta, rho, gamma, R, theta
)

#################################
### the last period, i.e. T-1
###############################

# Note: the last period decision is the decision made for consumption in T-1 knowing that what is left-over to period T will be fully consumed. So the decision in period T-1 effectively pins down consumption in period T too. 

# Initialize the vector function 
cVec = []  # Start with an empty list. Lists can hold any Python object.
vVec = []  # Empty list for values
mVec = []
# fine grid of m

for m in m_grid:
    print(">>>> m = {}".format(m))
    # Find c(m) such that u'(c) = \beta * R * E[v'(m-c(m)))] } for each m
    # Compute v(a) = v(m-c(m)) for each m  
    nvalue = lambda c: abs( c**(-rho) - gothic.VP_Tminus1(m - c)) # note that V takes in a, not m! 
     
    # define upper bound of consumption to be current cash on hand plus smallest income shock in the discretized distribution support, i.e. the natural borrowing limit
    c_max = m + gamma[0] * theta.X[0] / R 

    res = minimize(
        nvalue,
        np.array([0.3 * m]),  ## an initial guess of the c
        method="trust-constr",  ## trust-constr turns out a good constrained optimization algorithm
        bounds=(
            (1e-12, 0.999 * c_max),
        ),  ## consumption has to be between 0.0 and c_max
        options={"gtol": 1e-12, "disp": False},
    )

    c = res["x"][0]  # maximizer of the value
    v = -res["fun"]  ## value
    cVec.append(c)
    vVec.append(v)


# %%
mVec = m_grid 
aVec = np.array(mVec) - np.array(cVec) # Note that this does not take into account the natural borrowing limit 
sVec = aVec / np.array(mVec) * 100
sVec 

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)

plt.subplot(1, 2, 1)  # row 1, column 2, count 1
plt.plot(mVec, cVec, 'b', linewidth=5, linestyle='-')
plt.title('Consumption')
plt.xlabel(r"$m_{T-1}$")    
plt.ylabel(r"${c}_{T-1}(m_{T-1})$")

plt.subplot(1, 2, 2)  # row 1, column 2, count 1
plt.plot(mVec, sVec, 'g', linewidth=5, linestyle='--')
plt.title('Saving rate')
plt.xlabel(r"$m_{T-1}$")    
plt.ylabel(r"$a_{T-1}/m_{T-1}(\%)$")

fig.suptitle('Figure 2')
plt.tight_layout()
plt.show()


# %% [markdown]
# # Plot figure 15 using backward iteration code
# 

# %%
#############################################
### Interpolation of the c(m) function above
#############################################

# Note: This step is necessary because the recursion is based off the Euler equation:
# u'(c_t(m)) = \beta * G^(1-\rho) * R * E_t[u'( c_{t+1}( (m - c_t)*R + \theta ) )]
# The issue is that the argument "m - c_t)*R + \theta" is a continuous value and we have a discrete policy function c(m) from above
# So we need to make it continuous by interpolating the discrete values. One way to do this is to use "spline" interpolation where each neighboring pair of points is fit with a function, often a cubic. 

# Set up the interpolation:
cFunc = InterpolatedUnivariateSpline(mVec, cVec, k=3)  

## save the grid in a dictionary
mGrid_life = {T - 1: mVec}
cGrid_life = {T - 1: cVec}

## save the interpolated function in a dictionary
cFunc_life = {T - 1: cFunc}

# %%
# Example of interpolated consumption function using cubic spline (k=3) 
# Note: since the policy function is almost linear, the interpolation is very good!  
cVec_spline = cFunc(m_grid_fine)
plt.plot(m_grid, cVec, 'b', linewidth=5, linestyle='-')
plt.plot(m_grid_fine, cVec_spline, 'r', linewidth=2, linestyle='--')

# %%
#######################################
### backward to earlier periods t < T-1
#######################################

for t in range(T - 2, 0, -1):
    print(">>>> t = {}".format(t))
    cVec = []

    for m in mVec:

        nvalue = lambda c: abs( c**(-rho) - gothic.VP_t(m - c, c_prime=cFunc) ) 
     
        # define upper bound of consumption to be current cash on hand plus smallest income shock in the discretized distribution support, i.e. the natural borrowing limit
        c_max = m + gamma[0] * theta.X[0] / R 

        res = minimize(
            nvalue,
            np.array([0.3 * m]),  ## an initial guess of the c
            method="trust-constr",  ## trust-constr turns out a good constrained optimization algorithm
            bounds=(
                (1e-12, 0.999 * c_max),
            ),  ## consumption has to be between 0.0 and c_max
            options={"gtol": 1e-12, "disp": False},
        )

        c = res["x"][0]  # maximizer of the value
        # c = np.minimum(c, m) # Ensures that agent cannot consume more than cash on hand; so no borrowing 
        cVec.append(c)    

    # Update the consumption function 
    cFunc = InterpolatedUnivariateSpline(mVec, cVec, k=3)

    # Save the policy grid in a dictionary
    mGrid_life[t] = mVec
    cGrid_life[t] = cVec

    print("mVec: ", mVec)
    print("cVec: ", cVec)

    # Save interpolated function for next iteration 
    cFunc_life[t] = cFunc



# %%
aGrid_life = {}
sGrid_life = {}

# Construct a dictionary for the saving rates 
for t in range(T-1, 0, -1):
    print(">>>> t = {}".format(t))

    aVec = np.array(mVec) - np.array(cGrid_life[t])
    sVec = aVec / np.array(mVec) * 100

    aGrid_life[t] = aVec
    sGrid_life[t] = sVec

#######################################
### Plot Figure 15 
#######################################

fig, (ax1, ax2) = plt.subplots(1, 2)

plt.subplot(1, 2, 1)  # row 1, column 2, count 1
plt.plot(mVec, cGrid_life[1], 'b', linewidth=5, linestyle='-')
plt.title('Consumption')
plt.xlabel(r"$m_{T-1}$")    
plt.ylabel(r"${c}_{T-1}(m_{T-1})$")

plt.subplot(1, 2, 2)  # row 1, column 2, count 1
plt.plot(mVec, sGrid_life[1], 'g', linewidth=5, linestyle='--')
plt.title('Saving rate')
plt.xlabel(r"$m_{T-1}$")    
plt.ylabel(r"$a_{T-1}/m_{T-1}(\%)$")

fig.suptitle('Figure 15')
plt.tight_layout()
plt.show()

# %%
# Compare the first to the last period consumption and saving rates 

plt.plot(mVec, sGrid_life[T-1], 'g', linewidth=5, linestyle='-', label="Period {}".format(T-1))
plt.plot(mVec, sGrid_life[1], 'g', linewidth=5, linestyle='--', label="Period 1")
plt.title('Saving rates across periods')
plt.xlabel(r"$m_{T-1}$")    
plt.ylabel(r"$a_{T-1}/m_{T-1}(\%)$")
plt.legend(loc="upper left")

# %%
# Lifecyle of saving rates for individual with fixed cash on hand
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

color = ['tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple'
            ]

years = range(1, T-1, 1)
mlist = [0, 1, 2, 3, 4]
for m in mlist:
    slife = []
    for t in years:
        slife.append(sGrid_life[t][m])
    plt.plot(years, slife, color[m], linewidth=5, linestyle=linestyle_tuple[m][1], label="m = {}".format(m))

plt.title(r'Saving rates across lifecycle, given wealth level $m$')
plt.xlabel("Years")    
plt.ylabel(r"$a_t/m_t(\%)$")
plt.legend(loc="best")
plt.savefig('srate_lifecycle.png', dpi = 300)

# %%