import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dotmap import DotMap

# adapeted from https://scipython.com/blog/the-lorenz-attractor/


# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 2, 200


def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


# Integrate the Lorenz equations on the time grid t
t = np.linspace(0, tmax, n)
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the line multi-coloured by plotting it in segments of length s which
# change in colour across the whole time series.
s = 10
c = np.linspace(0, 1, n)
for i in range(0, n - s, s):
    ax.plot(x[i:i + s + 1], y[i:i + s + 1], z[i:i + s + 1], color=(1, c[i], 0), alpha=0.4)

# Remove all the axis clutter, leaving just the curve.
# ax.set_axis_off()
# plt.show()

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig2 = plt.figure()
ax2 = fig.gca(projection='3d')

data_X = np.zeros((1, 3))
data_Seq = []
new_init = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(10, 3))
for row in new_init:
    u, v, w = row  # row[0], row[1], row[2]
    f = odeint(lorenz, (u, v, w), t, args=(sigma, beta, rho))
    x, y, z = f.T
    l = DotMap()
    l.states = f
    data_X = np.concatenate((data_X, f))
    data_Seq.append(l)
    s = 10
    c = np.linspace(0, 1, n)
    for i in range(0, n - s, s):
        ax2.plot(x[i:i + s + 1], y[i:i + s + 1], z[i:i + s + 1], color=(1, c[i], 0), alpha=0.4)

# Remove all the axis clutter, leaving just the curve.
ax2.set_axis_off()
plt.show()

# Analysis
from mbrl_resources import train_network, Net
from reacher_pd import create_dataset_t_only

X = data_X[1:, :]
dX = data_X[1:, :] - data_X[:-1, :]
dataset_1s = [X, dX]
dataset_ct = create_dataset_t_only(data_Seq)

model_1s = train_network(dataset_1s, Net(structure=[3, 100, 100, 3]))
model_ct = train_network(dataset_ct, Net(structure=[4, 100, 100, 3]))

new_init = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(1, 3))
