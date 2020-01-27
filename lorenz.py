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


import plotly.graph_objects as go

fig = go.Figure()

fig.update_layout(
    width=1500,
    height=800,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=0,
                y=1.0707,
                z=1,
            )
        ),
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode='manual'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


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
    s = 1
    c = np.linspace(0, 1, n)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        # color=(1, c[i], 0),
        marker=dict(
            size=1,
            color=np.arange(len(x)),
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ))
    #

# Analysis
from mbrl_resources import train_network, Net
from reacher_pd import create_dataset_t_only

X = data_X[1:, :]
dX = data_X[1:, :] - data_X[:-1, :]
dataset_1s = [X, dX]
dataset_ct = create_dataset_t_only(data_Seq)
p = DotMap()
p.opt.n_epochs = 20
p.opt.batch_size = 100
p.learning_rate = 0.001

model_1s, train_log1 = train_network(dataset_1s, Net(structure=[3, 100, 100, 3]), parameters=p)

model_ct, train_log2 = train_network(dataset_ct, Net(structure=[4, 100, 100, 3]), parameters=p)

new_init = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(1, 3))
predictions_1 = [new_init.squeeze()]
predictions_2 = [new_init.squeeze()]
for i in range(1, n):
    pred_t = model_ct.predict(np.hstack((predictions_1[-1], i)))
    pred_no_t = predictions_2[-1] + model_1s.predict(predictions_2[-1])
    predictions_1.append(pred_t.squeeze())
    predictions_2.append(pred_no_t.squeeze())

    # mse_t.append(np.square(groundtruth - pred_t).mean())
    # mse_no_t.append(np.square(groundtruth - pred_no_t).mean())
    current = pred_no_t.squeeze()

p_1 = np.stack(predictions_1)
p_2 = np.stack(predictions_2)

fig.add_trace(go.Scatter3d(
    x=p_1[:, 0], y=p_1[:, 1], z=p_1[:, 2],
    # color=(1, c[i], 0),
    marker=dict(
        size=0,
        # color=np.arange(len(x)),
        # colorscale='Viridis',
    ),
    line=dict(
        color='red',
        width=4
    ),
    name='Continuous Parameterization'
))

fig.add_trace(go.Scatter3d(
    x=p_2[:, 0], y=p_2[:, 1], z=p_2[:, 2],
    # color=(1, c[i], 0),
    marker=dict(
        size=0,
        # color=np.arange(len(x)),
        # colorscale='Viridis',
    ),
    line=dict(
        color='black',
        width=4
    ),
    name='One Step Parameterization'

))

fig.show()
