import numpy as np
import os
import sys
import hydra
import torch

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dotmap import DotMap
import logging

# adapeted from https://scipython.com/blog/the-lorenz-attractor/
log = logging.getLogger(__name__)


def sim_lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


@hydra.main(config_path='conf/lorenz.yaml')
def lorenz(cfg):
    # Lorenz paramters and initial conditions
    sigma, beta, rho = cfg.lorenz.sigma, cfg.lorenz.beta, cfg.lorenz.rho
    u0, v0, w0 = cfg.lorenz.ex.u0, cfg.lorenz.ex.v0, cfg.lorenz.ex.w0

    # Maximum time point and total number of time points
    tmax, n = cfg.lorenz.tmax, cfg.lorenz.n

    # Integrate the Lorenz equations on the time grid t
    t = np.linspace(0, tmax, n)
    f = odeint(sim_lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
    x, y, z = f.T

    num_traj = cfg.experiment.num_traj
    if cfg.collect_data:
        data_X = np.zeros((1, 3))
        data_Seq = []

        new_init = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(num_traj, 3))

        for row in new_init:
            u, v, w = row  # row[0], row[1], row[2]
            f = odeint(sim_lorenz, (u, v, w), t, args=(sigma, beta, rho))
            x, y, z = f.T
            l = DotMap()
            l.states = f
            data_X = np.concatenate((data_X, f))
            data_Seq.append(l)


        X = data_X[1:, :]
        dX = data_X[1:, :] - data_X[:-1, :]
        if cfg.save_data:
            log.info("Saving new default data")
            torch.save((X, dX), hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'step' + cfg.data_dir)
            torch.save((data_Seq), hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'traj' + cfg.data_dir)
            log.info(f"Saved trajectories to {cfg.data_dir}")
    else:
        X, dX = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'step' + cfg.data_dir)
        data_Seq = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'traj' + cfg.data_dir)

    # Analysis
    from mbrl_resources import train_network, Net
    from reacher_pd import create_dataset_t_only

    dataset_1s = [X, dX]
    dataset_ct = create_dataset_t_only(data_Seq)

    p = DotMap()
    p.opt.n_epochs = 20
    p.opt.batch_size = 100
    p.learning_rate = 0.001

    if cfg.train_models:
        model_1s, train_log1 = train_network(dataset_1s, Net(structure=[3, 100, 100, 3]), parameters=p)
        model_ct, train_log2 = train_network(dataset_ct, Net(structure=[4, 100, 100, 3]), parameters=p)

        if cfg.save_models:
            log.info("Saving new default models")
            torch.save((model_1s, train_log1), hydra.utils.get_original_cwd() + '/models/lorenz/' + 'step' + cfg.model_dir)
            torch.save((model_ct, train_log2), hydra.utils.get_original_cwd() + '/models/lorenz/' + 'traj' + cfg.model_dir)

    else:
        model_1s, train_log1 = torch.load(hydra.utils.get_original_cwd() + '/models/lorenz/' + 'step' + cfg.model_dir)
        model_ct, train_log2 = torch.load(hydra.utils.get_original_cwd() + '/models/lorenz/' + 'traj' + cfg.model_dir)

        # new_init = np.random.uniform(low=[-25, -25, -25], high=[25, 25, 25], size=(1, 3))
    traj = np.random.randint(num_traj)
    new_init = data_Seq[traj].states[0]
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

    if cfg.plot:
        import plotly.graph_objects as go

        fig = go.Figure()

        for dat in data_Seq:
            x, y, z = dat.states[:, 0], dat.states[:, 1], dat.states[:, 2]
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

        # TODO make this a manual plot (not Scatter3d to remove background)
        fig.update_layout(
            width=1500,
            height=800,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[-100, 100], ),
                yaxis=dict(nticks=4, range=[-100, 100], ),
                zaxis=dict(nticks=4, range=[-100, 100], ),
                aspectratio=dict(x=1, y=1, z=0.7),
                    aspectmode='manual'
            ),
            margin=dict(r=10, l=10, b=10, t=10),
            # scene=dict(
            #     camera=dict(
            #         up=dict(
            #             x=0,
            #             y=0,
            #             z=1
            #         ),
            #         eye=dict(
            #             x=0,
            #             y=1.0707,
            #             z=1,
            #         )
            #     ),
            #     aspectratio=dict(x=1, y=1, z=0.7),
            #     aspectmode='manual'
            # ),
            plot_bgcolor='white',
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.show()
        fig.write_image(os.getcwd() + "/lorenz.pdf")

def plot_learning():
    raise NotImplementedError("TODO")

if __name__ == '__main__':
    sys.exit(lorenz())
