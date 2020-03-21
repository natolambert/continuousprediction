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
from plot import plot_lorenz, plot_mse, plot_mse_err

# adapeted from https://scipython.com/blog/the-lorenz-attractor/
log = logging.getLogger(__name__)


# TODO (big): update this file to use the new setup that we've come up with

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

    num_traj = cfg.lorenz.num_traj
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
            # Add parameters the way that the generation object is
            # TODO take generic parameters rather than only PD Target
            l.P = cfg.lorenz.beta
            l.D = cfg.lorenz.rho
            l.target = cfg.lorenz.sigma

            data_Seq.append(l)

        if cfg.plot: plot_lorenz(data_Seq, cfg, predictions=None)

        if cfg.save_data:
            log.info("Saving new default data")
            torch.save((data_Seq), hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)
            log.info(f"Saved trajectories to {cfg.data_dir}")
    else:
        data_Seq = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)

    # Analysis
    from dynamics_model import DynamicsModel
    from reacher_pd import create_dataset_step, create_dataset_traj
    from plot import plot_loss

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble

    if cfg.train_models:
        if traj:
            dataset = create_dataset_traj(data_Seq, threshold=0.95)
        else:
            dataset = create_dataset_step(data_Seq)

    model = DynamicsModel(cfg)
    train_logs, test_logs = model.train(dataset, cfg)

    plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=True)

    if cfg.save_models:
        log.info("Saving new default models")
        torch.save(model,
                   hydra.utils.get_original_cwd() + '/models/lorenz/' + cfg.model.str + '.dat')

    # for i in range(cfg.num_eval):
    #     for m in cfg.models_to_eval:
    #         traj = np.random.randint(num_traj)
    #         new_init = data_Seq[traj].states[0]
    #         predictions_1 = [new_init.squeeze()]
    #         predictions_2 = [new_init.squeeze()]
    #         for i in range(1, n):
    #             pred_t = model_ct.predict(np.hstack((predictions_1[-1], i)))
    #             pred_no_t = predictions_2[-1] + model_1s.predict(predictions_2[-1])
    #             predictions_1.append(pred_t.squeeze())
    #             predictions_2.append(pred_no_t.squeeze())
    #
    #             # mse_t.append(np.square(groundtruth - pred_t).mean())
    #             # mse_no_t.append(np.square(groundtruth - pred_no_t).mean())
    #             current = pred_no_t.squeeze()
    #
    #         p_1 = np.stack(predictions_1)
    #         p_2 = np.stack(predictions_2)


def plot_learning():
    raise NotImplementedError("TODO")


if __name__ == '__main__':
    sys.exit(lorenz())
