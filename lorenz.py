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
from plot import plot_lorenz, plot_mse, plot_mse_err, plot_states, setup_plotting
from evaluate import test_models

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

def collect_data(cfg):
    num_traj = cfg.lorenz.num_traj
    sigma = cfg.lorenz.sigma
    beta = cfg.lorenz.beta
    rho = cfg.lorenz.rho

    tmax, n = cfg.lorenz.tmax, cfg.lorenz.n
    t = np.linspace(0, tmax, n)

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
        l.P = beta
        l.D = rho
        l.target = sigma

        data_Seq.append(l)
    return data_Seq


@hydra.main(config_path='conf/lorenz.yaml')
def lorenz(cfg):

    # u0, v0, w0 = cfg.lorenz.ex.u0, cfg.lorenz.ex.v0, cfg.lorenz.ex.w0

    # # Maximum time point and total number of time points
    # tmax, n = cfg.lorenz.tmax, cfg.lorenz.n
    #
    # # Integrate the Lorenz equations on the time grid t
    # t = np.linspace(0, tmax, n)
    # f = odeint(sim_lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
    # x, y, z = f.T

    train = cfg.mode == 'train'

    if not train:
        # Lorenz paramters and initial conditions

        #data_X = np.zeros((1, 3))

        train_data = collect_data(cfg)
        test_data = collect_data(cfg)

        #TODO: fix this, setup_plotting needs model
        if cfg.plot:
            setup_plotting({cfg.model.str: model})
            plot_lorenz(train_data, cfg, predictions=None)

        log.info("Saving new default data")
        torch.save((train_data, test_data), hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)
        log.info(f"Saved trajectories to raw{cfg.data_dir}")
    else:
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)

    # Analysis
    if train:

        from dynamics_model import DynamicsModel
        from reacher_pd import create_dataset_step, create_dataset_traj
        from plot import plot_loss

        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        if traj:
            dataset = create_dataset_traj(train_data, control_params=cfg.model.training.control_params,
                                            train_target=cfg.model.training.train_target, threshold=0.95)
        else:
            dataset = create_dataset_step(train_data, delta=delta)

        model = DynamicsModel(cfg, env = "Lorenz")
        train_logs, test_logs = model.train(dataset, cfg)
        log.info("Saving new default models")
        torch.save(model,
                   hydra.utils.get_original_cwd() + '/models/lorenz/' + cfg.model.str + '.dat')
        setup_plotting({cfg.model.str: model})
        plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=True)

        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)
        setup_plotting({cfg.model.str: model})
        plot_lorenz(train_data, cfg, predictions=None)

    # models = {}
    # for model_type in cfg.models_to_eval:
    #     models[model_type] = torch.load(hydra.utils.get_original_cwd() + '/models/lorenz/' + model_type + ".dat")

    # mse_evald = []
    # for i in range(cfg.num_eval):
    #     traj_idx = np.random.randint(num_traj)
    #     traj = data_Seq[traj_idx]
    #     MSEs, predictions = test_models([traj], models)
    #
    #     MSE_avg = {key: np.average(MSEs[key], axis=0) for key in MSEs}
    #
    #     mse = {key: MSEs[key].squeeze() for key in MSEs}
    #     mse_sub = {key: mse[key][mse[key] < 10 ** 5] for key in mse}
    #     pred = {key: predictions[key] for key in predictions}
    #     mse_evald.append(mse)
    #     #
    #     # plot_states(traj.states, pred, save_loc="Predictions; traj-" + str(traj_idx), idx_plot=[0,1,2], show=False)
    #     # plot_mse(mse_sub, save_loc="Error; traj-" + str(traj_idx), show=False)
    #     plot_lorenz([traj], cfg, predictions=pred)
    #
    #
    # plot_mse_err(mse_evald, save_loc="Err Bar MSE of Predictions", show=True)


if __name__ == '__main__':
    sys.exit(lorenz())
