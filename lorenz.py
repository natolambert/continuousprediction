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
from evaluate import test_models

# adapeted from https://scipython.com/blog/the-lorenz-attractor/
log = logging.getLogger(__name__)


def sim_lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


def collect_data(cfg):
    num_traj = cfg.num_trials
    sigma = cfg.lorenz.sigma
    beta = cfg.lorenz.beta
    rho = cfg.lorenz.rho

    tmax, n = cfg.lorenz.tmax, cfg.lorenz.n
    t = np.linspace(0, tmax, n)

    data_Seq = []
    new_init = np.random.uniform(low=[5, 5, 5], high=[10, 10, 10], size=(num_traj, 3))

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
    from dynamics_model import DynamicsModel

    mode = cfg.mode
    name = cfg.env.label
    if cfg.plot:
        from plot import plot_lorenz, plot_mse, plot_mse_err, plot_states, setup_plotting, plot_loss

    if mode == 'collect':
        train_data = collect_data(cfg)
        test_data = collect_data(cfg)

        model = DynamicsModel(cfg)
        # TODO: fix this, setup_plotting needs model
        if cfg.plot:
            setup_plotting({cfg.model.str: model})
            plot_lorenz(train_data, cfg, predictions=None)

        log.info("Saving new default data")
        torch.save((train_data, test_data),
                   hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)
        log.info(f"Saved trajectories to raw{cfg.data_dir}")
    else:
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)



    # Analysis
    if mode == 'train':
        from reacher_pd import create_dataset_step, create_dataset_traj

        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        if traj:
            dataset = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'temp_traj.dat')
            # dataset = create_dataset_traj(train_data, control_params=cfg.model.training.control_params,
            #                               train_target=cfg.model.training.train_target)
            # torch.save(dataset, hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'temp_traj.dat')
        else:
            dataset = create_dataset_step(train_data, delta=delta)

        model = DynamicsModel(cfg, env="Lorenz")
        train_logs, test_logs = model.train(dataset, cfg)
        log.info("Saving new default models")
        torch.save(model,
                   hydra.utils.get_original_cwd() + '/models/lorenz/' + cfg.model.str + '.dat')
        if cfg.plot:
            plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=True)

        # log.info(f"Loading default data")
        # (train_data, test_data) = torch.load(
        #     hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir)
        # setup_plotting({cfg.model.str: model})
        # plot_lorenz(train_data, cfg, predictions=None)

    elif mode == 'plot':
        # TODO add plotting code for predictions
        # Load test data
        log.info(f"Loading default data")
        (test_data, _) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/' + cfg.env.label + '/' + 'raw' + cfg.data_dir)

        # Load models
        log.info("Loading models")
        model_types = cfg.plotting.models
        models = {}
        f = hydra.utils.get_original_cwd() + '/models/' + cfg.env.label + '/'
        for model_type in model_types:
            models[model_type] = torch.load(f + model_type + ".dat")

        graph_file = 'Plots'
        os.mkdir(graph_file)
        setup_plotting(models)

        # Select a random subset of training data
        # idx = np.random.randint(0, len(data), num)
        idx = np.random.choice(np.arange(len(test_data)), size=min(len(test_data), cfg.plotting.num_eval_test),
                               replace=False)
        dat = [test_data[i] for i in idx]

        for entry in dat:
            entry.states = entry.states[0:cfg.plotting.t_range]
            # entry.rewards = entry.rewards[0:cfg.plotting.t_range]
            # entry.actions = entry.actions[0:cfg.plotting.t_range]

        MSEs, predictions = test_models(dat, models, env=name)

        from plot import plot_mse_err, plot_lorenz

        # plot_lorenz(test_data, cfg, predictions=predictions)
        mse_evald = []
        for i, id in list(enumerate(idx)):
            mse = {key: MSEs[key][i].squeeze() for key in MSEs}
            mse_sub = {key: [(x if x < 10 ** 5 else float("nan")) for x in mse[key]] for key in mse}
            mse_evald.append(mse)

        plot_mse_err(mse_evald, log_scale=True, title="Average MSE", save_loc=graph_file + 'mse.pdf', show=True,
                     legend=False)


if __name__ == '__main__':
    sys.exit(lorenz())
