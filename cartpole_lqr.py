import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dotmap import DotMap

import mujoco_py
import torch

import gym
from envs import *
from gym.wrappers import Monitor

import hydra
import logging

log = logging.getLogger(__name__)

from policy import LQR
from plot import plot_cp, plot_loss, setup_plotting

from dynamics_model import DynamicsModel
from reacher_pd import run_controller


###########################################
#                Datasets                 #
###########################################

def create_dataset_traj(data, control_params=True, train_target=True, threshold=0.0, delta=False, t_range=0):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    """
    data_in, data_out = [], []
    for sequence in data:
        states = sequence.states
        if t_range:
            states = states[:t_range]
        K = sequence.K
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters

                # The randomely continuing is something I thought of to shrink
                # the datasets while still having a large variety
                if np.random.random() < threshold:
                    continue
                dat = [states[i], j - i]
                dat.append(K)
                data_in.append(np.hstack(dat))
                # data_in.append(np.hstack((states[i], j-i, target)))
                if delta:
                    data_out.append(states[j] - states[i])
                else:
                    data_out.append(states[j])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0, is_lstm = False, lstm_batch = 0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        if t_range:
            states = states[:t_range]
        for i in range(states.shape[0] - 1):
            if 'actions' in sequence.keys():
                actions = sequence.actions
                if t_range:
                    actions = actions[:t_range]
                data_in.append(np.hstack((states[i], actions[i])))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
            else:
                data_in.append(np.array(states[i]))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
        if is_lstm:
            remainder = len(data_out)%lstm_batch
            if remainder:
                data_out = data_out[:len(data_out)-remainder]
                data_in = data_in[:len(data_in)-remainder]
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def collect_data_lqr(cfg, plot=False):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """

    env_model = cfg.env.name
    env = gym.make(env_model)
    # if (cfg.video):
    # env = Monitor(env, hydra.utils.get_original_cwd() + '/trajectories/reacher/video',
    # video_callable = lambda episode_id: episode_id==1,force=True)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []
    if (cfg.PID_test):
        target = np.random.rand(5) * 2 - 1

    s = np.random.randint(0, 100)
    for i in range(cfg.num_trials):
        log.info('Trial %d' % i)
        if (cfg.PID_test):
            env.seed(0)
        else:
            env.seed(s + i)
        s0 = env.reset()

        m_c = env.masscart
        m_p = env.masspole
        m_t = m_c + m_p
        g = env.gravity
        l = env.length
        A = np.array([
            [0, 1, 0, 0],
            [0, g * m_p / m_c, 0, 0],
            [0, 0, 0, 1],
            [0, 0, g * m_t / (l * m_c), 0],
        ])

        B = np.array([
            [0, 1 / m_c, 0, -1 / (l * m_c)],
        ])

        Q = np.diag([.5, .05, 1, .05])

        R = np.ones(1)

        n_dof = np.shape(A)[0]
        if cfg.data_mode == 'chaotic':
            modifier = .75 * np.random.random(4)
            lim = cfg.trial_timesteps
        elif cfg.data_mode == 'unstable':
            modifier = 1.5 * np.random.random(4) - .75
            env.theta_threshold_radians = 2 * env.theta_threshold_radians
            # default 2.4
            env.x_threshold = 2 * env.x_threshold
            lim = cfg.trial_timesteps
        else:
            modifier = .5 * np.random.random(4) + 1
            lim = cfg.trial_timesteps
        policy = LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0])
        policy.K = np.multiply(policy.K, modifier)
        if cfg.data_mode == 'rand':
            from policy import randomPolicy
            print("Running random policy")
            policy = randomPolicy(dX=4, dU=1, variance=0.5)
        elif cfg.data_mode == 'set':
            policy.K = np.array(cfg.K)

        # print(type(env))
        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video=cfg.video)
        while len(dotmap.states) < lim:
            env.seed(s)
            env.reset()
            if cfg.data_mode == 'chaotic':
                modifier = .75 * np.random.random(4)
            elif cfg.data_mode == 'unstable':
                modifier = 1.5 * np.random.random(4) - .75
                env.theta_threshold_radians = 2 * env.theta_threshold_radians
                # default 2.4
                env.x_threshold = 2 * env.x_threshold
            else:
                modifier = .5 * np.random.random(4) + 1
            policy = LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0])
            policy.K = np.multiply(policy.K, modifier)
            if cfg.data_mode == 'rand':
                from policy import randomPolicy
                print("Running random policy")
                policy = randomPolicy(dX=4, dU=1, variance=0.5)
            elif cfg.data_mode == 'set':
                policy.K = np.array(cfg.K)
            dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video=cfg.video)
            print(f"- Repeat simulation")
            s += 1
            # if plot and len(dotmap.states)>0: plot_cp(dotmap.states, dotmap.actions)

        if plot: plot_cp(dotmap.states, dotmap.actions, save=True)

        if not cfg.data_mode == 'rand':
            dotmap.K = np.array(policy.K).flatten()
        logs.append(dotmap)
        s += 1

    return logs


###########################################
#           Plotting / Output             #
###########################################


def log_hyperparams(cfg):
    log.info(cfg.model.str + ":")
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/cartpole_lqr.yaml')
def contpred(cfg):
    train = cfg.mode == 'train'
    # Collect data
    if not train:
        log.info(f"Collecting new trials")

        exper_data = collect_data_lqr(cfg, plot=cfg.plot)
        test_data = collect_data_lqr(cfg, plot=cfg.plot)

        log.info("Saving new default data")
        torch.save((exper_data, test_data),
                   hydra.utils.get_original_cwd() + '/trajectories/' +cfg.env.label + '/raw' + cfg.data_dir)
        log.info(f"Saved trajectories to {'/trajectories/' +cfg.env.label + '/raw' + cfg.data_dir}")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        # Todo re-save data
        (exper_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/'+cfg.env.label + '/raw' + cfg.data_dir)

    if train:
        it = range(cfg.copies) if cfg.copies else [0]
        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta
        is_lstm = cfg.model.lstm

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        log_hyperparams(cfg)

        for i in it:
            print('Training model %d' % i)

            # if cfg.model.training.num_traj:
            #     train_data = exper_data[:cfg.model.training.num_traj]
            # else:
            train_data = exper_data

            if traj:
                dataset = create_dataset_traj(exper_data, control_params=cfg.model.training.control_params,
                                              train_target=cfg.model.training.train_target,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
            else:
                dataset = create_dataset_step(train_data, delta=delta, is_lstm = is_lstm, lstm_batch = cfg.model.optimizer.batch)

            model = DynamicsModel(cfg)
            train_logs, test_logs = model.train(dataset, cfg)

            # setup_plotting({cfg.model.str: model})
            # plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)

            log.info("Saving new default models")
            f = hydra.utils.get_original_cwd() + '/models/' + cfg.env.label + '/'
            if cfg.exper_dir:
                f = f + cfg.exper_dir + '/'
                if not os.path.exists(f):
                    os.mkdir(f)
            copystr = "_%d" % i if cfg.copies else ""
            f = f + cfg.model.str + copystr + '.dat'
            torch.save(model, f)
        # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless


if __name__ == '__main__':
    sys.exit(contpred())
