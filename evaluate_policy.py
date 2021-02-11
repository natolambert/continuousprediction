import sys
import hydra
import logging
import numpy as np
from dotmap import DotMap
import gym
from envs import *
import torch
import pickle
import matplotlib.pyplot as plt
import cv2
from policy import NN
from dynamics_model import DynamicsModel

log = logging.getLogger(__name__)

def create_dataset_traj(data, t_range=0):
    """
    Creates dataset for training trajectory-based network

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    t_range: how far into a sequence to train for
    """
    data_in, data_out = [], []
    for id, sequence in enumerate(data):
        if id % 5 == 0: log.info(f"- processing seq {id}")
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        NN_param = sequence.params
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i + 1, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters
                dat = [states[i], j - i]
                dat.extend(NN_param)
                data_in.append(np.hstack(dat))
                data_out.append(states[j])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)
    return data_in, data_out

def create_dataset_traj_same_initial(data, t_range=0):
    """
    Creates dataset for training trajectory-based network

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    t_range: how far into a sequence to train for
    """
    data_in, data_out = [], []
    for id, sequence in enumerate(data):
        if id % 5 == 0: log.info(f"- processing seq {id}")
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        NN_param = sequence.params
        n = states.shape[0]
        for i in range(n - 1):  # From one state p
            dat = [states[0], i + 1]
            dat.extend(NN_param)
            data_in.append(np.hstack(dat))
            data_out.append(states[i+1])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)
    return data_in, data_out

def run_controller(env, horizon, policy):
    """
    Runs a Reacher3d gym environment for horizon timesteps, making actions according to policy

    :param env: A gym object
    :param horizon: The number of states forward to look
    :param policy: A policy object (see other python file)
    """

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    for i in range(horizon):
        # todo(action)
        action, t = policy.act(observation)

        next_obs, reward, done, info = env.step(action)

        if done:
            logs.actions = np.array(logs.actions)
            logs.rewards = np.array(logs.rewards)
            logs.states = np.array(logs.states)
            return logs

        # Log
        logs.actions.append(action)
        logs.rewards.append(reward)
        logs.states.append(observation.squeeze())
        observation = next_obs

    logs.actions = np.array(logs.actions)
    logs.rewards = np.array(logs.rewards)
    logs.states = np.array(logs.states)
    return logs

def collect_data(cfg, env):
    logs = []
    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_width = cfg.h_width
    h_layers = cfg.h_layers
    lower_bound = cfg.param_bounds[0]
    higher_bound = cfg.param_bounds[1]
    num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)
    for i in range(cfg.eval_trials):
        log.info('Trial %d' % i)
        env.seed(i)
        s0 = env.reset()

        params = np.random.rand(num_params) * (higher_bound - lower_bound) + lower_bound

        if (i == 0):
            policy = NN(h_layers, h_width, n_in, n_out)
        else:
            policy.update_params(params)

        dotmap = run_controller(env, horizon=cfg.eval_trial_length, policy=policy)

        dotmap.params = params
        logs.append(dotmap)

    return logs

@hydra.main(config_path='conf/nn_plan.yaml')
def evaluate_policy(cfg):
    f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
    file_est = f + cfg.dir_traj_state_action + '_iter' + str(iter) + '_' + cfg.dir_traj_est + '.dat'
    states_actions_dict_est = pickle.load(open(file_est, "rb"))
    print(states_actions_dict_est['states'].shape)

    file = f + cfg.dir_traj_state_action + '_iter' + str(iter) + '.dat'
    states_actions_dict = pickle.load(open(file, "rb"))
    print(states_actions_dict['states'].shape)
    print(states_actions_dict['actions'].shape)

@hydra.main(config_path='conf/nn_plan.yaml')
def evaluate_dynamics_model(cfg):
    if cfg.plot:
        f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
        model_done = f + cfg.dir_model_prediction + '_t_opt2_done.dat'
        model_no_hidden = f + cfg.dir_model_prediction + '_t_opt2_no_hidden.dat'
        model_one_step = f + cfg.dir_model_prediction + '_one_step.dat'
        saved_run_done = pickle.load(open(model_done, 'rb'))
        saved_run_no_hidden = pickle.load(open(model_no_hidden, 'rb'))
        saved_run_one_step = pickle.load(open(model_one_step, 'rb'))
        plt.title("Average prediction MSE over " + str(cfg.eval_trials) + " Trials")
        plt.xlabel("Time steps")
        plt.ylabel("MSE")
        plt.plot(saved_run_done['x'], saved_run_done['y'], color='red', label='1 hidden')
        plt.plot(saved_run_no_hidden['x'], saved_run_no_hidden['y'], color='blue', label='no hidden')
        plt.plot(saved_run_one_step['x'], saved_run_one_step['y'], color='green', label='one step')
        plt.legend(loc='upper left')
        plt.show()
    else:
        env_model = cfg.env.name
        env_label = cfg.env.label
        env = gym.make(env_model)
        f = hydra.utils.get_original_cwd() + '/models/' + env_label + '/'
        model = torch.load(f + cfg.traj_model + '.dat')

        eval_data = collect_data(cfg, env)
        dataset = create_dataset_traj_same_initial(eval_data,
                                    t_range=cfg.eval_trial_length)

        est_states = model.predict(dataset[0]).numpy()
        errs = np.array([])
        for i in range(cfg.eval_trial_length - 1):
            err_sum = 0
            for j in range(cfg.eval_trials):
                err_sum += ((dataset[1][j*(cfg.eval_trial_length-1) + i] - est_states[j*(cfg.eval_trial_length-1) + i])**2).mean()
            errs = np.append(errs, err_sum/cfg.eval_trials)

        x = np.arange(1, cfg.eval_trial_length)
        saved_run = {}
        saved_run['x'] = x
        saved_run['y'] = errs
        f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
        save_file = f + cfg.dir_model_prediction + '_' + cfg.traj_model + '.dat'
        pickle.dump(saved_run, open(save_file, 'wb'))
        plt.title("Average prediction MSE over " + str(cfg.eval_trials) + " Trials")
        plt.xlabel("Time steps")
        plt.ylabel("MSE")
        plt.plot(x, errs, color='red')
        plt.show()

        if (cfg.eval_trials == 1):
            print("Rendering video")

            f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
            video_name_est = f + cfg.dir_model_eval_vid + '_' + cfg.dir_traj_est + '.mp4'
            video_name = f + cfg.dir_model_eval_vid + '.mp4'
            frame = cv2.cvtColor(env.render_est(eval_data[0].states[0], mode="rgb_array"), cv2.COLOR_BGR2RGB)
            height, width, layers = frame.shape
            video_est = cv2.VideoWriter(video_name_est, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

            for i in range(cfg.eval_trial_length-1):
                video_est.write(cv2.cvtColor(env.render_est(est_states[i], mode="rgb_array"), cv2.COLOR_BGR2RGB))

            video_est.release()
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
            for i in range(cfg.eval_trial_length-1):
                video.write(cv2.cvtColor(env.render_est(dataset[1][i], mode="rgb_array"), cv2.COLOR_BGR2RGB))
            video.release()


if __name__ == '__main__':
    sys.exit(evaluate_dynamics_model())
