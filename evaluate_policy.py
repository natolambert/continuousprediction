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
from matplotlib.pyplot import cm
import cv2
from policy import NN
from dynamics_model import DynamicsModel
from nn_policy import create_dataset_traj

log = logging.getLogger(__name__)

def create_dataset_traj_same_initial(data, t_range=0):
    """
    Creates dataset for training trajectory-based network with the same initial obs

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
    Runs a gym environment for horizon timesteps, making actions according to policy

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

        # if done:
        #     logs.actions = np.array(logs.actions)
        #     logs.rewards = np.array(logs.rewards)
        #     logs.states = np.array(logs.states)
        #     return logs

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
    '''
    Collects data for a gym environment, saving parameters for training network
    Uses a neural network as the policy for the run
    '''
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
def evaluate_states(cfg):
    # Setup environment
    env_model = cfg.env.name
    env_label = cfg.env.label
    env = gym.make(env_model)

    # Load in the states of the run
    f = hydra.utils.get_original_cwd() + '/opt_traj/'
    f_states = f + 'states/' + cfg.eval_dir + '_states.dat'
    f_est_states = f + 'states/' + cfg.eval_dir + '_est_states.dat'

    states = pickle.load(open(f_states, 'rb'))
    est_states = pickle.load(open(f_est_states, 'rb'))

    for i in range(len(states)):

        video_name_est = f + '/eval/videos/' + cfg.eval_dir + '_iter' + str(i) + '_est.mp4'
        video_name = f + '/eval/videos/' + cfg.eval_dir + '_iter' + str(i) + '.mp4'

        frame = cv2.cvtColor(env.render_est(est_states[i][0], mode="rgb_array"), cv2.COLOR_BGR2RGB)
        height, width, layers = frame.shape
        video_est = cv2.VideoWriter(video_name_est, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

        for j in range(len(est_states[i])):
            video_est.write(cv2.cvtColor(env.render_est(est_states[i][j], mode="rgb_array"), cv2.COLOR_BGR2RGB))
        for j in range(len(states[i])):
            video.write(cv2.cvtColor(env.render_est(states[i][j], mode="rgb_array"), cv2.COLOR_BGR2RGB))

        video_est.release()
        video.release()

    return None

@hydra.main(config_path='conf/nn_plan.yaml')
def evaluate_dynamics_model(cfg):

    # Setup environment
    env_model = cfg.env.name
    env_label = cfg.env.label
    env = gym.make(env_model)

    # Random seed setup
    np.random.seed(cfg.random_seed)
    env.seed(cfg.random_seed)

    if (cfg.do_eval):
        # load model for evaluation
        f = hydra.utils.get_original_cwd() + '/models/' + env_label + '/'
        model = torch.load(f + cfg.eval_dir + '_model.dat')

        # collect evaluation trajectories
        eval_data = collect_data(cfg, env)
        dataset = create_dataset_traj_same_initial(eval_data,
                                    t_range=cfg.eval_trial_length)

        # MSE calculation
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
        f = hydra.utils.get_original_cwd() + '/opt_traj/eval/models/'
        save_file = f + cfg.eval_dir + '.dat'
        pickle.dump(saved_run, open(save_file, 'wb'))

    if cfg.plot_dynamics:
        f = hydra.utils.get_original_cwd() + '/opt_traj/eval/models/'
        plot_dirs = cfg.plot_dir
        saved_evals = []
        for i in range(len(plot_dirs)):
            saved_eval = pickle.load(open(f + plot_dirs[i] + '.dat', 'rb'))
            saved_evals.append(saved_eval)
        plt.title("Average prediction MSE over " + str(cfg.eval_trials) + " Trials")
        plt.xlabel("Time steps")
        plt.ylabel("MSE")

        color=iter(cm.rainbow(np.linspace(0,1,len(plot_dirs))))
        for i in range(len(plot_dirs)):
            plt.plot(saved_evals[i]['x'], saved_evals[i]['y'], color = next(color), label = plot_dirs[i])

        plt.legend(loc='upper left')
        plt.show()

'''
probably should not do it like this, but at least it's organized
evaluate_dynamics_model: evaluates dynamics model on a number of evaluation trajectories
evaluate_states: saves videos of the runs
'''
if __name__ == '__main__':
    sys.exit(evaluate_states())
