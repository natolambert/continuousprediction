'''
TODO:
'''

'''
MBRL using NN control policy
Cartpole environment
'''

import sys
import hydra
import logging
import numpy as np
from dotmap import DotMap
import gym
from envs import *
from policy import NN
from dynamics_model import DynamicsModel
import torch
import cma
import pickle
import cv2

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

        if done == 0:
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
    for i in range(cfg.initial_num_trials):
        log.info('Trial %d' % i)
        env.seed(i)
        s0 = env.reset()

        params = np.random.rand(num_params) * (higher_bound - lower_bound) + lower_bound

        if (i == 0):
            policy = NN(h_layers, h_width, n_in, n_out)
        else:
            policy.update_params(params)

        dotmap = run_controller(env, horizon=cfg.initial_num_trial_length, policy=policy)

        dotmap.params = params
        logs.append(dotmap)

    return logs

def collect_data_specify(cfg):
    '''
    Collects data from previous runs
    Uses a neural network as the policy for the run
    '''
    logs = []
    f = hydra.utils.get_original_cwd() + '/opt_traj/'
    f_policies = f + 'policies/' + cfg.train_data_dir + '_policies.dat'
    f_states = f + 'states/' + cfg.train_data_dir + '_states.dat'
    policies = pickle.load(open(f_policies, 'rb'))
    states = pickle.load(open(f_states, 'rb'))
    for i in range(len(states)):
        log.info('Trial %d' % i)

        dotmap = DotMap()
        dotmap.states = np.array(states[i])
        dotmap.params = policies[i]
        logs.append(dotmap)

    return logs

def get_reward_cartpole(output_state):
    '''
    returns reward for the cartpole environment
    '''
    x_threshold = 9.6
    theta_threshold_radians = .418879  # 24 * 2 * math.pi / 360
    x = output_state[0]
    theta = output_state[2]
    done = bool(x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians)
    return 0 if done else 1

# CODE FOR RANDOM SHOOTING OPTIMIZER, USE CMA-ES INSTEAD
def cum_reward(policies, model, initial_obs, horizon):
    '''
    Calculates the cumulative reward of a run with a given policy and target
    :param policy: policy used to get actions
    :param model: model used to estimate dynamics
    :param initial_obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = np.zeros(len(policies))
    for i in range(horizon):
        big_dat = []
        for j in range(len(policies)):
            dat = [initial_obs, i + 1]
            dat.extend(policies[j])
            big_dat.append(np.hstack(dat))
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat).numpy()
        reward_sum += np.array([get_reward_cartpole(output_states[j]) for j in range(len(policies))])
    return reward_sum

# CODE FOR RANDOM SHOOTING OPTIMIZER, USE CMA-ES INSTEAD
def random_shooting_mpc_pool_helper(params):
    """Helper function used for multiprocessing"""
    num_random_configs, model, obs, horizon, seed, num_params, lower_bound, higher_bound = params
    np.random.seed(seed)
    policies = []
    for i in range(num_random_configs):
        policy = np.random.rand(num_params) * (higher_bound - lower_bound) + lower_bound
        policies.append(policy)
    rewards = cum_reward(policies, model, obs, horizon)
    return policies[np.argmax(rewards)], np.max(rewards)

# CODE FOR RANDOM SHOOTING OPTIMIZER, USE CMA-ES INSTEAD
def random_shooting_mpc(cfg, model, obs, horizon):
    '''
    Creates random PID configurations and returns the one with the best cumulative reward
    :param model: model to use to predict dynamics
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    :return: the PID policy that has the best cumulative reward, and the best cumulative reward (for evaluation)
    '''
    from multiprocessing import Pool
    num_random_configs = cfg.num_random_configs
    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_width = cfg.h_width
    h_layers = cfg.h_layers
    lower_bound = cfg.param_bounds[0]
    higher_bound = cfg.param_bounds[1]
    num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)
    with Pool(10) as p:
        function_inputs = [(num_random_configs // 10, model, obs, horizon, i, num_params, lower_bound, higher_bound) for i in range(10)]
        out = p.map(random_shooting_mpc_pool_helper, function_inputs)
    return max(out, key=lambda x: x[1])

def cmaes_opt_real(cfg, obs, horizon, env, nn_policy):
    '''
    CMA-ES optimizer
    :param obs: initial observation of the environment
    :param horizon: horizon for the optimization
    :param env: environment used to evaluate reward
    :param nn_policy: policy used on environment
    '''

    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_width = cfg.h_width
    h_layers = cfg.h_layers
    num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)

    def opt_func_env(policy_params):
        # version of the objective function w/out dynamics estimation
        nn_policy.update_params(policy_params)
        observation = env.reset(initial = obs)
        reward_sum = 0
        for i in range(horizon):
            action, _ = nn_policy.act(observation)

            next_obs, reward, done, info = env.step(action)
            reward_sum -= reward

            if done:
                break
            observation = next_obs
        return reward_sum

    opts = cma.CMAOptions()
    opts.set('tolfun', 1)
    es = cma.CMAEvolutionStrategy(num_params*[0], 1, opts)
    es.optimize(opt_func_env)
    res = es.result.xbest, es.result.fbest
    return res

def cmaes_opt(cfg, model, obs, horizon):
    '''
    CMA-ES optimizer
    :param model: model used for estimating dynamics to get reward
    :param obs: initial observation of the environment
    :param horizon: horizon for the optimization
    '''

    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_width = cfg.h_width
    h_layers = cfg.h_layers
    num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)

    def opt_func(policy_params):
        reward_sum = 0
        reward_array = np.zeros(horizon)
        big_dat = []
        for i in range(horizon):
            dat = [obs, i+1]
            dat.extend(policy_params)
            big_dat.append(np.hstack(dat))
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat).numpy()
        reward_array = np.array([get_reward_cartpole(output_states[j]) for j in range(horizon)])
        return -1*reward_array.sum()

    opts = cma.CMAOptions()
    opts.set('tolfun', 1)
    es = cma.CMAEvolutionStrategy(num_params*[0], 1, opts)
    es.optimize(opt_func)
    res = es.result.xbest, es.result.fbest
    return res

@hydra.main(config_path='conf/nn_plan.yaml')
def plan(cfg):

    # Environment setup
    env_model = cfg.env.name
    env_label = cfg.env.label
    env = gym.make(env_model)

    # Random seed setup
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    env.seed(cfg.random_seed)

    # Dynamics Estimation Model setup
    prob = cfg.model.prob
    ens = cfg.model.ensemble

    # Neural Netowrk Policy parameters
    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_layers = cfg.h_layers
    h_width = cfg.h_width

    # Save directory setup
    f = hydra.utils.get_original_cwd() + '/opt_traj/'
    f_model_load = hydra.utils.get_original_cwd() + '/models/' + env_label + '/' + cfg.load_model_name + '_model.dat'
    f_model_save = hydra.utils.get_original_cwd() + '/models/' + env_label + '/' + cfg.run_name + '_model.dat'
    f_policies = f + 'policies/' + cfg.run_name + '_policies.dat'
    f_states = f + 'states/' + cfg.run_name + '_states.dat'
    f_est_states = f + 'states/' + cfg.run_name + '_est_states.dat'
    f_rews = f + 'states/' + cfg.run_name + '_rews.dat'

    # GETTING THE MODEL, train/load model
    log.info("GETTING MODEL")
    if cfg.load_model:
        # load the model
        log.info("Loading model")
        model = torch.load(f_model_load)
    elif cfg.train_model:
        # train a new model
        log.info("Training new model")
        if (cfg.train_data):
            # collecting data from specified previous runs
            exper_data = collect_data_specify(cfg)
        else:
            # collecting data from environment with random NN policies
            exper_data = collect_data(cfg, env)
        log.info("Creating initial dataset for model training")
        num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)
        model = DynamicsModel(cfg, nn_policy_param_size=num_params)
        # making data into dataset for training
        dataset = create_dataset_traj(exper_data,
                                    t_range=cfg.initial_num_trial_length)
        shuffle_idxs = np.arange(0, dataset[0].shape[0], 1)
        np.random.shuffle(shuffle_idxs)
        training_dataset = (dataset[0][shuffle_idxs], dataset[1][shuffle_idxs])
        log.info(f"Training model P:{prob}, E:{ens}")
        # train model
        train_logs, test_logs = model.train(training_dataset, cfg)
    if cfg.save_model:
        torch.save(model, f_model_save)

    # keep track of rewards every iteration
    rews = []
    # keep track of states every iteration
    states = []
    states_est = []
    # keep track of policies every iteration
    policies = []
    # initialize the neural network policy
    nn_policy = NN(h_layers, h_width, n_in, n_out)

    # run + optimize n_iter number of items
    for i in range(cfg.n_iter):
        log.info(f"Iteration {i}")

        # initial observation
        obs = env.reset()
        # set horizon to be equal to number of timesteps
        horizon = cfg.plan_trial_timesteps
        # keep track of states every time step
        states_iter = []
        states_est_iter = []
        # keep track of the cumulative reward this iter
        rews_iter = 0

        # run the optimizer
        if (cfg.no_est):
            policy, f_value = cmaes_opt_real(cfg, obs, horizon, env, nn_policy)
        else:
            policy, f_value = cmaes_opt(cfg, model, obs, horizon)
        policies.append(policy)

        # update the parameters for run
        nn_policy.update_params(policy)

        # reset the environment to the same start point
        # the cmaes optimization with real dynamics may move the environment from start
        obs = env.reset(initial = obs)

        # RUN OPTIMAL POLICY WITH LEARNED DYNAMICS
        big_dat = []
        for k in range(horizon):
            dat = [obs, k+1]
            dat.extend(policy)
            big_dat.append(np.hstack(dat))
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat).numpy()
        for k in range(horizon):
            states_est_iter.append(output_states[k])

        # RUN OPTIMAL POLICY WITH ACTUAL DYNAMICS
        for k in range(horizon):
            # step in environment
            action, _ = nn_policy.act(obs)
            action = np.clip(action, -1, 1)
            next_obs, reward, done, info = env.step(action)
            states_iter.append(next_obs)
            rews_iter += reward

            if done:
                break

            # set next observation
            obs = next_obs

        log.info(f"Final cumulative reward: {rews_iter}")
        rews.append(rews_iter)
        states.append(states_iter)
        states_est.append(states_est_iter)

    log.info(f"Final cumulative rewards: {rews}")
    log.info(f"Mean cumrew: {np.mean(rews)}")
    log.info(f"stddev cumrew: {np.std(rews)}")

    if (cfg.save_states):
        pickle.dump(states, open(f_states, 'wb'))
        pickle.dump(states_est, open(f_est_states, 'wb'))
    if (cfg.save_rews):
        pickle.dump(rews, open(f_rews, 'wb'))
    if (cfg.save_policies):
        pickle.dump(policies, open(f_policies, 'wb'))


if __name__ == '__main__':
    sys.exit(plan())
