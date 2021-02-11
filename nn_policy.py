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

        done = get_reward_cartpole(next_obs)

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

def get_reward_cartpole(output_state):
    x_threshold = 9.6
    theta_threshold_radians = .418879  # 24 * 2 * math.pi / 360
    x = output_state[0]
    theta = output_state[2]
    done = bool(x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians)
    return 0 if done else 1

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

def cmaes_opt(cfg, model, obs, horizon, no_est = False, nn_policy = None, env = None):
    print("running cmaes")
    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_width = cfg.h_width
    h_layers = cfg.h_layers
    lower_bound = cfg.param_bounds[0]
    higher_bound = cfg.param_bounds[1]
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
    #opts = {'bounds': [[lower_bound]*num_params, [higher_bound]*num_params], 'tolfun': 1e-7}
    es = cma.CMAEvolutionStrategy(num_params*[0], 1, opts)
    if (no_est):
        es.optimize(opt_func_env)
    else:
        es.optimize(opt_func)
    res = es.result.xbest
    return res

def save_vid_est(initial_obs, policy_params, iter, cfg, env, model, video = None):
    '''
    helper function for saving videos of estimated trajectories
    :param initial_obs: initial observation
    :param nn_policy: policy used for action
    :param iter: current iteration for file name
    :param cfg: config file
    :param env: environment object
    :param model: model used for dynamics estimation
    '''
    f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
    video_name = f + cfg.dir_traj_vid + '_iter' + str(iter) + '_' + cfg.dir_traj_est + '.mp4'
    file = f + cfg.dir_traj_state_action + '_iter' + str(iter) + '_' + cfg.dir_traj_est + '.dat'
    states_actions_dict = {}
    states = np.array([])
    frame = cv2.cvtColor(env.render_est(initial_obs, mode="rgb_array"), cv2.COLOR_BGR2RGB)
    height, width, layers = frame.shape
    if (video == None):
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    big_dat = []
    for i in range(cfg.plan_trial_timesteps):
        dat = [initial_obs, i+1]
        dat.extend(policy_params)
        big_dat.append(np.hstack(dat))
    big_dat = np.vstack(big_dat)
    output_states = model.predict(big_dat).numpy()

    if (video == None):
        video.write(cv2.cvtColor(env.render_est(initial_obs, mode="rgb_array"), cv2.COLOR_BGR2RGB))
    np.append(states, initial_obs)
    if (cfg.replan_per_timestep):
        video.write(cv2.cvtColor(env.render_est(output_states[0], mode="rgb_array"), cv2.COLOR_BGR2RGB))
    else:
        for i in range(cfg.plan_trial_timesteps):
            np.append(states, output_states[i])
            video.write(cv2.cvtColor(env.render_est(output_states[i], mode="rgb_array"), cv2.COLOR_BGR2RGB))
    #video.release()
    #TODO: FIX THE STATES HERE LATER
    states_actions_dict['states'] = states
    pickle.dump(states_actions_dict, open(file, "wb"))
    return video

@hydra.main(config_path='conf/nn_plan.yaml')
def plan(cfg):
    # Step 1: run random base policy to collect data points
    # Environment setup
    env_model = cfg.env.name
    env_label = cfg.env.label
    env = gym.make(env_model)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    env.seed(cfg.random_seed)

    prob = cfg.model.prob
    ens = cfg.model.ensemble

    n_in = cfg.env.state_size
    n_out = cfg.env.action_size
    h_layers = cfg.h_layers
    h_width = cfg.h_width

    log.info('Initializing env: %s' % env_model)
    # collect data through reacher environment
    if cfg.load_model:
        f = hydra.utils.get_original_cwd() + '/models/' + env_label + '/'
        model = torch.load(f + cfg.traj_model + '.dat')
    elif (not cfg.no_est):
        log.info("Collecting initial data")
        exper_data = collect_data(cfg, env)

        # Step 2: Learn dynamics model
        # probabilistic model, ensemble training booleans
        # create dataset for model input/output
        log.info("Creating initial dataset for model training")
        num_params = n_in*h_width + 2*h_width + 1 + h_layers*(h_width*h_width + h_width)
        model = DynamicsModel(cfg, nn_policy_param_size=num_params)
        dataset = create_dataset_traj(exper_data,
                                    t_range=cfg.initial_num_trial_length)
    else:
        model = 0

    # retrain for n_iter iterations
    rews = np.zeros(cfg.n_iter)
    nn_policy = NN(h_layers, h_width, n_in, n_out)
    for i in range(cfg.n_iter):
        log.info(f"Iteration {i}")

        if not cfg.load_model and not cfg.no_est:
            if i==0 or cfg.retrain_model:
                # shuffle dataset each iteration to avoid retraining on the same data points -> overfitting
                shuffle_idxs = np.arange(0, dataset[0].shape[0], 1)
                np.random.shuffle(shuffle_idxs)
                # config for choosing number of points to train on
                if (cfg.num_training_points == 0):
                    training_dataset = (dataset[0][shuffle_idxs], dataset[1][shuffle_idxs])
                else:
                    training_dataset = (
                        dataset[0][shuffle_idxs[:cfg.num_training_points]],
                        dataset[1][shuffle_idxs[:cfg.num_training_points]])
                log.info(f"Training model P:{prob}, E:{ens}")
                # train model
                train_logs, test_logs = model.train(training_dataset, cfg)
                if cfg.save_model:
                    f = hydra.utils.get_original_cwd() + '/models/' + env_label + '/'
                    torch.save(model, f + cfg.traj_model + '.dat')

        # initial observation
        obs = env.reset()
        if (cfg.horizon == 0):
            horizon = int(cfg.plan_trial_timesteps / cfg.num_MPC_per_iter)
        else:
            horizon = cfg.horizon
        for j in range(cfg.num_MPC_per_iter):
            if (cfg.replan_per_timestep):
                log.info(f"TIME STEP {j}")
            # plan using MPC with random shooting optimizer
            if (cfg.use_cmaes):
                policy = cmaes_opt(cfg, model, obs, horizon, no_est = cfg.no_est, nn_policy = nn_policy, env = env)
            else:
                policy, policy_reward = random_shooting_mpc(cfg, model, obs, horizon)
            nn_policy.update_params(policy)
            if (cfg.save_video_est):
                if (j == 0):
                    video_est = save_vid_est(obs, policy, i, cfg, env, model)
                else:
                    video_est = save_vid_est(obs, policy, i, cfg, env, model, video = video_est)
            # collect data on trajectory
            logs = DotMap()
            logs.states = []
            logs.actions = []
            logs.rewards = []
            logs.times = []

            logs.params = policy
            obs = env.reset(initial = obs)
            if(cfg.save_video and j == 0):
                f = hydra.utils.get_original_cwd() + '/opt_traj/traj'
                file = f + cfg.dir_traj_state_action + '_iter' + str(iter) + '_' + cfg.dir_traj_est + '.dat'
                states_actions_dict = {}
                states = np.array([])
                actions = np.array([])
                video_name = f + cfg.dir_traj_vid + '_iter' + str(i) + '.mp4'
                frame = cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_BGR2RGB)
                height, width, layers = frame.shape
                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
            if (j==0):
                video.write(cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_BGR2RGB))
            np.append(states, obs)
            for k in range(horizon):
                if (not cfg.replan_per_timestep):
                    log.info(f"Time step: {k}")
                # step in environment
                action, _ = nn_policy.act(obs)
                action = np.clip(action, -1, 1)
                next_obs, reward, done, info = env.step(action)
                if (cfg.save_video):
                    np.append(states, next_obs)
                    np.append(actions, action)
                    video.write(cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_BGR2RGB))
                rews[i] += get_reward_cartpole(next_obs)

                if done:
                    break

                # collect data on trajectory
                logs.actions.append(action)
                logs.rewards.append(reward)
                logs.states.append(obs.squeeze())
                # set next observation
                obs = next_obs
                if (cfg.replan_per_timestep):
                    break
                # don't collect last observation, does not make sense when creating dataset
                if (j == cfg.num_MPC_per_iter - 1 and k == horizon - 1):
                    continue

            logs.actions = np.array(logs.actions)
            logs.rewards = np.array(logs.rewards)
            logs.states = np.array(logs.states)

            if not cfg.load_model and cfg.retrain_model:
                data_in, data_out = create_dataset_traj([logs],
                                                        t_range=cfg.plan_trial_timesteps)
                dataset = (np.append(dataset[0], data_in, axis=0), np.append(dataset[1], data_out, axis=0))
            if done:
                break

        if (cfg.save_video):
            video_est.release()
            video.release()
            states_actions_dict['states'] = states
            states_actions_dict['actions'] = actions
            pickle.dump(states_actions_dict, open(file, 'wb'))

        log.info(f"Final cumulative reward: {rews[i]}")

    log.info(f"Final cumulative rewards: {rews}")
    log.info(f"Mean cumrew: {np.mean(rews)}")
    log.info(f"stddev cumrew: {np.std(rews)}")


if __name__ == '__main__':
    sys.exit(plan())
