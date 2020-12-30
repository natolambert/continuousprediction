'''
TODO:
'''

'''
Version of replan.py that works with the final 3 values in reacher obs (fingertip - goal)
make sure in reacher.yaml state_indices includes states 18, 19, 20 as the last 3 states
'''

import sys
import hydra
import logging
import numpy as np
from dotmap import DotMap
import mujoco_py
import gym
from envs import *
from policy import PID
from dynamics_model import DynamicsModel
import torch

log = logging.getLogger(__name__)


def create_dataset_traj(data, threshold=0.0, t_range=0):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    t_range: how far into a sequence to train for
    """
    data_in, data_out = [], []
    for id, sequence in enumerate(data):
        if id % 5 == 0: log.info(f"- processing seq {id}")
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        if id > 99:
            continue
        P = sequence.P
        D = sequence.D
        target = sequence.target
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i + 1, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters

                # The randomely continuing is something I thought of to shrink
                # the datasets while still having a large variety
                if np.random.random() < threshold:
                    continue
                dat = [states[i], j - i]
                dat.extend([P, D])
                dat.append(target)
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
        action, t = policy.act(np.arctan2(observation[j][5:10], observation[j][:5]))

        next_obs, reward, done, info = env.step(action)

        if done:
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


def collect_initial_data(cfg, env):
    """
    Collect data for environment model
    :param env: Reacher3d environment
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []
    for i in range(cfg.initial_num_trials):
        log.info('Trial %d' % i)
        env.seed(i)
        s0 = env.reset()

        P = np.random.rand(5) * 5
        I = np.zeros(5)
        D = np.random.rand(5)

        # Samples target uniformely from [-1, 1]
        target = np.random.rand(5) * 2 - 1

        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)

        dotmap = run_controller(env, horizon=cfg.initial_num_trial_length, policy=policy)

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    return logs


def get_reward(output_state, action):
    '''
    Calculates the reward given output_state action and target
    Uses simple np.linalg.norm between joint positions
    '''
    reward_ctrl = - np.square(action).sum() * 0.01
    reward_dist = - np.linalg.norm(output_state[-3:])
    reward = reward_dist + reward_ctrl
    return reward


def cum_reward(policies, model, initial_obs, horizon):
    '''
    Calculates the cumulative reward of a run with a given policy and target
    :param policy: policy used to get actions
    :param model: model used to estimate dynamics
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = np.zeros(len(policies))
    obs = np.full((len(policies), initial_obs.shape[0]), initial_obs)
    for i in range(horizon):
        big_dat = []
        big_action = []
        for j in range(len(policies)):
            dat = [initial_obs, i + 1]
            dat.extend([policies[j].get_P(), policies[j].get_D()])
            dat.append(policies[j].get_target())
            action, _ = policies[j].act(np.arctan2(obs[j][5:10], obs[j][:5]))
            big_action.append(action)
            big_dat.append(np.hstack(dat))
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat).numpy()
        reward_sum += np.array([get_reward(output_states[j], big_action[j]) for j in range(len(policies))])
        obs = output_states
    return reward_sum


def random_shooting_mpc_pool_helper(params):
    """Helper function used for multiprocessing"""
    num_random_configs, model, obs, horizon, seed = params
    np.random.seed(seed)
    policies = []
    for i in range(num_random_configs):
        P = np.random.rand(5) * 5
        I = np.zeros(5)
        D = np.random.rand(5)
        target_PID = np.random.rand(5) * 2 - 1
        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target_PID)
        policies.append(policy)
    rewards = cum_reward(policies, model, obs, horizon)
    optimal_policy = policies[np.argmax(rewards)]
    return PID(dX=5, dU=5, P=optimal_policy.get_P(), I=np.zeros(5), D=optimal_policy.get_D(),
               target=optimal_policy.get_target()), np.max(rewards)


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
    with Pool(10) as p:
        function_inputs = [(num_random_configs // 10, model, obs, cfg.horizon_traj, i) for i in range(10)]
        out = p.map(random_shooting_mpc_pool_helper, function_inputs)
    return max(out, key=lambda x: x[1])


@hydra.main(config_path='conf/plan.yaml')
def plan(cfg):
    # Evaluation variables
    # reward at beginning of each iteration
    initial_reward = np.zeros(cfg.n_iter)
    # reward at end of each iteration
    final_reward = np.zeros(cfg.n_iter)
    # cumulative reward of trajectory at end of each iteration
    final_cum_reward = np.zeros(cfg.n_iter)

    # Step 1: run random base policy to collect data points
    # Environment setup
    env_model = cfg.env.name
    env = gym.make(env_model)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    log.info('Initializing env: %s' % env_model)
    # collect data through reacher environment
    log.info("Collecting initial data")
    exper_data = collect_initial_data(cfg, env)
    env.seed(cfg.random_seed)

    # Step 2: Learn dynamics model
    # probabilistic model, ensemble training booleans
    prob = cfg.model.prob
    ens = cfg.model.ensemble
    # create dataset for model input/output
    log.info("Creating initial dataset for model training")
    dataset = create_dataset_traj(exper_data,
                                  threshold=0,
                                  t_range=cfg.model.training.t_range)
    # create model object
    model = DynamicsModel(cfg)
    if cfg.load_model:
        label = cfg.env.label
        f = hydra.utils.get_original_cwd() + '/models/' + label + '/'
        model_one = torch.load(f + cfg.step_model + '.dat')
        model = torch.load(f + cfg.traj_model + '.dat')

    # retrain for n_iter iterations
    for i in range(cfg.n_iter):
        log.info(f"Iteration {i}")

        # shuffle dataset each iteration to avoid retraining on the same data points -> overfitting
        shuffle_idxs = np.arange(0, dataset[0].shape[0], 1)
        np.random.shuffle(shuffle_idxs)
        # config for choosing number of points to train on
        if (cfg.num_training_points == 0):
            training_dataset = (dataset[0][shuffle_idxs], dataset[1][shuffle_idxs])
        else:
            training_dataset = (
            dataset[0][shuffle_idxs[:cfg.num_training_points]], dataset[1][shuffle_idxs[:cfg.num_training_points]])
        if not cfg.load_model:
            log.info(f"Training model P:{prob}, E:{ens}")
            # train model
            train_logs, test_logs = model.train(training_dataset, cfg)

        # initial observation
        obs = env.reset()

        initial_reward[i] = get_reward(obs, 0)
        final_cum_reward[i] = initial_reward[i]
        # Code is split between replanning each time step vs. planning a set number of times per iteration
        # REPLAN ONCE EACH TIMESTEP
        if (cfg.num_MPC_per_iter == 0):
            for j in range(cfg.plan_trial_timesteps - 1):
                log.info(f"Trial timestep: {j}")
                # Plan using MPC with random shooting optimizer
                policy, policy_reward = random_shooting_mpc(cfg, model, obs, cfg.horizon_traj)
                # Only use first action from optimal policy
                action, _ = policy.act(np.arctan2(obs[j][5:10], obs[j][:5]))
                # step in environment
                next_obs, reward, done, info = env.step(action)
                if done:
                    break
                # If replanning each time step, using the trajectory model only makes sense without retraining (ie. multiple iterations)
                # We only add state transition dynamics to the dataset for retraining
                dat = [obs.squeeze(), 1]
                dat.extend([policy.get_P() / 5, policy.get_D()])
                dat.append(policy.get_target())
                dataset = (np.append(dataset[0], np.hstack(dat).reshape(1, -1), axis=0),
                           np.append(dataset[1], next_obs.reshape(1, -1), axis=0))
                # Set next observation
                obs = next_obs
                # Calculate the cumulative reward
                # final_cum_reward[i] += reward  # get_reward(obs, action, target_env)
            final_cum_reward[i] += get_reward(obs, action)
        # PLAN A SET NUMBER OF TIMES PER ITERATION
        else:
            # horizon is based on number of times to plan on each trajectory
            horizon = int(cfg.plan_trial_timesteps / cfg.num_MPC_per_iter)
            for j in range(cfg.num_MPC_per_iter):
                # plan using MPC with random shooting optimizer
                policy, policy_reward = random_shooting_mpc(cfg, model, obs, horizon)

                # collect data on trajectory
                logs = DotMap()
                logs.states = []
                logs.actions = []
                logs.rewards = []
                logs.times = []

                # collect data on planned optimal policy
                logs.target = policy.get_target()
                logs.P = policy.get_P() / 5
                logs.I = np.zeros(5)
                logs.D = policy.get_D()

                for k in range(horizon):
                    # step in environment
                    action, _ = policy.act(np.arctan2(obs[j][5:10], obs[j][:5]))
                    next_obs, reward, done, info = env.step(action)
                    if done:
                        break

                    # collect data on trajectory
                    logs.actions.append(action)
                    logs.rewards.append(reward)
                    logs.states.append(obs.squeeze())
                    # set next observation
                    obs = next_obs
                    # don't collect last observation, does not make sense when creating dataset
                    if (j == cfg.num_MPC_per_iter - 1 and k == horizon - 1):
                        continue
                    final_cum_reward[i] += get_reward(obs, action)

                logs.actions = np.array(logs.actions)
                logs.rewards = np.array(logs.rewards)
                logs.states = np.array(logs.states)

                if not cfg.load_model:
                    data_in, data_out = create_dataset_traj(logs,
                                                            threshold=0,
                                                            t_range=cfg.model.training.t_range)
                    dataset = (np.append(dataset[0], data_in, axis=0), np.append(dataset[1], data_out, axis=0))
                if done:
                    break
        # Calculate the final reward
        final_reward[i] = get_reward(obs, 0)

        if (cfg.num_MPC_per_iter == 0):
            final_cum_reward[i] = final_cum_reward[i] / (cfg.plan_trial_timesteps)
        else:
            final_cum_reward[i] = final_cum_reward[i] / cfg.plan_trial_timesteps

        # log.info(f"Initial rewards: {initial_reward}")
        # log.info(f"Final rewards: {final_reward}")
        log.info(f"Final cumulative rewards: {final_cum_reward[i]}")

    log.info(f"Mean cumrew: {np.mean(final_cum_reward)}")
    log.info(f"stddev cumrew: {np.std(final_cum_reward)}")

if __name__ == '__main__':
    sys.exit(plan())
