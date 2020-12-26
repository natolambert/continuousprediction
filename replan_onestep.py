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


def create_dataset_step(data, delta=True, t_range=0, is_lstm=False, lstm_batch=0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: List of DotMaps, each dotmap is a trajectory
    delta: Whether model predicts change in state or next state.
    t_range: How far into the trajectory to train on
    is_lstm: whether or not we are training with an lstm network
    lstm_batch: size of batch for lstm

    Notes:
      - No PID parameters in the outputted dataset
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        if t_range > 0:
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
            remainder = len(data_out) % lstm_batch
            if remainder:
                data_out = data_out[:len(data_out) - remainder]
                data_in = data_in[:len(data_in) - remainder]
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out


def obs2q(obs):
    """
    Helper function that returns the first five values in obs
    :param obs: the 21 length observation array
    :returns: the first 5 values (the cosine of joint positions)
    """
    if len(obs) < 5:
        return obs
    else:
        return obs[0:5]


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
        action, t = policy.act(obs2q(observation))

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
    for i in range(cfg.initial_num_trials):  # Generates initial_num_trials trajectories
        log.info('Trial %d' % i)
        env.seed(i)
        s0 = env.reset()

        P = np.random.rand(5) * 5
        I = np.zeros(5)
        D = np.random.rand(5)

        # Samples target uniformely from [-1, 1]
        target = np.random.rand(5) * 2 - 1

        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)

        dotmap = run_controller(env, horizon=cfg.plan_trial_timesteps, policy=policy)
        # Runs PID controller to generate a trajectory with horizon plan_trial_timesteps

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    return logs


def get_reward(output_state, action, target):
    '''
    Calculates the reward given output_state action and target
    Uses simple np.linalg.norm between joint positions
    '''
    reward_ctrl = - np.square(action).sum() * 0.01
    theta = [np.arctan2(output_state[5:10], output_state[:5])]
    reward_dist = - np.linalg.norm(theta - target)
    reward = reward_dist + reward_ctrl
    return reward


def cum_reward(action_seq, model, target, obs, horizon):
    '''
    NOT UPDATED WITH PID_PLAN, USE CUM_REWARD_STACKED() FUNCTION BELOW
    Calculates the cumulative reward of a run with a given policy and target
    :param action_seq: sequence of actions, length: horizon
    :param model: model used to estimate dynamics
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = 0
    for i in range(horizon):
        data_in = np.hstack((obs, action_seq[i]))
        # added variable to only reform dataset with selected state indices if on first pass
        output_state = model.predict(np.array(data_in)[None], reform=(i == 0))[0].numpy()
        this_reward = get_reward(output_state, action_seq[i], target)
        reward_sum += this_reward
        obs = output_state
    return reward_sum


def cum_reward_stacked(policies, model, target, obs, horizon, PID_plan):
    '''
    Parallelizes cum_reward by concatenating inputs along batch dimension
    Calculates the cumulative reward of a run with a given policy and target
    :param policies: sequence of actions, length: horizon
    :param model: model used to estimate dynamics
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    :param PID_plan: use random PID parameters, instead of random actions to plan
    '''
    reward_sum = np.zeros(len(policies))
    obs = np.full((len(policies), obs.shape[0]), obs)
    for i in range(horizon):
        big_dat = []
        big_action = []
        for j in range(len(policies)):
            if (PID_plan):
                action, _ = policies[j].act(obs2q(obs[j]))
            else:
                action = policies[j][i]
            dat = np.hstack((obs[j], action))
            big_action.append(action)
            # print(np.array([np.hstack(dat)]).shape)
            big_dat.append(dat)
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat, reform=(i == 0)).numpy()
        reward_sum += np.array([get_reward(output_states[k], big_action[k],target) for k in range(len(policies))])
        obs = output_states
    return reward_sum


def random_shooting_mpc_pool_helper(params):
    """Helper function used for multiprocessing"""
    num_random_configs, horizon, model, target, obs, PID_plan, seed = params
    np.random.seed(seed)
    policies = []
    rewards = np.array([])
    for i in range(num_random_configs):
        if (PID_plan):
            P = np.random.rand(5) * 5
            I = np.zeros(5)
            D = np.random.rand(5)
            target_PID = np.random.rand(10) * 2 - 1
            target_PID = np.random.rand(5) * 2 - 1

            policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target_PID)
            policies.append(policy)
        else:
            action_seq = (np.random.rand(horizon, 5) - 0.5) * 2
            policies.append(action_seq)
        # Uncomment this line if not using stacked
        # rewards = np.append(rewards, cum_reward(action_seq, model, target, obs, horizon))
    rewards = cum_reward_stacked(policies, model, target, obs, horizon, PID_plan)
    optimal_policy = policies[np.argmax(rewards)]
    if (PID_plan):
        return PID(dX=5, dU=5, P=optimal_policy.get_P(), I=np.zeros(5), D=optimal_policy.get_D(),
                   target=optimal_policy.get_target()), np.max(rewards)
    else:
        return policies[np.argmax(rewards)], np.max(rewards)


def random_shooting_mpc(cfg, target, model, obs, horizon):
    '''
    Creates random action configurations and returns the one with the best cumulative reward
    :param target: target to aim for
    :param model: model to use to predict dynamics
    :param obs: observation to start calculating from
    :param horizon: how far to plan for
    :return: the policy that has the best cumulative reward, and the best cumulative reward (for evaluation)
    '''
    num_parallel_threads = 10
    num_random_configs = cfg.num_random_configs
    if num_random_configs % num_parallel_threads != 0:
        raise ValueError("Number of Parallel Threads must perfectly divide Num Random Configs")

    from multiprocessing import Pool
    with Pool(num_parallel_threads) as p:
        function_inputs = [(num_random_configs // num_parallel_threads, horizon, model, target, obs, cfg.PID_plan, i)
                           for i in range(num_parallel_threads)]
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
    # environment setup
    env_model = cfg.env.name
    log.info('Initializing env: %s' % env_model)
    env = gym.make(env_model)
    env.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    # get a random target to work towards
    # cosines and sines of 5 joint angles, length 10 vector
    # target = np.random.rand(10) * 2 - 1

    # log.info(f"Planning towards target: {target}")
    # collect data through reacher environment
    log.info("Collecting initial data")
    exper_data = collect_initial_data(cfg, env)

    # Step 2: Learn dynamics model
    # probabilistic model, ensemble training booleans
    prob = cfg.model.prob
    ens = cfg.model.ensemble
    # create dataset for model input/output
    log.info("Creating initial dataset for model training")
    dataset = create_dataset_step(exper_data,
                                  t_range=cfg.model.training.t_range)
    # create model
    model = DynamicsModel(cfg)
    if cfg.load_model:
        label = cfg.env.label
        f = hydra.utils.get_original_cwd() + '/models/' + label + '/'
        model = torch.load(f + cfg.step_model + '.dat')

    # run for n_iter iterations
    for i in range(cfg.n_iter):
        log.info(f"Iteration {i}")

        # shuffle data to avoid retraining on old data -> overfitting
        shuffle_idxs = np.arange(0, dataset[0].shape[0], 1)
        np.random.shuffle(shuffle_idxs)
        # setting how many training points to train on each iteration
        if (cfg.num_training_points == 0):
            training_dataset = (dataset[0][shuffle_idxs], dataset[1][shuffle_idxs])
        else:
            training_dataset = (
            dataset[0][shuffle_idxs[:cfg.num_training_points]], dataset[1][shuffle_idxs[:cfg.num_training_points]])

        if not cfg.load_model:
            log.info(f"Training model P:{prob}, E:{ens}")
            # train model
            train_logs, test_logs = model.train(training_dataset, cfg)
        # get initial observation
        obs = env.reset()
        target_env = np.random.rand(5) * 2 - 1

        initial_reward[i] = get_reward(obs, 0, target_env)
        # log.info(f"Initial rewards: {initial_reward}")
        final_cum_reward[i] = initial_reward[i]
        # split into two parts, replan each timestep vs. plan once per iteration
        # REPLAN EACH TIMESTEP
        if (cfg.num_MPC_per_iter == 0):
            for j in range(cfg.plan_trial_timesteps - cfg.horizon_step - 1):
                log.info(f"Trial timestep: {j}")
                # plan using MPC with random shooting optimizer
                action_seq, policy_reward = random_shooting_mpc(cfg, target_env, model, obs, cfg.horizon_step)
                # get first action from optimal policy
                if cfg.PID_plan:
                    action, _ = action_seq.act(obs2q(obs))
                else:
                    action = action_seq[0]
                # step in environment
                next_obs, reward, done, info = env.step(action)
                if done:
                    break
                # state transition dynamics to dataset for retraining next iteration
                data_in = np.hstack((obs, action)).reshape(1, -1)
                data_out = (next_obs - obs).reshape(1, -1)
                dataset = (np.append(dataset[0], data_in.reshape(1, -1), axis=0),
                           np.append(dataset[1], data_out.reshape(1, -1), axis=0))
                # set next observation
                obs = next_obs
                # calculate final cumulative reward
                final_cum_reward[i] += reward  # get_reward(obs, action, target_env)
        else:
            # PLAN ONCE EACH ITERATION
            # horizon = number of timesteps of trajectory
            horizon = int(cfg.plan_trial_timesteps / cfg.num_MPC_per_iter)
            for j in range(cfg.num_MPC_per_iter):
                action_seq, policy_reward = random_shooting_mpc(cfg, target_env, model, obs, horizon)
                # prepare to log trajectory data
                logs = DotMap()
                logs.states = []
                logs.actions = []
                logs.rewards = []
                logs.times = []

                for k in range(horizon):
                    # get action from optimal policy
                    if cfg.PID_plan:
                        action, _ = action_seq.act(obs2q(obs))
                    else:
                        action = action_seq[k]
                    # step in environment
                    next_obs, reward, done, info = env.step(action)
                    if done:
                        break
                    # collect trajectory data
                    logs.actions.append(action)
                    logs.rewards.append(reward)
                    logs.states.append(obs.squeeze())
                    # set next observation
                    obs = next_obs
                    if (j == cfg.num_MPC_per_iter - 1 and k == horizon - 1):
                        continue
                    # calculate cumulative reward
                    final_cum_reward[i] += get_reward(obs, action, target_env)

                logs.actions = np.array(logs.actions)
                logs.rewards = np.array(logs.rewards)
                logs.states = np.array(logs.states)

                if not cfg.load_model:
                    data_in, data_out = create_dataset_step(logs,
                                                        t_range=cfg.model.training.t_range)
                    # dataset = (data_in, data_out)
                    dataset = (np.append(dataset[0], data_in, axis=0), np.append(dataset[1], data_out, axis=0))
                if done:
                    break

        final_reward[i] = get_reward(obs, 0, target_env)
        final_cum_reward[i] = final_cum_reward[i] / (cfg.plan_trial_timesteps - cfg.horizon_step)
        # log.info(f"Final rewards: {final_reward}")
        log.info(f"Final cumulative rewards: {final_cum_reward[i]}")

    log.info(f"Mean cumrew: {np.mean(final_cum_reward)}")
    log.info(f"stddev cumrew: {np.std(final_cum_reward)}")

if __name__ == '__main__':
    sys.exit(plan())
