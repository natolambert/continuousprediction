''' TODO:
    1. get_reward uses arccos(), but output state is sometimes out of the domain
        sign of inaccurate predictions? or postprocessing scaling?
        min: -1.7, max: 1.4 -> right now, i've just clipped the values to 1 and -1

        Changed to just use the target (might make more sense because of obs2q()->
        PID parameters work on first five values of obs, which means target is cosine of joint positions)

        Added PID policy action to the reward calculation, but cannot match the reward in environment file perfectly,
        since we cannot get the 3d vector of fingertip
    2. does having the target remain the same through all MPC runs make sense?
    3. evaluate, have some results to look at
        does using the MPC rewards as a way to evaluate make sense?
        maybe also include prediction error using test_data? (it's commented out right now)
        what does hw 4 cs285 use for eval_return?
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

        dotmap = run_controller(env, horizon=cfg.plan_trial_timesteps, policy=policy)

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    return logs

def get_reward(output_state, target, action):
    '''
    Calculates the reward given output_state and target
    Uses simple np.linalg.norm between joint positions
    '''
    reward_ctrl = - np.square(action).sum() * 0.01
    reward_dist = - np.linalg.norm(obs2q(output_state) - target)
    reward = reward_dist + reward_ctrl
    return reward

def cum_reward(policies, model, target, obs, horizon):
    '''
    Calculates the cumulative reward of a run with a given policy and target
    :param policy: policy used to get actions
    :param model: model used to estimate dynamics
    :param target: target used to estimate the reward
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = np.zeros(len(policies))
    for i in range(horizon):
        big_dat = []
        big_action = []
        for policy in policies:
            dat = [obs, i+1]
            dat.extend([policy.get_P(), policy.get_D()])
            dat.append(target)
            action, _ = policy.act(obs2q(obs))
            big_action.append(action)
            # print(np.array([np.hstack(dat)]).shape)
            big_dat.append(np.hstack(dat))
        big_dat = np.vstack(big_dat)
        # print(big_dat.shape)
        output_states = model.predict(big_dat).numpy()
        reward_sum += np.array([get_reward(output_states[j], target, big_action[j]) for j in range(len(policies))])
    return reward_sum

def random_shooting_mpc_pool_helper(params):
    """Helper function used for multiprocessing"""
    num_random_configs, target, model, obs, horizon, seed = params
    np.random.seed(seed)
    policies = []
    for i in range(num_random_configs):
        P = np.random.rand(5) * 5
        I = np.zeros(5)
        D = np.random.rand(5)
        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
        policies.append(policy)
    rewards = cum_reward(policies, model, target, obs, horizon)
    optimal_policy = policies[np.argmax(rewards)]
    return PID(dX=5, dU=5, P=optimal_policy.get_P(), I=np.zeros(5), D=optimal_policy.get_D(), target=target), np.max(rewards)

def random_shooting_mpc(cfg, target, model, obs, horizon):
    '''
    Creates random PID configurations and returns the one with the best cumulative reward
    :param target: target to aim for
    :param model: model to use to predict dynamics
    :param obs: observation to start calculating from
    :return: the PID policy that has the best cumulative reward, and the best cumulative reward (for evaluation)
    '''
    from multiprocessing import Pool
    num_random_configs = cfg.num_random_configs
    with Pool(10) as p:
        function_inputs = [(num_random_configs//10, target, model, obs, cfg.horizon, i) for i in range(10)]
        out = p.map(random_shooting_mpc_pool_helper, function_inputs)
    return max(out, key=lambda x:x[1])

@hydra.main(config_path='conf/plan.yaml')
def plan(cfg):
    # Following http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf
    # model-based reinforcement learning version 1.5

    # Evaluation variables
    # reward at beginning of each iteration
    initial_reward = np.zeros(cfg.n_iter)
    # reward at end of each iteration
    final_reward = np.zeros(cfg.n_iter)
    # cumulative reward of trajectory at end of each iteration
    final_cum_reward = np.zeros(cfg.n_iter)

    # VD: Added empty lists because they aren't present
    data_in = []
    data_out = []

    # Step 1: run random base policy to collect data points
    # get a target to work towards, training is still done on random targets to not affect exploration

    #target = np.random.rand(5) * 2 - 1
    target = np.array([0.46567452, -0.95595055, 0.67755277, 0.56301844, 0.93220489])

    log.info(f"Planning towards target: {target}")
    # collect data through reacher environment
    log.info("Collecting initial data")
    env_model = cfg.env.name
    env = gym.make(env_model)
    log.info('Initializing env: %s' % env_model)
    exper_data = collect_initial_data(cfg, env)
    #test_data = collect_initial_data(cfg, env)

    # Step 2: Learn dynamics model
    # probabilistic model, ensemble training booleans
    prob = cfg.model.prob
    ens = cfg.model.ensemble
    # create dataset for model input/output
    log.info("Creating initial dataset for model training")
    dataset = create_dataset_traj(exper_data,
                                  threshold=cfg.model.training.filter_rate,
                                  t_range=cfg.model.training.t_range)
    # create and train model
    model = DynamicsModel(cfg)
    for i in range(cfg.n_iter):
        log.info(f"Training iteration {i}")
        log.info(f"Training model P:{prob}, E:{ens}")

        shuffle_idxs = np.arange(0, dataset[0].shape[0], 1)
        np.random.shuffle(shuffle_idxs)
        dataset = (dataset[0][shuffle_idxs], dataset[1][shuffle_idxs])

        train_logs, test_logs = model.train(dataset, cfg)
        obs = env.reset()
        print("Initial observation: " + str(obs))
        initial_reward[i] = get_reward(obs, target, 0)
        final_cum_reward[i] = initial_reward[i]
        # RUN MPC ONCE EACH TIMESTEP
        if(cfg.num_MPC_per_iter == 0):
            for j in range(cfg.plan_trial_timesteps-cfg.horizon-1):
                log.info(f"Trial timestep: {j}")
                # Step 3: Plan
                # Moved obs assignment to the end of the loop
                policy, policy_reward = random_shooting_mpc(cfg, target, model, obs, cfg.horizon)
                action, _ = policy.act(obs2q(obs))
                next_obs, reward, done, info = env.step(action)
                if done:
                    break
                '''
                TODO: What to add to the dataset here?
                Running policy for multiple steps?
                For now, just append one step (transition dynamics)
                '''
                dat = [obs.squeeze(), 1]
                dat.extend([policy.get_P() / 5, policy.get_D()])
                dat.append(target)
                dataset = (np.append(dataset[0],np.hstack(dat).reshape(1,-1), axis=0), np.append(dataset[1],next_obs.reshape(1,-1), axis=0))
                obs = next_obs
                final_cum_reward[i] += get_reward(obs, target, action)
        # RUN MPC SOME NUMBER OF TIMES IN AN ITERATION
        else:
            horizon = int(cfg.plan_trial_timesteps/cfg.num_MPC_per_iter)
            for j in range(cfg.num_MPC_per_iter):
                policy, policy_reward = random_shooting_mpc(cfg, target, model, obs, horizon)
                initial_obs = obs

                logs = DotMap()
                logs.states = []
                logs.actions = []
                logs.rewards = []
                logs.times = []

                logs.target = target
                logs.P = policy.get_P() / 5
                logs.I = np.zeros(5)
                logs.D = policy.get_D()

                for k in range(horizon):
                    action, _ = policy.act(obs2q(obs))
                    next_obs, reward, done, info = env.step(action)
                    if done:
                        break

                    logs.actions.append(action)
                    logs.rewards.append(reward)
                    logs.states.append(obs.squeeze())
                    obs = next_obs
                    if (j == cfg.num_MPC_per_iter-1 and k == horizon-1):
                        continue
                    final_cum_reward[i] += get_reward(obs, target, action)

                logs.actions = np.array(logs.actions)
                logs.rewards = np.array(logs.rewards)
                logs.states = np.array(logs.states)

                data_in, data_out = create_dataset_traj(exper_data,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
                dataset = (np.append(dataset[0], data_in, axis=0), np.append(dataset[1],data_out, axis=0))
                if done:
                    break
        final_reward[i] = get_reward(obs, target, 0)

        if (cfg.num_MPC_per_iter == 0):
            final_cum_reward[i] = final_cum_reward[i]/(cfg.plan_trial_timesteps-cfg.horizon)
        else:
            final_cum_reward[i] = final_cum_reward[i]/cfg.plan_trial_timesteps

        log.info(f"Initial rewards: {initial_reward}")
        log.info(f"Final rewards: {final_reward}")
        log.info(f"Final cumulative rewards: {final_cum_reward}")

if __name__ == '__main__':
    sys.exit(plan())
