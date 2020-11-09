''' Questions:
    1. What are we adding to the dataset each training step, just a one step forward data point, or one whole trajectory with the planned policy?
    So are we adding (initial -> output), or (output-1 -> output) or both?
    2. How do we calculate the reward at each time step without the environment?
    3. What happens when we pass the end of the environment simulation (the model will not be accurate anymore) when using MPC? Is the way I am calculating this correct?
    4. Are we using the same target every iteration?
'''

''' TODO:
    1. get_reward function
    2. what to add to the dataset at each planning
    3. Figure out how to evaluate the model at each iteration (for both dynamics model accuracy and control policy)
'''

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

def run_controller(env, horizon, policy):
    """
    Runs a Reacher3d gym environment for horizon timesteps, making actions according to policy

    :param env: A gym object
    :param horizon: The number of states forward to look
    :param policy: A policy object (see other python file)
    """

    # WHat is going on here?
    def obs2q(obs):
        if len(obs) < 5:
            return obs
        else:
            return obs[0:5]

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    for i in range(horizon):
        state = observation
        action, t = policy.act(obs2q(state))

        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
            return logs

        # Log
        logs.actions.append(action)
        logs.rewards.append(reward)
        logs.states.append(observation.squeeze())

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

        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy)

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    return logs

def cum_reward(policy, model, target, obs, horizon):
    '''
    Calculates the cumulative reward of a run with a given policy and target
    :param policy: policy used to get actions
    :param model: model used to estimate dynamics
    :param target: target used to estimate the reward
    :param obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = 0
    for i in range(horizon):
        dat = [obs, i+1]
        dat.extend([policy.get_P(), policy.get_D()])
        dat.append(target)
        output_state = model.predict(dat)
        '''
        TODO: get_reward() NOT WRITTEN
        probably dependent on output_state and target
        simple np.linalg.norm distance?
        '''
        reward_sum += get_reward(output_state, target)
    return reward_sum



def random_shooting_mpc(cfg, target, model, obs):
    '''
    Creates random PID configurations and returns the one with the best cumulative reward
    :param target: target to aim for
    :param model: model to use to predict dynamics
    :param obs: observation to start calculating from
    :return: the PID policy that has the best cumulative reward
    '''
    num_random_configs = cfg.num_random_configs
    policies = []
    rewards = np.array([])
    for i in range(num_random_configs):
        P = np.random.rand(5) * 5
        I = np.zeros(5)
        D = np.random.rand(5)
        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
        policies.append(policy)
        rewards = np.append(rewards, cum_reward(policy, model, target, obs, cfg.horizon))
    return policies[np.argmax(rewards)]

@hydra.main(config_path='conf/plan.yaml')
def plan(cfg):
    # Following http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf
    # model-based reinforcement learning version 1.5

    # Step 1: run random base policy to collect data points
    # get a target to work towards, training is still done on random targets to not affect exploration
    target = np.random.rand(5) * 2 - 1
    log.info(f"Planning towards target: {target}")
    # collect data through reacher environment
    log.info("Collecting initial data")
    env_model = cfg.env.name
    env = gym.make(env_model)
    log.info('Initializing env: %s' % env_model)
    exper_data = collect_data(cfg, env)
    test_data = collect_data(cfg, env)

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
        log.info(f"Training model P:{prob}, E:{ens}")
        train_logs, test_logs = model.train(dataset, cfg)
        obs = env.reset()
        for j in range(cfg.trial_timesteps-cfg.horizon-1):
            # Step 3: Plan
            policy = random_shooting_mpc(cfg, target, model, obs)
            action, _ = policy.act(obs)
            observation, reward, done, info = env.step(action)
            if done:
                break
            '''
            TODO: What to add to the dataset here?
            Running policy for multiple steps?
            '''

if __name__ == '__main__':
    sys.exit(plan())
