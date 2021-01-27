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
#import mujoco_py
import gym
from envs import *
from policy import PID
from policy import LQR
from dynamics_model import DynamicsModel
import torch

log = logging.getLogger(__name__)


def create_dataset_traj(data, threshold=0.0, t_range=0, env_label="reacher"):
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
        if (env_label == "reacher"):
            P = sequence.P
            D = sequence.D
            target = sequence.target
        elif (env_label == "cartpole"):
            K = sequence.K
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
                if (env_label == "reacher"):
                    dat.extend([P, D])
                    dat.append(target)
                elif (env_label == "cartpole"):
                    dat.append(K)
                data_in.append(np.hstack(dat))
                data_out.append(states[j])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)
    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: List of DotMaps, each dotmap is a trajectory
    delta: Whether model predicts change in state or next state.
    t_range: How far into the trajectory to train on

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
        if (len(observation) < 5):
            action, t = policy.act(observation)
        else:
            action, t = policy.act(np.arctan2(observation[5:10], observation[:5]))

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


def collect_data_lqr(cfg, env):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    s = np.random.randint(0, 100)
    for i in range(cfg.initial_num_trials):
        log.info('Trial %d' % i)
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
            lim = cfg.initial_num_trial_length
        elif cfg.data_mode == 'unstable':
            modifier = 1.5 * np.random.random(4) - .75
            env.theta_threshold_radians = 2 * env.theta_threshold_radians
            # default 2.4
            env.x_threshold = 2 * env.x_threshold
            lim = cfg.initial_num_trial_length
        else:
            modifier = .5 * np.random.random(4) + 1
            lim = cfg.initial_num_trial_length
        policy = LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0])
        policy.K = np.multiply(policy.K, modifier)
        # print(type(env))
        dotmap = run_controller(env, horizon=cfg.initial_num_trial_length, policy=policy)
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
            dotmap = run_controller(env, horizon=cfg.initial_num_trial_length, policy=policy)
            print(f"- Repeat simulation")
            s += 1

        dotmap.K = np.array(policy.K).flatten()
        logs.append(dotmap)
        s += 1

    return logs


def collect_data_reacher(cfg, env):
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


def get_reward_cartpole(output_state):
    # x_threshold = 9.6
    # theta_threshold_radians = .418879  # 24 * 2 * math.pi / 360
    # x = output_state[0]
    # theta = output_state[2]
    # done = bool(x < -self.x_threshold \
    #             or x > self.x_threshold \
    #             or theta < -self.theta_threshold_radians \
    #             or theta > self.theta_threshold_radians)
    # return 1 if done else 0
    reward = output_state[0] ** 2 + output_state[2] ** 2
    return -reward/200


def get_reward(output_state, action):
    '''
    Calculates the reward given output_state action and target
    Uses simple np.linalg.norm between joint positions
    '''
    reward_ctrl = - np.square(action).sum() * 0.01
    reward_dist = - np.linalg.norm(output_state[-3:])
    reward = reward_dist + reward_ctrl
    return reward/500


def cum_reward(policies, model, initial_obs, horizon, traj, action_plan, env_label):
    '''
    Calculates the cumulative reward of a run with a given policy and target
    :param policy: policy used to get actions
    :param model: model used to estimate dynamics
    :param initial_obs: observation to start calculating from
    :param horizon: number of time steps to calculate for
    '''
    reward_sum = np.zeros(len(policies))
    obs = np.full((len(policies), initial_obs.shape[0]), initial_obs)
    for i in range(horizon):
        big_dat = []
        big_action = []
        for j in range(len(policies)):
            if (action_plan):
                action = policies[j][i]
            else:
                if (env_label == "reacher"):
                    action, _ = policies[j].act(np.arctan2(obs[j][5:10], obs[j][:5]))
                else:
                    action, _ = policies[j].act(obs[j])

            if (traj):
                dat = [initial_obs, i + 1]
                if (env_label == "reacher"):
                    dat.extend([policies[j].get_P() / 5, policies[j].get_D()])
                    dat.append(policies[j].get_target())
                elif (env_label == "cartpole"):
                    dat.append(policies[j].get_K())
                big_dat.append(np.hstack(dat))
            else:
                dat = np.hstack((obs[j], action))
                big_dat.append(dat)
            big_action.append(action)
        big_dat = np.vstack(big_dat)
        if (traj):
            output_states = model.predict(big_dat).numpy()
        else:
            output_states = model.predict(big_dat, reform=(i == 0)).numpy()
        if (env_label == "reacher"):
            reward_sum += np.array([get_reward(output_states[j], big_action[j]) for j in range(len(policies))])
        elif (env_label == "cartpole"):
            reward_sum += np.array([get_reward_cartpole(output_states[j]) for j in range(len(policies))])
        obs = output_states
    return reward_sum

def cum_reward_stacked(policies, model, obs, horizon, action_plan, env_label):
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
            if not action_plan: # np.arctan2(obs[j][5:10], obs[j][:5])
                # action, _ = policies[j].act(obs2q(obs[j]))
                if (env_label == "reacher"):
                    action, _ = policies[j].act(np.arctan2(obs[j][5:10], obs[j][:5]))
                else:
                    action, _ = policies[j].act(obs[j])
            else:
                action = policies[j][i]
            dat = np.hstack((obs[j], action))
            big_action.append(action)
            # print(np.array([np.hstack(dat)]).shape)
            big_dat.append(dat)
        big_dat = np.vstack(big_dat)
        output_states = model.predict(big_dat, reform=(i == 0)).numpy()
        reward_sum += np.array([get_reward(output_states[k], big_action[k]) for k in range(len(policies))])
        obs = output_states
    return reward_sum

def random_shooting_mpc_pool_helper(params):
    """Helper function used for multiprocessing"""
    num_random_configs, model, obs, horizon, traj, action_plan, env_label, seed = params
    LQR_optimal = np.array([-0.70710678, -4.2906244, -37.45394052, -8.06650137])
    np.random.seed(seed)
    policies = []
    for i in range(num_random_configs):
        if (action_plan):
            if (env_label == "reacher"):
                # nol changed below so actions are in correct region (-1,1)
                action_seq = np.random.rand(horizon, 5) * 2 - 1
            elif (env_label == "cartpole"):
                action_seq = np.random.rand(horizon, 1) * 2 - 1
            policies.append(action_seq)
        else:
            if (env_label == "reacher"):
                P = np.random.rand(5) * 5
                I = np.zeros(5)
                D = np.random.rand(5)
                target_PID = np.random.rand(5) * 2 - 1
                policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target_PID)
            elif (env_label == "cartpole"):
                K = np.random.uniform(LQR_optimal * 1.5, LQR_optimal * .5)
                policy = LQR(np.zeros((4, 4)), np.zeros((4, 1)), np.zeros(4), 0, K=K)
            policies.append(policy)
    if traj:
        rewards = cum_reward(policies, model, obs, horizon, traj, action_plan, env_label)
    else:
        rewards = cum_reward_stacked(policies, model, obs, horizon, action_plan, env_label)
    optimal_policy = policies[np.argmax(rewards)]
    if (action_plan or env_label == "cartpole"):
        return policies[np.argmax(rewards)], np.max(rewards)
    else:
        if env_label == "reacher":
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
        function_inputs = [(num_random_configs // 10, model, obs, horizon, cfg.model.traj, cfg.action_plan,
                            cfg.env.label, i) for i in range(10)]
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
    env_label = cfg.env.label
    env = gym.make(env_model)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    env.seed(cfg.random_seed)

    traj = cfg.model.traj
    prob = cfg.model.prob
    ens = cfg.model.ensemble

    log.info('Initializing env: %s' % env_model)
    # collect data through reacher environment
    if cfg.load_model:
        f = hydra.utils.get_original_cwd() + '/models/' + env_label + '/'
        model = torch.load(f + cfg.traj_model + '.dat' if traj else f+ cfg.step_model + '.dat')
    else:
        log.info("Collecting initial data")

        if (env_label == "reacher"):
            exper_data = collect_data_reacher(cfg, env)
        elif (env_label == "cartpole"):
            exper_data = collect_data_lqr(cfg, env)

        # Step 2: Learn dynamics model
        # probabilistic model, ensemble training booleans
        # create dataset for model input/output
        log.info("Creating initial dataset for model training")
        model = DynamicsModel(cfg)
        if (traj):
            dataset = create_dataset_traj(exper_data,
                                          threshold=0,
                                          t_range=cfg.model.training.t_range,
                                          env_label=env_label)
        else:
            dataset = create_dataset_step(exper_data,
                                          t_range=cfg.model.training.t_range)

    # retrain for n_iter iterations
    rews = np.zeros(cfg.n_iter)
    for i in range(cfg.n_iter):
        log.info(f"Iteration {i}")

        if not cfg.load_model:
            if i == 0 or cfg.retrain_model:
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

        # initial observation
        obs = env.reset()
        if (env_label == "reacher"):
            log.info(f"Reacher goal {env.goal}")
            initial_reward[i] = get_reward(obs, 0)
            final_cum_reward[i] = initial_reward[i]

        # Code is split between replanning each time step vs. planning a set number of times per iteration
        # REPLAN ONCE EACH TIMESTEP
        if (cfg.num_MPC_per_iter == 0):
            for j in range(cfg.trial_timesteps - 1):
                if (j%50 == 0): log.info(f"Trial timestep: {j}")
                # Plan using MPC with random shooting optimizer

                policy, policy_reward = random_shooting_mpc(cfg, model, obs,
                                                            cfg.horizon_traj if traj else cfg.horizon_step)
                # Only use first action from optimal policy
                if (cfg.action_plan):
                    action = policy[0]
                else:
                    if (env_label == "reacher"):
                        print(np.arctan2(obs[5:10], obs[:5]))
                        action, _ = policy.act(np.arctan2(obs[5:10], obs[:5]))
                    elif (env_label == "cartpole"):
                        action, _ = policy.act(obs)
                    action = np.clip(action, -1, 1)

                # step in environment
                next_obs, reward, done, info = env.step(action)
                rews[i] += reward/cfg.trial_timesteps
                if done:
                    break
                # If replanning each time step, using the trajectory model only makes sense without retraining (ie. multiple iterations)
                # We only add state transition dynamics to the dataset for retraining
                if not cfg.load_model and cfg.retrain_model:
                    if (traj):
                        dat = [obs.squeeze(), 1]
                        if (env_label == "reacher"):
                            dat.extend([policy.get_P() / 5, policy.get_D()])
                            dat.append(policy.get_target())
                        elif (env_label == "cartpole"):
                            dat.append(policy.get_K())
                        data_in = np.hstack(dat).reshape(1, -1)
                        data_out = next_obs.reshape(1, -1)
                    else:
                        data_in = np.hstack((obs, action)).reshape(1, -1)
                        data_out = (next_obs - obs).reshape(1, -1)
                    dataset = (np.append(dataset[0], data_in, axis=0),
                               np.append(dataset[1], data_out, axis=0))
                # Set next observation
                obs = next_obs
                # Calculate the cumulative reward
                # final_cum_reward[i] += reward  # get_reward(obs, action, target_env)
            final_cum_reward[i] += get_reward(obs, action)
        # PLAN A SET NUMBER OF TIMES PER ITERATION
        else:
            # horizon is based on number of times to plan on each trajectory
            horizon = int(cfg.trial_timesteps / cfg.num_MPC_per_iter)
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
                if not cfg.action_plan:
                    if (env_label == "reacher"):
                        logs.target = policy.get_target()
                        logs.P = policy.get_P() / 5
                        logs.I = np.zeros(5)
                        logs.D = policy.get_D()
                    elif (env_label == "cartpole"):
                        logs.K = policy.get_K()

                for k in range(horizon):
                    # step in environment
                    if (cfg.action_plan):
                        action = policy[k]
                    else:
                        if (env_label == "reacher"):
                            action, _ = policy.act(np.arctan2(obs[5:10], obs[:5]))
                        else:
                            action, _ = policy.act(obs)
                        action = np.clip(action, -1, 1)

                    next_obs, reward, done, info = env.step(action)
                    rews[i] += reward/cfg.trial_timesteps

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

                if not cfg.load_model and cfg.retrain_model:
                    if (traj):
                        data_in, data_out = create_dataset_traj(logs,
                                                                threshold=0,
                                                                t_range=cfg.model.training.t_range,
                                                                env_label=env_label)
                    else:
                        data_in, data_out = create_dataset_step(logs,
                                                                t_range=cfg.model.training.t_range)
                    dataset = (np.append(dataset[0], data_in, axis=0), np.append(dataset[1], data_out, axis=0))
                if done:
                    break


        log.info(f"Final cumulative rewards: {rews[i]}")

    log.info(f"Mean cumrew: {np.mean(rews)}")
    log.info(f"stddev cumrew: {np.std(rews)}")


if __name__ == '__main__':
    sys.exit(plan())
