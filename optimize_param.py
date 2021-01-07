"""
The purpose of this file is to load in different trajectories and compare how model types predict control performance.
There are three options:
- Gausian process mapping from control policy and s0 to reward
- One step model rolling out predicted trajectory from initial state, cumulative reward
- trajectory model predicting trajectory and cumulative reward
"""

import sys

import hydra
import logging
import itertools

import gym
import torch
import numpy as np
import cma

from plot import *
from evaluate import test_models
import gpytorch
from mbrl_resources import obs2q
from policy import LQR
from reacher_pd import run_controller

from ax import (
    ComparisonOp,
    ParameterType, Parameter, RangeParameter, ChoiceParameter,
    FixedParameter, OutcomeConstraint, SimpleExperiment, Models,
    Arm, Metric, Runner, OptimizationConfig, Objective, Data,
    SearchSpace
)
from ax.plot.trace import optimization_trace_single_method
from ax.plot.contour import plot_contour

import numpy as np
import pandas as pd
import plotly

import os
import sys
import hydra
import logging

# import plotly.graph_objects as go
# from plot import save_fig, plot_learning

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)


def gen_search_space(cfg):
    l = []
    for key, item in cfg.space.items():
        if item.value_type == 'float':
            typ = ParameterType.FLOAT
        elif item.value_type == 'int':
            typ = ParameterType.INT
        elif item.value_type == 'bool':
            typ == Parameter.BOOL
        else:
            raise ValueError("invalid search space value type")

        if item.type == 'range':
            ss = RangeParameter(
                name=key, parameter_type=typ, lower=item.bounds[0], upper=item.bounds[1], log_scale=item.log_scale,
            )
        elif item.type == 'fixed':
            ss = FixedParameter(name=key, value=item.bounds, parameter_type=typ)
        elif item.type == 'choice':
            ss = ChoiceParameter(name=key, parameter_type=typ, values=item.bounds)
        else:
            raise ValueError("invalid search space parameter type")
        l.append(ss)
    return l


def get_reward_reacher(state, action):
    # Copied from the reacher env, without self.state calls
    vec = state[-3:]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum() * 0.01
    reward = reward_dist  # + reward_ctrl
    return reward


def get_reward_cp(state, action):
    # custom reward for sq error from x=0, theta = 0
    reward = state[0] ** 2 + state[2] ** 2
    return -reward


def get_reward_cf(state, action):
    # custom reward for sq error from x=0, theta = 0
    # reward = np.cos(state[4])*np.cos(state[3])
    # if (np.rad2deg(state[3]) < 5) and (np.rad2deg(state[4]) < 5):
    #     reward = 1
    # else:
    #     reward = 0
    reward = -state[3] ** 2 - state[4] ** 2
    return reward


def get_reward(predictions, actions, r_function):
    # takes in the predicted trajectory and returns the reward
    rewards = {}
    num_traj = len(actions)
    for m_label, state_data in predictions.items():
        r = []
        for i in range(num_traj):
            r_sub = 0
            cur_states = state_data[i]
            cur_actions = actions[i]
            for s, a in zip(cur_states, cur_actions):
                # TODO need a specific get reward function for the reacher env
                r_sub += r_function(s, a)
            r.append(r_sub)
        rewards[m_label] = (r, np.mean(r), np.std(r))

    return rewards


from policy import PID


def eval_rch(parameters):
    P = np.array([parameters["p1"], parameters["p2"], parameters["p3"], parameters["p4"], parameters["p5"]])
    I = np.zeros(5)
    # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
    D = np.array([parameters["d1"], parameters["d2"], parameters["d3"], parameters["d4"], parameters["d5"]])
    target = np.array([parameters["t1"], parameters["t2"], parameters["t3"], parameters["t4"], parameters["t5"]])

    policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
    # env = gym.make("Reacher3d-v2")
    s0 = env.reset()
    env.goal = rch_goal
    rews = []
    for i in range(1):
        # print(type(env))
        dotmap = run_controller(env, horizon=500, policy=policy, video=False)

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D

        rews.append(np.sum(dotmap.rewards) / 500)
        # if len(dotmap.actions) < 200:
        #     rews[-1] += dotmap.rewards[-1]*(200-len(dotmap.actions))
        dotmap.states = np.stack(dotmap.states)
        dotmap.rewards = rews[-1]
        dotmap.actions = np.stack(dotmap.actions)
        exp_data.append(dotmap)
    r = np.mean(rews)
    log.info(
        f"Parameter eval in env achieved r {np.round(r, 3)}, var {np.round(np.std(rews), 3)}")

    return {"Reward": (np.mean(rews), 0.01), }


def eval_rch_model(parameters):
    # env = gym.make("Reacher3d-v2")
    P = np.array([parameters["p1"], parameters["p2"], parameters["p3"], parameters["p4"], parameters["p5"]])
    I = np.zeros(5)
    # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
    D = np.array([parameters["d1"], parameters["d2"], parameters["d3"], parameters["d4"], parameters["d5"]])
    target = np.array([parameters["t1"], parameters["t2"], parameters["t3"], parameters["t4"], parameters["t5"]])
    k_param = np.concatenate((P, D, target))
    rews = []
    for i in range(20):

        s0 = env.reset()
        env.goal = rch_goal

        # print(type(env))
        t_range = np.arange(1, 500, 1)

        s_tile = np.tile(s0, 499).reshape(499, -1)
        k_tile = np.tile(k_param, 499).reshape(499, -1)
        input = np.concatenate((s_tile, t_range.reshape(-1, 1), k_tile), axis=1)
        states = traj_model.predict(input)
        traj = np.concatenate((s0.reshape(1, -1)[:, traj_model.state_indices], states.numpy()), axis=0)
        rew = 0  # get_reward(np.concatenate((s0.reshape(1,-1),states.numpy()),axis=0),np.zeros(200,1),get_reward_cp)
        for t in traj:
            rew += get_reward_reacher(t, 0) / 500
        rews.append(rew)

    # r = np.mean(rews)
    log.info(
        f"Parameter eval in model achieved r {np.round(np.mean(rews), 3)}, var {np.round(np.std(rews), 3)}")
    # THIS LINE IS WEIRD
    return {"Reward": (np.mean(rews), np.max(np.std(rews)), 0.01), }

    # return {"Reward": (np.mean(rews), np.std(rews)), }


def eval_rch_model_scaled(parameters):
    # env = gym.make("Reacher3d-v2")
    s0 = env.reset()
    env.goal = rch_goal
    parameters = np.multiply([3, 3, 3, 3, 3, .66, .66, .66, .66, .66, 1, 1, 1, 1, 1], parameters)
    parameters += [2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0]

    # P = np.array([parameters["p1"], parameters["p2"], parameters["p3"], parameters["p4"], parameters["p5"]])
    # I = np.zeros(5)
    # # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
    # D = np.array([parameters["d1"], parameters["d2"], parameters["d3"], parameters["d4"], parameters["d5"]])
    # target = np.array([parameters["t1"], parameters["t2"], parameters["t3"], parameters["t4"], parameters["t5"]])
    k_param = parameters  # np.concatenate((P, D, target))

    rews = []
    if np.any(k_param[:5]) > 5 or np.any(k_param[5:10] > 1) or np.any(k_param[:10] < 0) \
            or np.any(np.abs(k_param[10:]) > 3):
        rews = [-1000]
    else:
        for i in range(10):
            s0 = env.reset()
            env.goal = rch_goal

            # print(type(env))
            t_range = np.arange(1, 500, 1)

            s_tile = np.tile(s0, 499).reshape(499, -1)
            k_tile = np.tile(k_param, 499).reshape(499, -1)
            input = np.concatenate((s_tile, t_range.reshape(-1, 1), k_tile), axis=1)
            states = traj_model.predict(input)
            traj = np.concatenate((s0.reshape(1, -1)[:, traj_model.state_indices], states.numpy()), axis=0)
            rew = 0  # get_reward(np.concatenate((s0.reshape(1,-1),states.numpy()),axis=0),np.zeros(200,1),get_reward_cp)
            for t in traj:
                rew += get_reward_reacher(t, 0) / 500
            rews.append(rew)

    # r = np.mean(rews)
    # log.info(
    #     f"Parameter eval in model achieved r {np.round(np.mean(rews), 3)}, var {np.round(np.std(rews), 3)}")
    # return -np.mean(rews)
    return -np.mean(rews)


def eval_cp_model(parameters):
    k_param = [parameters["k1"], parameters["k2"], parameters["k3"], parameters["k4"]]
    # env = gym.make("Cartpole-v0")
    rews = []
    for i in range(1):  # 20):
        s0 = env.reset()
        t_range = np.arange(1, 200, 1)

        s_tile = np.tile(s0, 199).reshape(199, -1)
        k_tile = np.tile(k_param, 199).reshape(199, -1)
        input = np.concatenate((s_tile, t_range.reshape(-1, 1), k_tile), axis=1)
        states = traj_model.predict(input)
        traj = np.concatenate((s0.reshape(1, -1), states.numpy()), axis=0)
        rew = 0  # get_reward(np.concatenate((s0.reshape(1,-1),states.numpy()),axis=0),np.zeros(200,1),get_reward_cp)
        for t in traj:
            rew += np.exp(get_reward_cp(t, 0)) / 200
        rews.append(rew)

    log.info(
        f"Parameter eval in model {np.round(k_param, 3)} achieved r {np.round(np.mean(rews), 3)}, var {np.round(np.std(rews), 3)}")
    return {"Reward": (np.mean(rews), np.std(rews)), }


def eval_cp_model_scaled(parameters):
    k_param = parameters  # [parameters["k1"], parameters["k2"], parameters["k3"], parameters["k4"]]
    # shift and scaling
    k_param += [-1, -5, -12, -10]
    k_param = np.multiply([1, 2, 4, 2], k_param)
    if np.any(k_param > 0) or np.any(k_param < -75):
        rews = [-1000]
    else:
        # k_param[0] = k_param[0] -1
        # k_param[1] = 4*k_param[1] - 4
        # k_param[2] = 20*k_param[2] - 50
        # k_param[3] = 5*k_param[3] - 10
        # env = gym.make("Cartpole-v0")
        rews = []
        for i in range(20):
            s0 = env.reset()
            t_range = np.arange(1, 200, 1)

            s_tile = np.tile(s0, 199).reshape(199, -1)
            k_tile = np.tile(k_param, 199).reshape(199, -1)
            input = np.concatenate((s_tile, t_range.reshape(-1, 1), k_tile), axis=1)
            states = traj_model.predict(input)
            traj = np.concatenate((s0.reshape(1, -1), states.numpy()), axis=0)
            rew = 0  # get_reward(np.concatenate((s0.reshape(1,-1),states.numpy()),axis=0),np.zeros(200,1),get_reward_cp)
            for t in traj:
                rew += np.exp(get_reward_cp(t, 0)) / 200
            rews.append(rew)

    # log.info(
    #     f"Parameter eval in model {np.round(k_param, 3)} achieved r {np.round(np.mean(rews), 3)}")  # , var {np.round(np.std(rews), 3)}")
    return -np.mean(rews)


def eval_cp(parameters):
    # k_param = np.array([ -0.70710678,  -4.2906244,  -37.45394052,  -8.06650137])
    k_param = [parameters["k1"], parameters["k2"], parameters["k3"], parameters["k4"]]
    # k_param = np.array([ -2.062,  -8.75,  -17.412,  -3.   ])
    # These values are replaced and don't matter
    m_c = 1
    m_p = 1
    m_t = m_c + m_p
    g = 9.8
    l = .01
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

    policy = LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0])
    policy.K = np.array(k_param)
    # env = gym.make("Cartpole-v0")
    s0 = env.reset()
    rews = []
    for i in range(1):
        dotmap = run_controller(env, horizon=200, policy=policy, video=False)
        rews.append(np.sum(dotmap.rewards) / 200)
        # if len(dotmap.actions) < 200:
        #     rews[-1] += dotmap.rewards[-1]*(200-len(dotmap.actions))
        dotmap.states = np.stack(dotmap.states)
        dotmap.actions = np.stack(dotmap.actions)
        dotmap.rewards = rews[-1]
        dotmap.K = np.array(policy.K).flatten()
        exp_data.append(dotmap)
    r = np.mean(rews)
    log.info(
        f"Parameter eval in env {np.round(k_param, 3)} achieved r {np.round(r, 3)}, var {np.round(np.std(rews), 3)}")

    return {"Reward": (np.mean(rews), 0.01), }


class CartpoleMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = eval_cp(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


class CartpoleMetricModel(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = eval_cp_model(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


class ReacherMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = eval_rch(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


class ReacherMetricModel(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = eval_rch_model(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


def plot_results():
    labels = ['BO', 'Model BO', 'Model CMA-ES', 'LQR', 'Random Search']
    means = [0.9, 0.96, 0.95, 0.9274, 0.7856]
    stds = [0.1, 0.05, 0.03, 0.07, 0.25]
    # convert to cost function by multiplying by 200, then taking -log
    # bo =
    # bo_std =
    # bomodel =
    # bomodel_std =
    # cmamodel =
    # cmamodel_std =
    # randomsearch =
    # randomsearch_std =
    # lqr =
    # lqr_std =


@hydra.main(config_path='conf/mbrl.yaml')
def mbrl(cfg):
    # Environment setup
    global env
    env_model = cfg.env.name
    label = cfg.env.label
    np.random.seed(cfg.random_seed)
    env = gym.make(env_model)
    env.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    if cfg.plot:
        dirs = os.listdir(hydra.utils.get_original_cwd() + cfg.dir1)
        r1 = []
        if 'cartpole' in cfg.dir1:
            label = 'cartpole'
            lims = [0.9, 1.0]
            ind = 15
        else:
            label = 'reacher'
            lims = [-2, -0.25]
            ind = 25

        for d in dirs:
            files = glob.glob(hydra.utils.get_original_cwd() + cfg.dir1 + '/' + d + '/**.dat')
            r1_sub = []
            for g in files:
                rew_trial = torch.load(g)
                # if label== 'cartpole':
                #     # switch to cost
                #     # rew_trial = -np.log(rew_trial)
                #     rew_trial *= 200 #-np.log(rew_trial)
                # if label == 'cartpole':
                rew_trial = rew_trial.tolist()
                    # print(rew_trial[0])
                    # print([r for r in rew_trial])
                if type(rew_trial) == list:
                    for t in rew_trial:
                        r1_sub.append(t)
                else:
                    r1_sub.append(rew_trial)
            r1.append(np.maximum.accumulate(r1_sub).tolist())

        dirs = os.listdir(hydra.utils.get_original_cwd() + cfg.dir2)
        r2 = []
        for d in dirs:
            files = glob.glob(hydra.utils.get_original_cwd() + cfg.dir2 + '/' + d + '/**.dat')
            r2_sub = []
            for g in files:
                rew_trial = torch.load(g)
                # if label== 'cartpole':
                #     # switch to cost
                #     # rew_trial = -np.log(rew_trial)
                #     rew_trial *= 200 #-np.log(rew_trial)
                if type(rew_trial) == list:
                    for t in rew_trial:
                        r2_sub.append(t)
                else:
                    r2_sub.append(rew_trial)
            r2.append(np.maximum.accumulate(r2_sub).tolist())
        plotly.io.orca.config.executable = '/home/hiro/miniconda3/envs/ml_clean/lib/orca_app/orca'
        import plotly.io as pio
        pio.orca.config.use_xvfb = True
        print(r1)
        print(r2)
        from plot import plot_rewards_over_trials
        # if label == 'cartpole':
        plot_rewards_over_trials([np.stack(r1)[:, :ind].tolist(), np.stack(r2)[:, :ind].tolist()], "name", save=True,
                                 limits=lims)
        # plot_rewards_over_trials(np.stack(r1).tolist(), "name", save=True)
        quit()
    # bo = [.875,.952,.998,.862,.988,.821,.66,1,.954,.957]
    # print(np.mean(bo))
    # print(np.std(bo))
    # bomodel = [0.997,0.991, 0.929, 0.981, 0.813,  1.0, 0.982, 0.989,  0.984,  0.962]
    # print(np.mean(bomodel))
    # print(np.std(bomodel))
    # cma = [0.919, 0.979, 0.876, 0.995, 0.894,  .987,  0.936,  0.962, 0.943,  0.959]
    # print(np.mean(cma))
    # print(np.std(cma))
    # quit()
    # trajectories = torch.load(hydra.utils.get_original_cwd() + '/trajectories/' + label + '/raw' + cfg.data_dir)

    f = hydra.utils.get_original_cwd() + '/models/' + label + '/'
    model_one = torch.load(f + cfg.step_model + '.dat')
    model_traj = torch.load(f + cfg.traj_model + '.dat')

    # get rewards, control policy, etc for each type, and control parameters
    # data_train = trajectories[0]  # [::10] #+trajectories[1]
    # reward = [t['rewards'] for t in data_train]
    # states = [np.float32(t['states']) for t in data_train]
    # actions = [np.float32(t['actions']) for t in data_train]

    global exp_data
    global traj_model
    exp_data = []

    search_space = gen_search_space(cfg.problem)
    if label == "cartpole":
        eval_fn = eval_cp
        met = CartpoleMetric(name="Reward")
    elif label == "reacher":
        eval_fn = eval_rch
        met = ReacherMetric(name="Reward")
        # env = gym.make("Reacher3d-v2")
        global rch_goal
        env.reset()
        rch_goal = env.goal
        # rch_goal = np.random.uniform(low=-.5, high=.5, size=3)
        # env.goal = rch_goal
        log.info(f"Reacher env optimizing to goal {rch_goal}")

    # TODO three scenairos
    # 1. BO on the direct env (doable)
    # 2. BO somehow using the traj model to predict reward (can this be a traj model instead of GP?)
    # 3. CMA-ES (or CEM) on traj model to select parameters
    # 4. Make 3 into a loop of some kind

    exp = SimpleExperiment(
        name=cfg.problem.name,
        search_space=SearchSpace(search_space),
        evaluation_function=eval_fn,
        objective_name="Reward",
        # log_scale=True,
        minimize=cfg.metric.minimize,
        # outcome_constraints=outcome_con,
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=met,
            minimize=cfg.metric.minimize,
        ),
    )

    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config
    from ax.plot.contour import plot_contour

    def get_data(exper, skip=0):
        raw_data = exper.fetch_data().df.values
        rew = raw_data[:, 2].reshape(-1, 1)
        trials = exper.trials
        params_dict = [trials[i].arm.parameters for i in range(len(trials))]
        params = np.array(np.stack([list(p.values()) for p in params_dict]), dtype=float)
        cat = np.concatenate((rew[skip:], params[skip:, :]), axis=1)
        return cat

    if cfg.opt == 'cma-itr':
        sobol = Models.SOBOL(exp.search_space)
        num_search = cfg.bo.random
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()
        rand_data = get_data(exp)
        sobol_data = exp.eval()
        # torch.save(get_data(exp)[-1, 0], f"rew_{0}.dat")

        if label == 'cartpole':
            from cartpole_lqr import create_dataset_traj
        else:
            from reacher_pd import create_dataset_traj

        from dynamics_model import DynamicsModel

        if label == 'cartpole':
            from cartpole_lqr import create_dataset_traj
            eval_fn_cma = eval_cp_model_scaled
            n_opt = 4
            sig0 = 1

            def scale(x):
                x += [-1, -5, -12, -5]
                x = np.multiply([1, 2, 4, 2], x)
                return x
        else:
            from reacher_pd import create_dataset_traj
            eval_fn_cma = eval_rch_model_scaled
            n_opt = 15
            sig0 = .35

            def scale(x):
                x = np.multiply([3, 3, 3, 3, 3, .66, .66, .66, .66, .66, 1, 1, 1, 1, 1], x)
                x += [2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0]
                return x

        for n in range(cfg.bo.cma):
            log.info(f"Optimizing traj {n}")
            dataset = create_dataset_traj(exp_data, control_params=cfg.model.training.control_params,
                                          train_target=cfg.model.training.train_target,
                                          threshold=cfg.model.training.filter_rate,
                                          t_range=cfg.model.training.t_range)

            def unison_shuffled_copies(a, b):
                assert len(a) == len(b)
                p = np.random.permutation(len(a))
                return a[p], b[p]

            d1, d2 = unison_shuffled_copies(dataset[0], dataset[1])
            traj_model = DynamicsModel(cfg)
            train_logs, test_logs = traj_model.train((d1, d2), cfg)
            es = cma.CMAEvolutionStrategy(n_opt * [0], sig0, {'verbose': 0, 'maxiter': 10})
            es.optimize(eval_fn_cma)
            # es.result_pretty()
            res = es.result.xbest
            res = scale(res)
            # torch.save(exp_data[-1].rewards, f"rew_{n + 1}.dat")
            torch.save([exp_d.rewards for exp_d in exp_data], f"rew_full.dat")

            log.info("Trial")
            # log.info(f"Final CMA params: {res}")
            if label == 'cartpole':
                final_params = {
                    "k1": res[0],
                    "k2": res[1],
                    "k3": res[2],
                    "k4": res[3],
                }
            else:
                final_params = {
                    'p1': res[0],
                    'p2': res[1],
                    'p3': res[2],
                    'p4': res[3],
                    'p5': res[4],
                    'd1': res[5],
                    'd2': res[6],
                    'd3': res[7],
                    'd4': res[8],
                    'd5': res[9],
                    't1': res[10],
                    't2': res[11],
                    't3': res[12],
                    't4': res[13],
                    't5': res[14]}
            r = []
            if label == 'cartpole':
                val = eval_cp(final_params)
            else:
                val = eval_rch(final_params)
            r.append(val["Reward"][0])

        log.info(f"Reward of final values {np.mean(r)}, std {np.std(r)}")
        log.info(r)



    else:
        log.info(f"Running {cfg.bo.random} Sobol initialization trials...")
        sobol = Models.SOBOL(exp.search_space)
        num_search = cfg.bo.random
        for i in range(num_search):
            exp.new_trial(generator_run=sobol.gen(1))
            exp.trials[len(exp.trials) - 1].run()
            # torch.save(get_data(exp)[-1, 0], f"rew_{i}.dat")

        num_opt = cfg.bo.optimized
        rand_data = get_data(exp)
        sobol_data = exp.eval()
        log.info(f"Completed random search with mean reward {np.mean(rand_data[:, 0])}")

        if cfg.opt == 'bo':
            gpei = Models.BOTORCH(experiment=exp, data=sobol_data)
            for i in range(num_opt):
                # if (i % 5) == 0 and cfg.plot_during:
                #     plot = plot_contour(model=gpei,
                #                         param_x="N",
                #                         param_y="L",
                #                         metric_name="Energy_(uJ)", )
                #     data = plot[0]['data']
                #     lay = plot[0]['layout']
                #
                #     render(plot)

                log.info(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
                # Reinitialize GP+EI model at each step with updated data.
                batch = exp.new_trial(generator_run=gpei.gen(1))
                gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
                # torch.save(get_data(exp)[-1, 0], f"rew_{i + cfg.bo.random}.dat")

            # raw_data = exp.fetch_data().df.values
            # rew = raw_data[:, 2].reshape(-1, 1)
            # trials = exp.trials
            # params_dict = [trials[i].arm.parameters for i in range(len(trials))]
            # params = np.array(np.stack([list(p.values()) for p in params_dict]), dtype=float)
            # cat = np.concatenate((rew, params), axis=1)
            load = get_data(exp)
            sorted_all = load[load[:, 0].argsort()]
            if not cfg.metric.minimize: sorted_all = sorted_all[::-1]  # reverse if minimize

            # def print_results(label, values):
            # if label == 'cartpole':
            log.info("10 best param, rewards")
            for i in range(5):
                log.info(
                    f"Rew {np.round(sorted_all[i, 0], 4)}, param {np.round(np.array(sorted_all[i, 1:], dtype=float), 3)}")
            torch.save(get_data(exp)[:, 0], f"rew_full.dat")

            log.info("EVAL ON SYSTEM")
            log.info(f"Final BO params: {sorted_all[0, :]}")
            if label == 'cartpole':
                final_params = {
                    "k1": sorted_all[0, 1],
                    "k2": sorted_all[0, 2],
                    "k3": sorted_all[0, 3],
                    "k4": sorted_all[0, 4],
                }
            else:
                final_params = {
                    'p1': sorted_all[0, 0],
                    'p2': sorted_all[0, 1],
                    'p3': sorted_all[0, 2],
                    'p4': sorted_all[0, 3],
                    'p5': sorted_all[0, 4],
                    'd1': sorted_all[0, 5],
                    'd2': sorted_all[0, 6],
                    'd3': sorted_all[0, 7],
                    'd4': sorted_all[0, 8],
                    'd5': sorted_all[0, 9],
                    't1': sorted_all[0, 10],
                    't2': sorted_all[0, 11],
                    't3': sorted_all[0, 12],
                    't4': sorted_all[0, 13],
                    't5': sorted_all[0, 14]}
            r = []
            for i in range(cfg.bo.num_eval):
                if label == 'cartpole':
                    val = eval_cp(final_params)
                else:
                    val = eval_rch(final_params)
                r.append(val["Reward"][0])

            log.info(f"Reward of final values {np.mean(r)}, std {np.std(r)}")

            # log.info(r)

            # plot_learn = plot_learning(exp, cfg)
            # # go.Figure(plot_learn).show()
            # save_fig([plot_learn], "optimize")
            #
            # plot = plot_contour(model=gpei,
            #                     param_x="k1",
            #                     param_y="k2",
            #                     metric_name="Reward",
            #                     lower_is_better=cfg.metric.minimize)
            # save_fig(plot, dir=f"k1k2rew")
            #
            # plot = plot_contour(model=gpei,
            #                     param_x="k3",
            #                     param_y="k4",
            #                     metric_name="Reward",
            #                     lower_is_better=cfg.metric.minimize)
            # save_fig(plot, dir=f"k3k4rew")

        elif cfg.opt == 'bo-model':
            if label == 'cartpole':
                from cartpole_lqr import create_dataset_traj
            else:
                from reacher_pd import create_dataset_traj

            from dynamics_model import DynamicsModel
            dataset = create_dataset_traj(exp_data, control_params=cfg.model.training.control_params,
                                          train_target=cfg.model.training.train_target,
                                          threshold=cfg.model.training.filter_rate,
                                          t_range=cfg.model.training.t_range)

            traj_model = DynamicsModel(cfg)
            train_logs, test_logs = traj_model.train(dataset, cfg)

            # change to model evaluation!
            if label == 'cartpole':
                met = CartpoleMetricModel(name="Reward")
                exp.evaluation_function = eval_cp_model
            else:
                met = ReacherMetricModel(name="Reward")
                exp.evaluation_function = eval_rch_model

            optimization_config_model = OptimizationConfig(
                objective=Objective(
                    metric=met,
                    minimize=cfg.metric.minimize,
                ),
            )

            exp.optimization_config = optimization_config_model
            gpei = Models.BOTORCH(experiment=exp, data=sobol_data)
            for i in range(num_opt):
                # if (i % 5) == 0 and cfg.plot_during:
                #     plot = plot_contour(model=gpei,
                #                         param_x="N",
                #                         param_y="L",
                #                         metric_name="Energy_(uJ)", )
                #     data = plot[0]['data']
                #     lay = plot[0]['layout']
                #
                #     render(plot)

                log.info(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
                # Reinitialize GP+EI model at each step with updated data.
                batch = exp.new_trial(generator_run=gpei.gen(1))
                gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

            # raw_data = exp.fetch_data().df.values
            # rew = raw_data[:, 2].reshape(-1, 1)
            # trials = exp.trials
            # params_dict = [trials[i].arm.parameters for i in range(len(trials))]
            # params = np.array(np.stack([list(p.values()) for p in params_dict]), dtype=float)
            # cat = np.concatenate((rew, params), axis=1)
            bomodel = get_data(exp, skip=cfg.bo.random)
            sorted_all = bomodel[bomodel[:, 0].argsort()]
            if not cfg.metric.minimize: sorted_all = sorted_all[::-1]  # reverse if minimize

            log.info(f"{5} best param, rewards")
            for i in range(5):
                log.info(
                    f"Rew {np.round(sorted_all[i, 0], 4)}, param {np.round(np.array(sorted_all[i, 1:], dtype=float), 3)}")
            log.info("EVAL ON SYSTEM")
            log.info(f"Final BO params: {sorted_all[0, :]}")
            if label == 'cartpole':
                final_params = {
                    "k1": sorted_all[0, 1],
                    "k2": sorted_all[0, 2],
                    "k3": sorted_all[0, 3],
                    "k4": sorted_all[0, 4],
                }
            else:
                final_params = {
                    'p1': sorted_all[0, 0],
                    'p2': sorted_all[0, 1],
                    'p3': sorted_all[0, 2],
                    'p4': sorted_all[0, 3],
                    'p5': sorted_all[0, 4],
                    'd1': sorted_all[0, 5],
                    'd2': sorted_all[0, 6],
                    'd3': sorted_all[0, 7],
                    'd4': sorted_all[0, 8],
                    'd5': sorted_all[0, 9],
                    't1': sorted_all[0, 10],
                    't2': sorted_all[0, 11],
                    't3': sorted_all[0, 12],
                    't4': sorted_all[0, 13],
                    't5': sorted_all[0, 14]}
            r = []
            for i in range(cfg.bo.num_eval):
                if label == 'cartpole':
                    val = eval_cp(final_params)
                else:
                    val = eval_rch(final_params)
                r.append(val["Reward"][0])
            log.info(f"Reward of final values {np.mean(r)}, std {np.std(r)}")
            log.info(r)

            plot_learn = plot_learning(exp, cfg)
            # go.Figure(plot_learn).show()
            save_fig([plot_learn], "optimize")

            plot = plot_contour(model=gpei,
                                param_x="k1",
                                param_y="k2",
                                metric_name="Reward",
                                lower_is_better=cfg.metric.minimize)
            save_fig(plot, dir=f"k1k2rew")

            plot = plot_contour(model=gpei,
                                param_x="k3",
                                param_y="k4",
                                metric_name="Reward",
                                lower_is_better=cfg.metric.minimize)
            save_fig(plot, dir=f"k3k4rew")
            # raise NotImplementedError("TODO")

        elif cfg.opt == 'cma':
            if label == 'cartpole':
                from cartpole_lqr import create_dataset_traj
                eval_fn_cma = eval_cp_model_scaled
                n_opt = 4
                sig0 = 1

                def scale(x):
                    x += [-1, -5, -12, -5]
                    x = np.multiply([1, 2, 4, 2], x)
                    return x
            else:
                from reacher_pd import create_dataset_traj
                eval_fn_cma = eval_rch_model_scaled
                n_opt = 15
                sig0 = .5

                def scale(x):
                    x += [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0]
                    x = np.multiply([3, 3, 3, 3, 3, .66, .66, .66, .66, .66, 1, 1, 1, 1, 1], x)
                    return x

            from dynamics_model import DynamicsModel
            # dataset = create_dataset_traj(exp_data, control_params=cfg.model.training.control_params,
            #                               train_target=cfg.model.training.train_target,
            #                               threshold=cfg.model.training.filter_rate,
            #                               t_range=cfg.model.training.t_range)
            # # global traj_model
            #
            # traj_model = DynamicsModel(cfg)
            # train_logs, test_logs = traj_model.train(dataset, cfg)
            traj_model = model_traj

            es = cma.CMAEvolutionStrategy(n_opt * [0], sig0, {'verbose': 0, 'maxiter': 10})
            es.optimize(eval_fn_cma)
            # es.result_pretty()
            res = es.result.xbest
            res = scale(res)

            log.info("EVAL ON SYSTEM")
            log.info(f"Final CMA params: {res}")
            if label == 'cartpole':
                final_params = {
                    "k1": res[0],
                    "k2": res[1],
                    "k3": res[2],
                    "k4": res[3],
                }
            else:
                final_params = {
                    'p1': res[0],
                    'p2': res[1],
                    'p3': res[2],
                    'p4': res[3],
                    'p5': res[4],
                    'd1': res[5],
                    'd2': res[6],
                    'd3': res[7],
                    'd4': res[8],
                    'd5': res[9],
                    't1': res[10],
                    't2': res[11],
                    't3': res[12],
                    't4': res[13],
                    't5': res[14]}
            r = []
            for i in range(cfg.bo.num_eval):
                if label == 'cartpole':
                    val = eval_cp(final_params)
                else:
                    val = eval_rch(final_params)
                r.append(val["Reward"][0])
            log.info(f"Reward of final values {np.mean(r)}, std {np.std(r)}")
            log.info(r)
        else:
            raise NotImplementedError("Other types of opt tbd")

    # torch.save(r, "final_rews.dat")
    if label == 'cartpole': log.info(f"Optimal params: {cfg.optimal}")


def save_fig(plot, dir):
    plotly.io.orca.config.executable = '/home/hiro/miniconda3/envs/ml_clean/lib/orca_app/orca'
    import plotly.io as pio
    pio.orca.config.use_xvfb = True

    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    fig = go.Figure(fig)
    fig.update_layout(
        font_family="Times New Roman",
        font_color="Black",
        font_size=14,
        margin=dict(r=5, t=10, l=20, b=20)
    )
    fig.write_image(os.getcwd() + "/" + dir + '.pdf')


def plot_learning(exp, cfg):
    objective_means = np.array([[exp.trials[trial].objective_mean] for trial in exp.trials])
    cumulative = optimization_trace_single_method(
        y=np.maximum.accumulate(objective_means.T, axis=1) * 1.01, ylabel=cfg.metric.name,
        trace_color=tuple((83, 78, 194)),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    all = optimization_trace_single_method(
        y=objective_means.T, ylabel=cfg.metric.name,
        model_transitions=[cfg.bo.random], trace_color=tuple((114, 110, 180)),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )

    layout_learn = cumulative[0]['layout']
    layout_learn['paper_bgcolor'] = 'rgba(0,0,0,0)'
    layout_learn['plot_bgcolor'] = 'rgba(0,0,0,0)'
    layout_learn['showlegend'] = False

    d1 = cumulative[0]['data']
    d2 = all[0]['data']

    for t in d1:
        t['legendgroup'] = cfg.metric.name + ", cum. max"
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name + ", cum. max"
            t['line']['color'] = 'rgba(200,20,20,0.5)'
            t['line']['width'] = 4

    for t in d2:
        t['legendgroup'] = cfg.metric.name
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name
            t['line']['color'] = 'rgba(20,20,200,0.5)'
            t['line']['width'] = 4

    fig = {
        "data": d1 + d2,  # data,
        "layout": layout_learn,
    }
    return fig


if __name__ == '__main__':
    sys.exit(mbrl())
