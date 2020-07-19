import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from control import StateSpace


class StateSpaceEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Observation: Defined by config
    Actions: Defined by config
    """

    def __init__(self):
        print("Make sure to run setup on state space env")
        self.setup_ran = False

    def setup(self, cfg):
        self.cfg = cfg.env
        self.dt = self.cfg.params.dt

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        self.seed(seed=cfg.random_seed)

        a = np.mat(self.cfg.params.A)
        b = np.mat(self.cfg.params.B)
        c = np.mat(self.cfg.params.C)
        d = np.mat(self.cfg.params.D)

        self.dx = np.shape(a)[1]
        self.du = np.shape(b)[1]
        self.dy = np.shape(c)[0]

        low_a = np.ones((self.du)) * 10
        high_a = np.ones((self.du)) * 10
        self.action_space = self.action_space = spaces.Box(low=low_a,
                                                           high=high_a,
                                                           dtype=np.float32)

        low_obs = -np.ones((self.dx)) * np.inf
        high_obs = np.ones((self.dx)) * np.inf
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        self.sys = StateSpace(a, b, c, d, self.dt)
        self.reset()
        self.setup_ran = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        return np.array(np.matmul(self.sys.C, self.state)).squeeze()

    def _get_state(self):
        return self.state

    def step(self, action):
        if not self.setup_ran:
            raise ValueError("System not yet passed")
        # self.last_action = action
        last_state = self.state
        self.state = np.matmul(self.sys.A, last_state) + np.matmul(self.sys.B, action).reshape(self.dx,1)
        obs = self.get_obs()
        reward = self.get_reward(self.state, action)
        done = False
        return np.array(obs), reward, done, {}

    def get_reward(self, next_ob, action):
        # Update to include balance
        return -(np.mean(next_ob) ** 2 + np.mean(action) ** 2)

    def reset(self):
        self.state = self.np_random.uniform(low=-1, high=1, size=(self.dx,1))
        # return np.array(self.state
        return self.get_obs()