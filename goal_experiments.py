import numpy as np
import gym
from envs import *

from policy import PID

import time

"""
The main purpose of this file is to figure out how to get the reacher
to go towards the target

Findings:
    variable 0 is cos of rotation around y axis (y being vertical)
    variable 1 is cos of rotation about z axis
    variable 2 is cos of angle in first joint
    variable 3 affects angle of second joint
    variable 4 affects rotation of second jointy
"""

def obs2q(obs):
    return obs[0:5]

targets = [
# [0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0.1],
# [0, 0, 0, 0, 0.2],
np.cos([0, 0, 0, 0, 0]),
np.cos([10, 0, 0, 0, 0]),
np.cos([0, 10, 0, 0, 0]),
np.cos([0, 0, 10, 0, 0]),
np.cos([0, 0, 0, 10, 0]),
np.cos([0, 0, 0, 0, 10]),
# [0, 0, 0, 2, 0],
# [0, 0, 0, 3, 0],
# [0, 0, 0, 4, 0],
# [0, 0, 0, 10, 0],



# [0, 0, 0, .1, 0],
# [0, 0, 0, .2, 0],
# [0, 0, 0, .3, 0],
# [0, 0, 0, .4, 0],
# [0, 0, 0, 1, 0],
# [0, 0, 0, 2, 0],
# [0, 0, 0, 3, 0],
# [0, 0, 0, 0, 0],

# [0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0],

# [0, 0, 0.1, 0, 0],
# [0, 0, 0.2, 0, 0],
# # [0, 0, 0.3, 0, 0],
# # [0, 0, 0.4, 0, 0],
# # [0, 0, 0.5, 0, 0],
# # [0, 0, 0.6, 0, 0],
# [0, 0, 0.9, 0, 0],
# [0, 0, 1.001, 0, 0],
# [0, 0, 1.1, 0, 0],
# [0, 0, 1.3, 0, 0],
# [0, 0, 1.5, 0, 0],
# [0, 0, 2, 0, 0],

]
for target in targets:
    env = gym.make("Reacher3d-v2")
    # env = gym.make("Ant-v2")

    # theta = env.sim.data.qpos.flat[:5]

    P = np.array([4, 4, 1, 1, 1])
    I = np.zeros(5)
    D = np.array([0.2, 0.2, 2, 0.4, 0.4])
    # target = np.hstack((0, 0, env.goal))
    # variable 0 is rotation around y axis (y being vertical)
    # variable 1 is rotation about z axis
    # target=np.array((0,np.pi,np.pi,0,0))
    # target=np.array((0,1,1,1,1))
    # target = np.array([0.5, 0.5] + env.goal)# change this

    # print(target)

    policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)

    observation = env.reset()
    # print(env.get_body_com("fingertip"), env.get_body_com("target"))
    # print(observation)
    # print(env.goal)
    for t in range(500):
        env.render()
        state = observation
        action, t = policy.act(obs2q(state))
        # action = env.action_space.sample()*10

        # print(action)

        observation, reward, done, info = env.step(action)
        print(env.goal)
        print(observation)

        time.sleep(.0005)
