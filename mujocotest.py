import gym
import time
import reacher3d
env = gym.make('Reacher3d-v2')
env.reset()
env.render()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(action)
print(observation)
for i in range(1000):
# # for i in range(1):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # print(action)
    # print(observation)
    env.render()
    # time.sleep(.03)
