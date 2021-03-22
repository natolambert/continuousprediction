from gym.envs.registration import register

register(
    id='Reacher3d-v2',
    entry_point='envs.reacher3d:Reacher3dEnv',
    # max_episode_steps=500,
    # reward_threshold=-200,
)


register(
    id='Cartpole-v0',
    entry_point='envs.cartpole:CartPoleContEnv',
    # max_episode_steps=500,
    # reward_threshold=-200,
)
register(
    id='Cartpole-mod-v0',
    entry_point='envs.cartpole_mod:CartPoleContEnv',
    # max_episode_steps=500,
    # reward_threshold=-200,
)

register(
    id='Crazyflie-v0',
    entry_point='envs.crazyflie:CrazyFlieEnv',
    # max_episode_steps=500,
    # reward_threshold=-200,
)

register(
    id='statespace-v0',
    entry_point='envs.statespace:StateSpaceEnv'
)

