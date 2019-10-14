from gym.envs.registration import register

register(
    id='Reacher3d-v2',
    entry_point='reacher3d.envs:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)
