# continuousprediction
Formulating Model-based RL Dynamics as a continuous rather then one step prediction

## Running the Code:

To run the code use the following steps:

1. Create an environment from the provided yml file
2. Find the folder for that with `echo $CONDA_PREFIX` on Mac or `echo %CONDA_PREFIX%` on Windows
3. Navigate to `envs/continuouspred/lib/python3.6/site-packages/gym/envs`
4. In the file __init__.py, add ```register(
    id='Reacher3d-v1',
    entry_point='gym.envs.mujoco:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)``` preferably around line 214, in the MuJoCo section
5. From the `reacher3d` folder in the repo, copy the `reacher3d.py` file into `mujoco` and copy `reacher3d.xml` into `mujoco/assets`
6. There will be another `__init__.py` file in the `mujoco` folder. Copy the line `from gym.envs.mujoco.reacher3d import Reacher3dEnv` into the bottom of that one
