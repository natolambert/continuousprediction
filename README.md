# continuousprediction
**Learning Accurate Long-term Dynamicsfor Model-based Reinforcement Learning** - 
(Nathan Lambert, Albert Wilcox, Howard Zhang, Kris Pister, Roberto Calandra)

## Abstract

Accurately predicting the dynamics of robotic systems is crucial to make use of model-based control. A common way to estimate dynamics is by modeling the one-step ahead prediction and then use it to recursively propagate the predicted state distribution over long horizons. Unfortunately, this approach is known to compound even small prediction errors, making long-term predictions inaccurate. In this paper we propose a new parametrizaion to supervised learning on state-action data to stably predict at longer horizons -- that we call a trajectory-based model. This trajectory-based model takes an initial state, a time index, and control parameters as inputs and predicts a state at that time.
Our results in simulated and experimental robotic tasks show accurate long term predictions, improved sample efficiency, and ability to predict task reward.

## Running the Code:


To run the code use the following steps:

1. Create a conda environment from the provided yml file and activate it
2. Installing mujoco will fail. See the repo for instructions: https://github.com/openai/mujoco-py
<!-- 2. Navigate into the `reacher3d` folder and run the command ```pip install -e .```
1. Create an environment from the provided yml file
2. Find the folder for that with `echo $CONDA_PREFIX` on Mac or `echo %CONDA_PREFIX%` on Windows
3. Navigate to `envs/continuouspred/lib/python3.6/site-packages/gym/envs`
4. In the file `__init__.py`, add ```register(
    id='Reacher3d-v1',
    entry_point='gym.envs.mujoco:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)``` preferably around line 214, in the MuJoCo section
5. From the `reacher3d` folder in the repo, copy the `reacher3d.py` file into `mujoco` and copy `reacher3d.xml` into `mujoco/assets`
6. There will be another `__init__.py` file in the `mujoco` folder. Copy the line `from gym.envs.mujoco.reacher3d import Reacher3dEnv` into the bottom of that one -->

## Using this for your robot:

To use this on your robot, here will be the process:
1. Create a new file this your `robot_name.py` (this is needed becuase the controller changes for each robot).
2. Create an environment config file in `conf/envs/robot_name.yml` with items like state dimension, control parameter dimension, and more for model training. Also create or re-used a core conf file like `reacher.yml` in `conf/`.
3. Create or modify existing data generation and trajectory-based model training code. See `create_dataset_traj( )` in multiple files for inspiration. The dimensions of this data must match the configuration.
4. The code should have two modes, train and collect. Collect runs the model and train will load objects from `dynamics_model.py` to train and save your model, if you so choose.
5. Use `evaluate.py` to view the model prediction accuracy.

## Core files for models and evaluation:
- `dynamics_model.py`: This class contains the modular class for dynamics models of the single step and trajectory parametrization. There is code to use neural networks and gaussian processes as the modelling tool.
- `policy.py`: This file contains the different controller parametrizations used in the experiments.
- `plot.py`: This file stores all the plotting functions used by the other files.

## Replicating Experiments:


Example of how to run efficiency code to train some models and then test them:

Train: `python3 efficiency.py training.num_traj=3,5,7,9 training.t_range=10,20,30,40 models=d,t training.copy=1,2,3,4,5 -m`

Test: `python3 efficiency.py mode=plot plotting.num_traj=[3,5,7,9] plotting.t_range=[10,20,30,40] plotting.models=[d,t] plotting.copy=[1,2,3,4,5] -m`

Example of how to run 

## Extra files currently not in use:

When examining the code, one will see a few extra files that represent potential future avenues for research. Some of these files are:
- `lorenz.py`: This was an attempt to model the long term behavior of the lorenz system. Results were mixed on this very challenging application.
- `stable_system.py`: This was used to evaluate how far into the future a trajectory-based model could predict a state-space system, but it was omitted from the paper.
