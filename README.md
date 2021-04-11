# continuousprediction
This repository was create for the paper: **Learning Accurate Long-term Dynamicsfor Model-based Reinforcement Learning** ([Arxiv](https://arxiv.org/abs/2012.09156), [page](https://www.natolambert.com/papers/2020-long-term-dynamics))

## Goal of the repository

Accurately predicting the dynamics of robotic systems is crucial to make use of model-based control. A common way to estimate dynamics is by modeling the one-step ahead prediction and then use it to recursively propagate the predicted state distribution over long horizons. Unfortunately, this approach is known to compound even small prediction errors, making long-term predictions inaccurate. In this paper we propose a new parametrizaion to supervised learning on state-action data to stably predict at longer horizons -- that we call a trajectory-based model. This trajectory-based model takes an initial state, a time index, and control parameters as inputs and predicts a state at that time.
Our results in simulated and experimental robotic tasks show accurate long term predictions, improved sample efficiency, and ability to predict task reward.

## Understanding compounding error

There are many datasets and pre-trained models in this repository. Here is how to navigate them.

The `cfg/envs/yaml` has a property `env.label` that directs how the data is saved. This enables some environments to have example trajectoreis with different properties. 
This is primarily the case for the Crazyflie and Reacher data. 
We have run more experiments on the simpler environments to illustrate fundamental tradeoffs.

Environment configurations that are more varied:
* state-space system gym environment parametrizations (the parent config `stable_sys.yaml` points to defaults in the `envs/` subdir):
    * `sys.yaml`: standard state-space configuration with upper triangular dynamics of dimension 3, action of dim 1, noise from uniform distribution of [-0.01,0.01].
    * `sys2.yaml`: state dimension increased to 9.
    * `sys3.yaml`: 27 dimensional state.
    * `sys4.yaml`: 81 dimensional state.
    * `sys5.yaml`: 243 dimensional state.
    * `sysn2.yaml`: original system, 10x noise.
    * `sysn3.yaml`: original system, 100x noise.
* variations on cartpole for studying unstable poles: not 100% created by configuration, some are still done by hand
    * a second configuration, `cartpole_mod.yaml` is provided for running a second set of dynamics.
    * _Note_: In this case, the separate environment is managed with an additional gym registration. 
      Open a PR if you will integrate this programatic generation into the cfg.
* we have also implemented `lorenz.py`, the famous [chaotic system](https://en.wikipedia.org/wiki/Lorenz_system) to show the divergence of another no noise, 3 dimensions nonlinear system.
Because there are no actions, all of the training and evaluation is handled in that specific file. 
  Instead of using `evaluate.py`, use `lorenz.py` with `mode=plot plot=true'
---
# Running the main repo code
For the original paper, visit the master branch.

To run the code use the following steps:

1. Create a conda environment from the provided yml file and activate it
2. Installing mujoco will fail. See the repo for instructions: https://github.com/openai/mujoco-py

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
- `mbrl_resource`: Other functions used for iterative data collection.

## Replicating Experiments:

For questions on configurations, see [Hyrda](https://hydra.cc/).

### Long-term prediction, section 5.2
This section has multiple files (`reacher_pd.py`, `cartpole_lqr.py`, `crazyflie_pd.py`, `crazyflie_hardware.py`) to collect data and train models, and a central file to evaluate results (`evaluate.py`). Because of a slightly different space (using hardware), `crazyflie_hardware.py` evaulates results by running it with `mode=eval`. An important config item is `data_dir` as this is where data will be saved, models will be saved from, and `evaluate.py` will test from.

Collect simulated data: `python reacher_pd.py models=t envs=reacher mode=collect`

Train models: `python reacher_pd.py models=t envs=reacher mode=train` or a sweep with multiple models `python reacher_pd.py -m models=d,de,t,te envs=reacher mode=train`


### Predicting unstable and period data, section 5.3

For this experiment, procesd as above, but the `data_dir` needs to be changed in the cartpole configuration file. Also, the `data_mode` in `conf/envs/cartpole.yaml` must be changed correspondingly.
The three datasets to be used are:
- Stable data: `trajectories/cartpole/rawl200_t100_v4.dat`
- Unstable data: `trajectories/cartpole/rawl200_t100_unstable.dat`
- Periodic data:`trajectories/cartpole/rawl200_t100_chaotic.dat`
These files can of course be recollected.


### Data efficiency, section 5.4

Example of how to run efficiency code to train some models and then test them (this experiment is more computationally intensive):

Train: `python3 efficiency.py training.num_traj=3,5,7,9 training.t_range=10,20,30,40 models=d,t training.copy=1,2,3,4,5 -m`

Test: `python3 efficiency.py mode=plot plotting.num_traj=[3,5,7,9] plotting.t_range=[10,20,30,40] plotting.models=[d,t] plotting.copy=[1,2,3,4,5] -m`

### Predicting reward, section 5.5

This example uses the file `reward_rank.py`. To run this, run `python reward_rank.py envs=cartpole`. It is currently not supported for any other environments.
