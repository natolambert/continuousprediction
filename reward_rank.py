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

import torch
import numpy as np

from plot import *
from evaluate import test_models
import gpytorch

log = logging.getLogger(__name__)

def get_reward_reacher(state, action):
    # Copied from the reacher env, without self.state calls
    vec = state[-3:]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum() * 0.01
    reward = reward_dist + reward_ctrl
    return reward

def get_reward(predictions, actions):
    # takes in the predicted trajectory and returns the reward
    rewards = {}
    num_traj = len(actions)
    for m_label, state_data in predictions.items():
        r = []
        for i in range(num_traj):
            r_sub = 0
            cur_states = state_data[i]
            cur_actions = actions[i]
            for s,a in zip(cur_states, cur_actions):
                # TODO need a specific get reward function for the reacher env
                r_sub += get_reward_reacher(s,a)
            r.append(r_sub)
        rewards[m_label] = (r, np.mean(r), np.std(r))

    return rewards

def pred_traj(model, control):
    # for a one-step model, predicts a trajectory from initial state all in simulation
    return 0

def train_gp(data):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_x = data[0]
    train_y = data[1]
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood

def predict_gp(test_x, model, likelihood):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    if False:
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

    return observed_pred


@hydra.main(config_path='conf/rewards.yaml')
def evaluate(cfg):
    # print("here")
    if cfg.env == 'lorenz':
        raise ValueError("No Reward in Lorenz System")
    graph_file = 'Plots'
    os.mkdir(graph_file)


    trajectories = torch.load(hydra.utils.get_original_cwd()+'/trajectories/reacher/'+cfg.data_dir)

    # TODO Three test cases, different datasets
    # create 3 datasets of 50 trajectories.
    # 1. same initial state, same goal. Vary PID, target parameters <- simplest case
    # 2. same initial state, different goals. Vary PID parameters
    # 3. different intial state, different goal. Hardest to order reward because potential reward is different

    f = hydra.utils.get_original_cwd() + '/models/reacher/'
    model_one = torch.load(f+cfg.step_model+'.dat')
    model_traj = torch.load(f+cfg.traj_model+'.dat')

    # get rewards, control policy, etc for each type, and control parameters
    data = trajectories[0]+trajectories[1]
    reward = [t['rewards'] for t in data]
    control = [np.concatenate((t['D'],t['P'],t['target'])) for t in data]
    states = [t['states'] for t in data]
    actions = [t['actions'] for t in data]

    # fit GP model to rewards
    split = int(len(data)*cfg.split)
    gp_x = [torch.Tensor(np.concatenate((s[0], c))) for s,c in zip(states,control)]
    gp_y = torch.Tensor(np.sum(np.stack(reward),axis=1))
    gp_x_train = gp_x[:split]
    gp_y_train = gp_y[:split]
    model, likelihood = train_gp((torch.stack(gp_x_train), gp_y_train))
    gp_x_test = gp_x[split:]
    gp_y_test = gp_y[split:] # TRUE REWARD
    gp_pred = predict_gp(torch.stack(gp_x_test), model, likelihood)
    mean_pred = gp_pred.mean
    err = (gp_y_test-mean_pred)**2

    # predict with one step and traj model
    models = {
        'p': model_one,
        't': model_traj,
    }
    MSEs, predictions = test_models(data[split:], models)

    # get dict of rewards for type of model
    pred_rewards = get_reward(predictions, actions[split:])

    # Load test data
    log.info(f"Loading default data")
    (train_data, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    # Load models
    log.info("Loading models")
    if cfg.plotting.copies:
        model_types = list(itertools.product(cfg.plotting.models, np.arange(cfg.plotting.copies)))
    else:
        model_types = cfg.plotting.models
    models = {}
    f = hydra.utils.get_original_cwd() + '/models/reacher/'
    if cfg.exper_dir:
        f = f + cfg.exper_dir + '/'
    for model_type in model_types:
        model_str = model_type if type(model_type) == str else ('%s_%d' % model_type)
        models[model_type] = torch.load(f + model_str + ".dat")


if __name__ == '__main__':
    sys.exit(evaluate())
