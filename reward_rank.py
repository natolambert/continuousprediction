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
from mbrl_resources import obs2q
log = logging.getLogger(__name__)

def get_reward_reacher(state, action):
    # Copied from the reacher env, without self.state calls
    vec = state[-3:]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum() * 0.01
    reward = reward_dist # + reward_ctrl
    return reward

def get_reward_cp(state, action):
    # custom reward for sq error from x=0, theta = 0
    reward = state[0]**2 + state[2]**2
    return -reward

def get_reward_cf(state, action):
    # custom reward for sq error from x=0, theta = 0
    # reward = np.cos(state[4])*np.cos(state[3])
    # if (np.rad2deg(state[3]) < 5) and (np.rad2deg(state[4]) < 5):
    #     reward = 1
    # else:
    #     reward = 0
    reward = -state[3]**2 - state[4]**2
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
            for s,a in zip(cur_states, cur_actions):
                # TODO need a specific get reward function for the reacher env
                r_sub += r_function(s,a)
            r.append(r_sub)
        rewards[m_label] = (r, np.mean(r), np.std(r))

    return rewards

def pred_traj(test_data, models, control = None, env=None, cfg = None, t_range=None):
    # for a one-step model, predicts a trajectory from initial state all in simulation
    log.info("Beginning testing of predictions")

    states, actions, initials = [], [], []

    if env == 'reacher' or env == 'crazyflie':
        P, D, target = [], [], []

        # Compile the various trajectories into arrays
        for traj in test_data:
            states.append(traj.states)
            actions.append(traj.actions)
            initials.append(traj.states[0, :])
            P.append(traj.P)
            D.append(traj.D)
            target.append(traj.target)

        P_param = np.array(P)
        P_param = P_param.reshape((len(test_data), -1))
        D_param = np.array(D)
        D_param = D_param.reshape((len(test_data), -1))
        target = np.array(target)
        target = target.reshape((len(test_data), -1))

        parameters = [[P[0], 0, D[0]],
                      [P[1], 0, D[1]]]
        if env == 'crazyflie':
            from crazyflie_pd import PidPolicy
            policy = PidPolicy(parameters, cfg.pid)
            policies = []
            for p,d in zip(P_param, D_param):
                policies.append(PidPolicy([[p[0], 0, d[0]],[p[1], 0, d[1]]],cfg.pid))
            # policies = [LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0]) for i in range(len(test_data))]


    elif env == 'cartpole':
        K = []

        # Compile the various trajectories into arrays
        for traj in test_data:
            states.append(traj.states)
            actions.append(traj.actions)
            initials.append(traj.states[0, :])
            K.append(traj.K)

        K_param = np.array(K)
        K_param = K_param.reshape((len(test_data), -1))

        # create LQR controllers to propogate predictions in one-step
        from policy import LQR

        # These values are replaced an don't matter
        m_c = 1
        m_p =1
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

        n_dof = np.shape(A)[0]
        modifier = .5 * np.random.random(
            4) + 1  # np.random.random(4)*1.5 # makes LQR values from 0% to 200% of true value
        policies = [LQR(A, B.transpose(), Q, R, actionBounds=[-1.0, 1.0]) for i in range(len(test_data))]
        for p, K in zip(policies, K_param):
            p.K = K

    # Convert to numpy arrays
    states = np.stack(states)
    actions = np.stack(actions)


    initials = np.array(initials)
    N, T, D = states.shape
    if len(np.shape(actions)) == 2:
        actions = np.expand_dims(actions, axis=2)
    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}

    ind_dict = {}
    for i, key in list(enumerate(models)):
        model = models[key]
        if model.traj:
            raise ValueError("Traj model conditioned on predicted states is invalid")
        indices = model.state_indices
        traj = model.traj

        ind_dict[key] = indices

        for i in range(1, T):
            if i >= t_range:
                continue
            # if control:
                # policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
                # act, t = control.act(obs2q(currents[key]))
            if env == 'crazyflie':
                acts = np.stack([[p.get_action(currents[key][i,3:6])] for i,p in enumerate(policies)]).reshape(-1,4)
            else:
                acts = np.stack([[p.act(obs2q(currents[key][i,:]))[0]] for i,p in enumerate(policies)])

            prediction = model.predict(np.hstack((currents[key], acts)))
            prediction = np.array(prediction.detach())

            predictions[key].append(prediction)
            currents[key] = prediction.squeeze()

    predictions = {key: np.array(predictions[key]).transpose([1, 0, 2]) for key in predictions}
    # MSEs = {key: np.square(states[:, :, ind_dict[key]] - predictions[key]).mean(axis=2)[:, 1:] for key in predictions}


    return 0, predictions

def train_gp(data):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.RBFKernel()
            # self.scaled_mod = gpytorch.kernels.ScaleKernel(self.covar_module)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            # covar_x = self.scaled_mod(x)
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
    ], lr=.1) # was .1

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

def predict_gp(test_x, model, likelihood, train_x=None, train_y=None):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        # observed_pred = model(test_x)

    if False:
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            if train_x and train_y:
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

    return observed_pred


@hydra.main(config_path='conf/rewards.yaml')
def reward_rank(cfg):
    # print("here")
    if cfg.env == 'lorenz':
        raise ValueError("No Reward in Lorenz System")

    label = cfg.env.label
    graph_file = 'Plots'
    os.mkdir(graph_file)

    trajectories = torch.load(hydra.utils.get_original_cwd()+'/trajectories/'+label+'/raw'+cfg.data_dir)

    # TODO Three test cases, different datasets
    # create 3 datasets of 50 trajectories.
    # 1. same initial state, same goal. Vary PID, target parameters <- simplest case
    # 2. same initial state, different goals. Vary PID parameters
    # 3. different intial state, different goal. Hardest to order reward because potential reward is different

    f = hydra.utils.get_original_cwd() + '/models/' +label +'/'
    model_one = torch.load(f+cfg.step_model+'.dat')
    model_traj = torch.load(f+cfg.traj_model+'.dat')

    # get rewards, control policy, etc for each type, and control parameters
    data_train = trajectories[0]#[::10] #+trajectories[1]
    reward = [t['rewards'] for t in data_train]
    states = [np.float32(t['states']) for t in data_train]
    actions = [np.float32(t['actions']) for t in data_train]
    if cfg.model.training.t_range < np.shape(states)[1]:
        states = [s[:cfg.model.training.t_range,:] for s in states]
        actions = [a[:cfg.model.training.t_range,:] for a in actions]

    if label == 'reacher':
        control = [np.concatenate((t['D'],t['P'],t['target'])) for t in data_train]
        r_func = get_reward_reacher
    elif cfg.env.label == 'cartpole':
        for vec_s, vec_a in zip(states, actions):
            vec_s[0, 1] = vec_s[0, 1].item()
            vec_s[0, 3] = vec_s[0, 3].item()
            vec_a[1] = vec_a[1].item()
        control = [t['K'] for t in data_train]
        r_func = get_reward_cp
    elif cfg.env.label=='crazyflie':
        control = [np.concatenate((t['D'],t['P'],t['target'])) for t in data_train]
        r_func = get_reward_cf

    reward = [np.sum([r_func(s,a) for s,a in zip(sta,act)]) for sta,act in zip(states,actions)]

    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.utils import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    # train_X = torch.rand(10, 2)
    # Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
    # Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
    # train_Y = standardize(Y)



    # fit GP model to rewards
    split = int(len(data_train)*cfg.split)
    gp_x =  [torch.Tensor(np.concatenate((s[0], c))) for s,c in zip(states,control)]
    #[torch.Tensor(np.concatenate((np.array([np.asscalar(np.array(i)) for i in s[0]])), c)) for s,c in zip(states,control)]
    if label == 'reacher':
        # gp_y = torch.Tensor(np.sum(np.stack(reward),axis=1))
        gp_y = torch.Tensor(reward)
    else:
        gp_y = torch.Tensor(reward)

    gp_x_train = gp_x[:split]
    gp_y_train = gp_y[:split]
    # model, likelihood = train_gp((torch.stack(gp_x_train), gp_y_train))

    log.info(f"Training GP model (can be slow)")
    gp = SingleTaskGP(torch.stack(gp_x_train), gp_y_train.reshape(-1,1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # gp_x_test = gp_x[split:]
    # gp_y_test = gp_y[split:] # TRUE REWARD
    # gp_pred = predict_gp(torch.stack(gp_x_test), model, likelihood, train_x=torch.stack(gp_x_train), train_y=np.stack(gp_y_train))

    # TODO neural network supervised learning
    # from dynamics_model import Net
    # supervised = Net(np.shape(gp_x)[1], np.shape(reward)[1], env=None)

    ## TEST SET WORK
    log.info("Testing models")
    data_test = trajectories[0] #[::10]
    reward = [t['rewards'] for t in data_test]
    states = [np.float32(t['states']) for t in data_test]
    actions = [np.float32(t['actions']) for t in data_test]
    if cfg.model.training.t_range < np.shape(states)[1]:
        states = [s[:cfg.model.training.t_range,:] for s in states]
        actions = [a[:cfg.model.training.t_range,:] for a in actions]

    if label == 'reacher':
        control = [np.concatenate((t['D'], t['P'], t['target'])) for t in data_test]
        r_func = get_reward_reacher
    elif cfg.env.label == 'cartpole':
        for vec_s, vec_a in zip(states, actions):
            vec_s[0, 1] = vec_s[0, 1].item()
            vec_s[0, 3] = vec_s[0, 3].item()
            vec_a[1] = vec_a[1].item()
        control = [t['K'] for t in data_test]
        r_func = get_reward_cp
    elif cfg.env.label == 'crazyflie':
        control = [np.concatenate((t['D'], t['P'], t['target'])) for t in data_test]
        r_func = get_reward_cf

    reward = [np.sum([r_func(s, a) for s, a in zip(sta, act)]) for sta, act in zip(states, actions)]

    log.info("Evaluating GP")
    gp_x_test = [torch.Tensor(np.concatenate((s[0], c))) for s,c in zip(states,control)]
    # gp_pred_test = predict_gp(torch.stack(gp_x_test), model, likelihood)
    # gp_pred_test = gp.forward(torch.stack(gp_x_test))
    gp_pred_test = gp.posterior(torch.stack(gp_x_test))
    # predict with one step and traj model
    models = {
        'p': model_one,
        't': model_traj,
    }
    MSEs, predictions = test_models(data_test, models, env = cfg.env.label, t_range = cfg.model.training.t_range)

    # get dict of rewards for type of model
    pred_rewards = get_reward(predictions, actions, r_func)


    models_step = {
        'p': model_one,
    }

    _, pred_drift = pred_traj(data_test, models_step, env = cfg.env.label, cfg=cfg, t_range = cfg.model.training.t_range)

    # get dict of rewards for type of model
    if cfg.model.training.t_range < np.shape(states)[1]:
        for key in pred_drift:
            pred_drift[key] = pred_drift[key][:,:cfg.model.training.t_range,:]
        actions = [a[:cfg.model.training.t_range,:] for a in actions]

    pred_rewards_true = get_reward(pred_drift, actions, r_func)


    if label == 'reacher' or label == 'crazyflie':
        cum_reward = [np.sum(rew) for rew in reward]
    else:
        cum_reward = reward

    gp_pr_test = gp_pred_test.mean.detach().numpy()
    nn_step_oracle = pred_rewards['p'][0]
    nn_step_drift = pred_rewards_true['p'][0]
    nn_traj = pred_rewards['t'][0]
    # Load test data
    print(f"Mean GP reward err: {np.mean((cum_reward-np.array(gp_pr_test))**2)}")
    print(f" - std dev:{np.std(gp_pr_test-cum_reward)}")
    print(f"Mean one step oracle reward err: {np.mean((cum_reward-np.array(nn_step_oracle))**2)}")
    print(f" - std dev:{np.std(np.array(nn_step_oracle)-cum_reward)}")
    print(f"Mean one step reward err: {np.mean((cum_reward-np.array(nn_step_drift))**2)}")
    print(f" - std dev:{np.std(np.array(nn_step_drift)-cum_reward)}")
    print(f"Mean traj reward err: {np.mean((cum_reward-np.array(nn_traj))**2)}")
    print(f" - std dev:{np.std(np.array(nn_traj)-cum_reward)}")

    arr = np.stack(sorted(zip(cum_reward, gp_pr_test, np.array(nn_step_oracle), np.array(nn_traj), np.array(nn_step_drift))))
    # arr = np.stack((cum_reward, gp_pr_test, nn_step, nn_traj))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(cum_reward, label='True Predicted Reward',  clip_on = True, ms=14, markevery=16, marker='o' )
    ax.plot(gp_pr_test, label='Gaussian Process',  clip_on = True, ms=14, markevery=19, marker='*' )
    ax.plot(nn_step_oracle, label='One-step Neural Network',  clip_on = True, ms=14, markevery=22, marker='+' )
    ax.plot(nn_traj, label='Trajectory-based Model',  clip_on = True, ms=14, markevery=25, marker='d' )

    # ax.plot(arr[:, 0], label='True Predicted Reward',  clip_on = True, ms=14, markevery=16, marker='o' )
    # ax.plot(arr[:, 1], label='Gaussian Process',  clip_on = True, ms=14, markevery=19, marker='*' )
    # ax.plot(arr[:, 2], label='One-step Neural Network',  clip_on = True, ms=14, markevery=22, marker='+' )
    # ax.plot(arr[:, 3], label='Trajectory-based Model',  clip_on = True, ms=14, markevery=25, marker='d' )
    # plt.plot(arr[:, 4], label='step-drift')
    import matplotlib
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    font = {'size': 18, 'family': 'serif', 'serif': ['Times New Roman']}
    font_axes = {'size': 18, 'family': 'serif'}
    matplotlib.rc('font', **font)
    # plt.xlabel('Sorted Trajectory', fontdict=font_axes)
    # plt.ylabel('Cumulative Episode Reward', fontdict=font)

    plt.yticks(fontname="Times New Roman")  # This argument will change the font.
    plt.xticks(fontname="Times New Roman")  # This argument will change the font.

    ax.set_xlabel('Sorted Trajectory', fontdict=font_axes)
    ax.set_ylabel('Cumulative Episode Reward', fontdict=font_axes)
    ax.set_xlim([0,100])
    if cfg.env.label=='cartpole':
        ax.set_ylim([-900,0])
    ax.legend()
    fig.tight_layout()
    fig.savefig("Reward_Predictions.pdf")

    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    # fig.savefig("Reward Predictions Error.png")



if __name__ == '__main__':
    sys.exit(reward_rank())
