"""
The purpose of this file is to load in pre-loaded data and pre-trained models and evaluate them
"""

import sys

import hydra
import logging
import itertools

import torch
import numpy as np

from plot import *
from mbrl_resources import obs2q

log = logging.getLogger(__name__)


def forward_var(model, x):
    assert model.prob, "only probablistic models have var"
    if type(x) == np.ndarray:
        x = torch.from_numpy(np.float64(x))
    variance = torch.zeros((x.shape[0], len(model.state_indices)))
    for n in model.nets:
        scaledInput = n.testPreprocess(x, model.cfg)
        variance += n.forward(scaledInput)[:, len(model.state_indices):] / len(model.nets)
        # prediction += n.forward(scaledInput)[:, :len(self.state_indices)] / len(self.nets)
    # return torch.sqrt(torch.exp(variance))
    return torch.exp(variance)


def test_models(test_data, models, verbose=False, env=None, compute_action=False, ret_var=False, t_range=np.inf):
    """
    Tests each of the models in the dictionary "models" on each of the trajectories in test_data.
    Note: this function uses Numpy arrays to handle multiple tests at once efficiently

    Parameters:
    ------------
    test_data: the trajectories to test on, N trajectories
    models: a dictionary of models to test, M models

    Returns:
     MSEs:           MSEs['x'] is a 2D array where the (i,j)th is the MSE for
                        the prediction at time j with the ith test trajectory
                        corresponding to model 'x'
     predictions:   predictions['x'] is a 3D array where the (i,j)th element
                        is an array with the predicted state at time j for the ith
                        test trajectory corresponding to model 'x'
    """

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

    if env == 'lorenz':
        P, D, target = [], [], []
        for traj in test_data:
            states.append(traj.states)
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

        states = np.stack(states)

    elif env == 'ss':
        for traj in test_data:
            states.append(traj.states)
            actions.append(traj.actions)
            initials.append(traj.states[0, :])

        states = np.stack(states)
        actions = np.stack(actions)

    else:
        # Convert to numpy arrays
        states = np.stack(states)
        actions = np.stack(actions)

    if compute_action:
        # create LQR controllers to propogate predictions in one-step
        from policy import LQR, PID
        if env == 'reacher':
            policies = [
                PID(dX=5, dU=5, P=P_param[i, :], I=np.array([0, 0, 0, 0, 0]), D=D_param[i, :], target=target[i, :]) for
                i in range(len(test_data))]

        elif env == 'crazyflie':
            from crazyflie_pd import PidPolicy
            parameters = [[P[0], 0, D[0]],
                          [P[1], 0, D[1]]]
            policy = PidPolicy(parameters, cfg.pid)
            policies = [
                PID(dX=5, dU=5, P=P_param[i, :], I=np.array([0, 0, 0, 0, 0]), D=D_param[i, :], target=target[i, :]) for
                i in range(len(test_data))]
        elif env == 'cartpole':

            # These values are replaced an don't matter
            m_c = 1
            m_p = 1
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

    initials = np.array(initials)
    N, T, D = states.shape
    A = actions.shape[-1]
    if len(np.shape(actions)) == 2:
        actions = np.expand_dims(actions, axis=2)
    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}

    variances = {key: [] for key in models}
    ind_dict = {}
    for i, key in list(enumerate(models)):
        if verbose and (i + 1) % 10 == 0:
            print("    " + str(i + 1))
        model = models[key]
        indices = model.state_indices
        traj = model.traj
        lstm = "lstm" in key or "rnn" in key #model.cfg.model.lstm

        ind_dict[key] = indices

        # # temp for plotting one-step
        # if i == 1:
        #     compute_action = False
        # elif i > 1:
        #     continue

        for i in range(1, T):
            # print(i)
            if i >= t_range:
                continue
            if lstm:
                # TODO translate to lstm code
                if traj:
                    raise NotImplementedError("Not supporting traj lstm yet")
                    dat = [initials[:, indices], i * np.ones((N, 1))]
                    if env == 'reacher' or env == 'lorenz' or env == 'crazyflie':
                        if model.control_params:
                            dat.extend([P_param, D_param])
                        if model.train_target:
                            dat.append(target)
                    elif env == 'cartpole':
                        dat.append(K_param)
                    prediction = np.array(model.predict(np.hstack(dat)).detach())

                else:
                    if i == 1:
                        if env == 'cartpole': A=1
                        actions_lstm = actions[:, i - 1, :].reshape(1, N, A)
                        states_lstm = currents[key].reshape(1, N, len(models[key].state_indices))
                    else:
                        actions_lstm = actions[:, :i, :].transpose(1, 0, 2)
                        states_lstm = np.concatenate((states_lstm, prediction), axis=0)
                        if True:
                            train_len = model.cfg.model.optimizer.batch
                            if np.shape(actions_lstm)[0]>train_len:
                                actions_lstm = actions_lstm[-train_len:]
                                states_lstm = states_lstm[-train_len:]
                    # input of shape (seq_len, batch, input_size)
                    # output of shape (seq_len, batch, input_size)
                    if env == 'lorenz':
                        raise NotImplementedError("TODO")
                        prediction = model.predict(np.array(currents[key]))
                        prediction = np.array(prediction.detach())
                    else:
                        prediction = model.predict_lstm(np.concatenate((states_lstm, actions_lstm), axis=2), num_traj=N)
                        prediction = np.array(prediction.detach())[-1, :, :].reshape(1, N,
                                                                                      len(models[key].state_indices))

                # included for not erroring (hacky, not used)
                var = np.zeros(np.shape(initials[:, indices]))
                variances[key].append(var)

                # Note - no probablistic LSTM models for now
                predictions[key].append(prediction[0])
                currents[key] = prediction.squeeze()
            else:

                if traj:
                    dat = [initials[:, indices], i * np.ones((N, 1))]
                    if env == 'reacher' or env == 'lorenz' or env == 'crazyflie':
                        if model.control_params:
                            dat.extend([P_param, D_param])
                        if model.train_target:
                            dat.append(target)
                    elif env == 'cartpole':
                        dat.append(K_param)
                    prediction = np.array(model.predict(np.hstack(dat)).detach())


                else:
                    if env == 'lorenz':
                        prediction = model.predict(np.array(currents[key]))
                        prediction = np.array(prediction.detach())
                    else:
                        if compute_action:
                            if env == 'cartpole':
                                acts = np.stack(
                                    [[p.act(obs2q(currents[key][i, :]))[0]] for i, p in enumerate(policies)])
                            else:
                                acts = np.stack(
                                    [[p.act(obs2q(currents[key][i, :]))[0]][0] for i, p in enumerate(policies)])
                        else:
                            acts = actions[:, i - 1, :]
                        prediction = model.predict(np.hstack((currents[key], acts)))
                        prediction = np.array(prediction.detach())

                # get variances if applicable
                if model.prob:
                    if traj:
                        f = np.hstack(dat)
                    else:
                        f = np.hstack((currents[key], acts))
                    var = forward_var(model, f).detach().numpy()
                else:
                    var = np.zeros(np.shape(initials[:, indices]))

                predictions[key].append(prediction)
                currents[key] = prediction.squeeze()
                variances[key].append(var)

    variances = {key: np.stack(variances[key]).transpose([1, 0, 2]) for key in variances}
    predictions = {key: np.array(predictions[key]).transpose([1, 0, 2]) for key in predictions}

    # MSEs = {key: np.square(states[:, :, ind_dict[key]] - predictions[key]).mean(axis=2)[:, 1:] for key in predictions}

    MSEscaled = {}
    for key in predictions:
        # scaling of error
        if env == 'crazyflie':
            # ind_dict[key] = [0,1,3,4,5]
            ind_dict[key] = [0, 1, 3, 4]
            pred_key = [0, 1, 3, 4]
        elif env == 'reacher':
            ind_dict[key] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17]
            pred_key = np.arange(np.shape(predictions[key][0])[1])
        else:
            ind_dict[key] = np.arange(D) #np.shape(prediction)[1])
            pred_key = np.arange(D) #np.shape(predictions[key][0])[1])  # changed from np.arange(np.shape(prediction)[1])
        if t_range < np.shape(states)[1]:
            l = t_range
        else:
            l = np.shape(states)[1]
        min_states = np.min(states[:, :l, ind_dict[key]], axis=(0, 1))
        max_states = np.ptp(states[:, :l, ind_dict[key]], axis=(0, 1))
        scaled_states = (states[:, :l, ind_dict[key]] - min_states) / max_states
        scaled_pred = (predictions[key][:, :, pred_key] - min_states) / max_states
        MSEscaled[key] = np.square(scaled_states - scaled_pred).mean(axis=2)[:, 1:]
        # print(key)
        # print(np.sum(np.sum(MSEscaled[key])))
    # MSEs = {key: np.array(MSEs[key]).transpose() for key in MSEs}
    # if N > 1:
    #     predictions = {key: np.array(predictions[key]).transpose([1,0,2]) for key in predictions} # vectorized verion
    # else:
    #     predictions = {key: np.stack(predictions[key]).squeeze() for key in predictions}

    # outcomes = {'mse': MSEs, 'predictions': predictions}
    if ret_var:
        return MSEscaled, predictions, variances
    else:
        return MSEscaled, predictions


def test_traj_ensemble(ensemble, test_data):
    """
    TODO: decide if this is useful or remove
    Tests each model in the ensemble on one test trajectory and plots the output
    """
    traj = test_data
    states = traj.states
    actions = traj.actions
    initial = states[0, :]

    model_predictions = [[] for _ in range(ensemble.n)]
    ensemble_predictions = []
    for i in range(1, states.shape[0]):
        x = np.hstack((initial, i, traj.P, traj.D, traj.target))
        ens_pred = ensemble.predict(x)
        ensemble_predictions.append(ens_pred.squeeze())
        for j in range(len(ensemble.models)):
            model = ensemble.models[j]
            model_pred = model.predict(x)
            model_predictions[j].append(model_pred.squeeze())

    ensemble_predictions = np.array(ensemble_predictions)
    model_predictions = [np.array(x) for x in model_predictions]
    # print(len(model_predictions))

    for i in range(7):
        fig, ax = plt.subplots()
        gt = states[:, i]
        plt.title("Predictions on one dimension")
        plt.xlabel("Timestep")
        plt.ylabel("State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot(gt, c='k', label='Groundtruth')
        plt.plot(ensemble_predictions[:, i])
        for pred in model_predictions:
            # print(pred.shape)
            plt.plot(pred[:, i], c='b')

        plt.legend()

        plt.show()


def find_deltas(test_data, models):
    """
    For sorted delta plots. Tests each model in 'models' on each test trajectory in 'test_data',
    finding the predicted deltas. The difference between this method and the standard test_models
    is that with this one, each prediction starts from a ground truth value

    Parameters:
        test_data: the N test trajectories to test on
        models: the M models to evaluate

    Returns:
        deltas_gt: ground truth deltas for each model
        deltas_pred: predicted deltas for each model

    """
    states, actions = [], []

    # Compile the various trajectories into arrays
    for traj in test_data:
        states.append(traj.states)
        actions.append(traj.actions)

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)

    N, T, D = states.shape

    # Iterate through each type of model for evaluation
    deltas_gt, deltas_pred = {}, {}
    for key in models:
        model = models[key]
        indices = model.state_indices
        if 't' in key or type(key) == tuple and 't' in key[0]:
            # This doesn't make sense for t models so not gonna bother with this
            continue
        else:
            inp = np.dstack((states[:, :, indices], actions))
            prediction = model.predict(inp.reshape((N * T, -1))).detach().numpy().reshape((N, T, -1))
            delta_pred = prediction[:, :-1, :] - states[:, :-1, indices]
            delta_gt = states[:, 1:, indices] - states[:, :-1, indices]
            deltas_gt[key] = delta_gt
            deltas_pred[key] = delta_pred

            # input = np.dstack((states[:, :, indices], actions)).reshape(N*T, -1)
            # prediction = model.predict(input)
            # prediction = np.array(prediction.detach()).reshape((N, T, len(indices)))
            # delta = prediction-states[:, :, indices]
            # deltas[key] = delta

    return deltas_gt, deltas_pred


def num_eval(gt, predictions, models, setting='gaussian', T_range=10000, verbose=False):
    """
    Evaluates the predictions in a way that creates one number

    Parameters:
        gt: NxTxD array of ground truth values
        predictions: a dictionary of NxTxD arrays of predictions from models
        setting: currently 'dot', 'mse'
            'dot': average over dimensions of dot product between ground truth and trajectories
            'mse': average over dimensions of MSE

    Returns:
        outputs: a dictionary of arrays of length N of evaluation
    """
    gt = gt[:, :T_range, :]
    predictions = {key: predictions[key][:, :T_range, :] for key in predictions}

    out = {}
    for i, model_type in list(enumerate(models)):
        if (i + 1) % 10 == 0 and verbose:
            print(i + 1)
        gt_subset = gt[:, :, models[model_type].state_indices]
        if setting == 'dot':
            N, T, D = gt_subset.shape
            gt_norm = gt_subset / np.linalg.norm(gt_subset, axis=1).reshape((N, 1, D))
            prediction_norm = predictions[model_type] / np.linalg.norm(predictions[model_type], axis=1).reshape(
                (N, 1, D))
            out[model_type] = np.sum(prediction_norm * gt_norm, axis=(1, 2)) / D
        elif setting == 'mse':
            out[model_type] = np.mean((predictions[model_type] - gt_subset) ** 2, axis=(1, 2))
        elif setting == 'gaussian':
            diff = 5 * (gt_subset - predictions[model_type])
            gauss = np.exp(-1 * np.square(diff))
            out[model_type] = np.mean(gauss, axis=(1, 2))
            # diff_dict = {key: gt_subset - predictions[key] for key in predictions}
            # gauss_dict = {key: np.exp(-1 * np.square(diff_dict[key])) for key in diff_dict}
            # out = {key: np.mean(gauss_dict[key], axis=(1, 2)) for key in gauss_dict}
        else:
            raise ValueError("Invalid setting: " + setting)
    return out


@hydra.main(config_path='conf/eval.yaml')
def evaluate(cfg):
    # print("here")
    name = cfg.env.label
    graph_file = 'Plots'
    os.mkdir(graph_file)

    if not name == 'lorenz':
        # Load test data
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/' + cfg.env.label + '/' + 'raw' + cfg.data_dir)

        if cfg.plotting.train_set:
            test_data = train_data

        # Load models
        log.info("Loading models")
        if cfg.plotting.copies:
            model_types = list(itertools.product(cfg.plotting.models, np.arange(cfg.plotting.copies)))
        else:
            model_types = cfg.plotting.models
        models = {}
        if cfg.data_mode_plot != 'stable':
            f = hydra.utils.get_original_cwd() + '/models/' + cfg.env.label + '/' + cfg.data_mode_plot + '/'
        else:
            f = hydra.utils.get_original_cwd() + '/models/' + cfg.env.label + '/'
        if cfg.exper_dir:
            f = f + cfg.exper_dir + '/'
        for model_type in model_types:
            model_str = model_type if type(model_type) == str else ('%s_%d' % model_type)
            models[model_type] = torch.load(f + model_str + ".dat")

    else:
        # # Load test data
        # Below was copied from lorenz... strange
        # log.info(f"Loading default data")
        # (train_data, test_data) = torch.load(hydra.utils.get_original_cwd() + '/trajectories/'+ cfg.env.label + '/' + 'raw' + cfg.data_dir_lorenz)

        # Load models
        log.info("Loading models")
        model_types = cfg.plotting.models
        models = {}
        f = hydra.utils.get_original_cwd() + '/models/' + cfg.env.label + '/'
        for model_type in model_types:
            if 'gp' in model_type:
                from GPy.core.model import load_model
                models[model_type] = load_model(f + model_type + ".dat")
            else:
                models[model_type] = torch.load(f + model_type + ".dat")

    if cfg.plotting.plot_states:
        plot_states_dist(test_data)

    # Plot
    def plot_helper(data, num, graph_file):
        """
        Helper to allow plotting for both train and test data without significant code duplication
        """
        os.mkdir(graph_file)

        # Select a random subset of training data
        # idx = np.random.randint(0, len(data), num)
        idx = np.random.choice(np.arange(len(data)), size=num, replace=False)
        dat = [data[i] for i in idx]

        for entry in dat:
            entry.states = entry.states[0:cfg.plotting.t_range]
            entry.rewards = entry.rewards[0:cfg.plotting.t_range]
            entry.actions = entry.actions[0:cfg.plotting.t_range]

        MSEs, predictions, variances = test_models(dat, models, env=name, compute_action=cfg.plotting.compute_action,
                                                   ret_var=True)

        setup_plotting(models)
        mse_evald = []
        sh = MSEs[model_types[0]][0].shape
        for i, id in list(enumerate(idx)):
            gt = data[id].states
            if cfg.plotting.copies:
                mse_all = {key: np.zeros((cfg.plotting.copies,) + sh) for key in cfg.plotting.models}
                for type, j in MSEs:
                    mse_all[type][j] = MSEs[(type, j)][i]
                mse = {key: np.median(mse_all[key], axis=0) for key in mse_all}
            else:
                mse = {key: MSEs[key][i].squeeze() for key in MSEs}
            mse_sub = {key: [(x if x < 10 ** 5 else float("nan")) for x in mse[key]] for key in mse}
            if not cfg.plotting.copies:
                pred = {key: predictions[key][i] for key in predictions}
                var = {key: variances[key][i] for key in variances}
            if cfg.plotting.all:
                file = "%s/test%d" % (graph_file, i + 1)
                os.mkdir(file)

                # TODO: fix this if it causes bugs
                if name == 'reacher':
                    gt = gt[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17]]
                    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                elif name == 'cartpole':
                    gt = gt[:, [0, 1, 2, 3]]
                    idx = [0, 1, 2, 3]
                elif name == 'crazyflie':
                    # gt = gt[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
                    # idx = [0,1,2,3,4,5,6,7,8,9,10,11]

                    gt = gt[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
                    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]

                if cfg.plotting.states:
                    # if
                    plot_states(gt, pred, variances=var, idx_plot=idx, save_loc=file + "/predictions", show=False)
                if cfg.plotting.mse:
                    plot_mse(mse_sub, save_loc=file + "/mse.pdf", show=False)
                # if cfg.plotting.sorted:
                #     ds = {key: deltas[key][i] for key in deltas}
                #     plot_sorted(gt, ds, idx_plot=[0,1,2,3], save_loc=file+"/sorted", show=False)

            # mse['zero'] = np.zeros(mse[next(iter(mse))].shape)

            mse_evald.append(mse)

        if cfg.plotting.sorted:
            # deltas = find_deltas(dat, models)
            deltas_gt, deltas_pred = find_deltas(dat, models)
            plot_sorted(deltas_gt, deltas_pred, idx_plot=[0, 1, 2, 3], save_loc='%s/sorted' % graph_file, show=False)

        if name == 'reacher' or name == 'crazyflie':
            y_min = .05
        elif name == 'cartpole':
            y_min = .0002
        else:
            y_min = .0001

        plot_mse_err(mse_evald, save_loc=("%s/Err Bar MSE of Predictions" % graph_file),
                     show=True, y_min=y_min, y_max=cfg.plotting.mse_y_max, legend=cfg.plotting.legend)
        # turn show off here

        mse_all = {key: [] for key in cfg.plotting.models}
        if cfg.plotting.copies:
            for key, copy in MSEs:
                mse_all[key].append(MSEs[(key, copy)])
            mse_all = {key: np.stack(mse_all[key]) for key in mse_all}
        mse_all = {key: np.mean(mse_all[key], axis=(1 if cfg.plotting.copies else 0)) for key in mse_all}
        if cfg.plotting.copies:
            mse_all = {key: np.median(mse_all[key], axis=0) for key in mse_all}
        plot_mse(mse_all, log_scale=True, title="Average MSE", save_loc=graph_file + '/mse.pdf', show=False)

    if cfg.plotting.num_eval_train:
        log.info("Plotting train data")

        file = graph_file + "/train_data"

        plot_helper(train_data, cfg.plotting.num_eval_train, file)

    if cfg.plotting.num_eval_test:
        log.info("Plotting test data")

        file = graph_file + '/test_data'

        plot_helper(test_data, cfg.plotting.num_eval_test, file)


if __name__ == '__main__':
    sys.exit(evaluate())
