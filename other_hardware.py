import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dotmap import DotMap

import torch

import hydra
import logging

log = logging.getLogger(__name__)

from dynamics_model import DynamicsModel

from mbrl_resources import create_dataset_step


def create_dataset_step(data, delta=True, t_range=0, is_lstm = False, lstm_batch = 0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        for i in range(states.shape[0] - 1):
            if 'actions' in sequence.keys():
                actions = sequence.actions
                if t_range:
                    actions = actions[:t_range]
                data_in.append(np.hstack((states[i], actions[i])))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
            else:
                data_in.append(np.array(states[i]))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
        if is_lstm:
            remainder = len(data_out)%lstm_batch
            if remainder:
                data_out = data_out[:len(data_out)-remainder]
                data_in = data_in[:len(data_in)-remainder]
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out

def log_hyperparams(cfg):
    log.info(cfg.model.str + ":")
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

def create_datasets(states, actions, seq_len):
    seqs = []
    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []
    for id, (s, a) in enumerate(zip(states, actions)):
        logs.states.append(s)
        logs.actions.append(a)
        if np.any(np.isnan(s)) or np.any(np.isnan(a)):
            print("NAN!")
        if (id + 1) % seq_len == 0:
            logs.states = np.array(logs.states)
            logs.actions = np.array(logs.actions)
            seqs.append(logs)
            logs = DotMap()
            logs.states = []
            logs.actions = []
            logs.rewards = []
            logs.times = []

    dataset_step = create_dataset_step(seqs, delta=True)

    return seqs, dataset_step


@hydra.main(config_path='conf/crazyflie_pd.yaml', strict=False)
def contpred(cfg):
    # Collect data
    # exp_raibert_0			test_raibert_0_replace_sw_off
    # test_raibert_0_hw		test_raibert_0_replace_swing
    # test_raibert_0_replace_stance	test_raibert_0_replace_swing_hw
    # /Users/nato/Downloads/hardware_data

    """
    ['base_velocity.npy', 'foot_pos_robot.npy', 'base_ori_euler.npy', 'base_pos_y.npy', 'base_pos_x.npy', 'base_pos_z.npy', 'data_info.txt', 'base_pos.npy', 'j_pos.npy', 'time.npy', 'action.npy', 'base_ang_vel.npy', 'j_vel.npy', 'contact.npy', 'base_ori_quat.npy']
    base_velocity.npy, shape (3805, 3)
    foot_pos_robot.npy, shape (3805, 12)
    base_ori_euler.npy, shape (3805, 3)
    base_pos_y.npy, shape (3805, 1)
    base_pos_x.npy, shape (3805,)
    base_pos_z.npy, shape (3805, 1)
    base_pos.npy, shape (3805, 3)
    j_pos.npy, shape (3805, 12)
    base_ang_vel.npy, shape (3805, 3)
    j_vel.npy, shape (3805, 12)
    contact.npy, shape (3805, 4)
    base_ori_quat.npy, shape (3805, 4)
    """
    state_dirs = ['base_velocity.npy', 'foot_pos_robot.npy', 'base_ori_euler.npy', 'base_pos_y.npy', 'base_pos_x.npy',
                  'base_pos_z.npy', 'base_pos.npy', 'j_pos.npy',  # 'action.npy', #'time.npy', 'data_info.txt',
                  'base_ang_vel.npy', 'j_vel.npy', 'contact.npy',  # 'base_ori_quat.npy'
                  ]
    action_dir = ['action.npy']

    base_dir = "/Users/nato/Downloads/hardware_data"
    sub_dir = "/exp_raibert_0"

    data = []
    for s in state_dirs:
        dir = base_dir + sub_dir + "/" + s
        data_sub = np.load(dir)
        if len(np.shape(data_sub)) == 1:
            data_sub = data_sub.reshape(-1, 1)
        data.append(data_sub)

    data_stack = data[0]
    for i in range(len(data) - 1):
        data_stack = np.concatenate((data_stack, data[i + 1]), axis=1)

    actions = np.load(base_dir + sub_dir + "/" + action_dir[0])
    logs, data_step = create_datasets(data_stack, actions, cfg.model.training.t_range)
    # torch.save()



    # setup_plotting({cfg.model.str: model})
    # plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)
    if cfg.mode == 'train':

        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")
        log_hyperparams(cfg)

        # for i in it:
        #     print('Training model %d' % i)
        #     if traj:
        #         dataset = create_dataset_traj(data_train, control_params=False,
        #                                       train_target=False,
        #                                       threshold=cfg.model.training.filter_rate,
        #                                       t_range=cfg.model.training.t_range)
        #     else:

        model = DynamicsModel(cfg, env="quadruped")
        train_logs, test_logs = model.train(data_step, cfg)
        log.info("Saving new default models")
        f = hydra.utils.get_original_cwd() + '/models/quadruped/'
        if cfg.exper_dir:
            f = f + cfg.exper_dir + '/'
            if not os.path.exists(f):
                os.mkdir(f)
        copystr = "_%d" % i if cfg.copies else ""
        f = f + cfg.model.str + copystr + '.dat'
        torch.save(model, f)
    # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless

    if cfg.mode == 'eval':
        model_types = ['d']  # ,'tp']
        model_types = ['d','d_state', 'pe', 'pe_state']  # ,'tp']
        model_types = ['d','d_state', 'pe', 'pe_state']  # ,'tp']

        models = {}
        f = hydra.utils.get_original_cwd() + '/models/quadruped/'
        for model_type in model_types:
            models[model_type] = torch.load(f + model_type + ".dat")

        from plot import plot_mse_err, setup_plotting, plot_states
        setup_plotting(models)
        MSEs, predictions, variances = eval_exp(logs, models)

        mse_evald = []
        sh = MSEs[model_types[0]][0].shape
        idx = np.arange(len(data_step))  # remember this line
        for i, id in list(enumerate(idx)):
            gt = data_step[id][0]
            pred = {key: predictions[key][i] for key in predictions}
            var = {key: variances[key][i] for key in variances}
            mse = {key: MSEs[key][i].squeeze() for key in MSEs}

            if False:
                # if
                plot_states(gt, pred, variances=var, idx_plot=[0, 1, 2], save_loc=f"predictions_{i}_", show=False)

            mse_evald.append(mse)

        plot_mse_err(mse_evald, save_loc=("Err Bar MSE of Predictions"),
                     show=True, legend=False)


def eval_exp(test_data, models, verbose=False, env=None, t_range=1000):
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

    # Compile the various trajectories into arrays
    for traj in test_data:
        states.append(traj.states)
        actions.append(traj.actions)
        initials.append(traj.states[0, :])

    states = np.stack(states)
    actions = np.stack(actions)


    initials = np.array(initials)
    N, T, D = states.shape
    if len(np.shape(actions)) == 2:
        actions = np.expand_dims(actions, axis=2)
    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}

    from evaluate import forward_var
    variances = {key: [] for key in models}
    ind_dict = {}
    N, T, D = states.shape
    A = actions.shape[-1]

    for i, key in list(enumerate(models)):
        if verbose and (i + 1) % 10 == 0:
            print("    " + str(i + 1))
        model = models[key]
        indices = model.state_indices
        traj = model.traj
        delta= model.delta
        lstm = "lstm" in key or "rnn" in key  # model.cfg.model.lstm

        ind_dict[key] = np.arange(D) #indices
        print(f"States in use{ind_dict[key]}")
        for i in range(1, T):
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
                        actions_lstm = actions[:, i - 1, :].reshape(1, N, A)
                        states_lstm = currents[key].reshape(1, N, len(models[key].state_indices))
                    else:
                        actions_lstm = actions[:, :i, :].transpose(1, 0, 2)
                        states_lstm = np.concatenate((states_lstm, prediction), axis=0)
                        if True:
                            train_len = model.cfg.model.optimizer.batch
                            if np.shape(actions_lstm)[0] > train_len:
                                actions_lstm = actions_lstm[-train_len:]
                                states_lstm = states_lstm[-train_len:]
                    # input of shape (seq_len, batch, input_size)
                    # output of shape (seq_len, batch, input_size)
                    if env == 'lorenz':
                        raise NotImplementedError("TODO")
                        prediction = model.predict(np.array(currents[key]))
                        prediction = np.array(prediction.detach())
                    else:
                        prediction = model.predict_lstm(np.concatenate((states_lstm, actions_lstm), axis=2),
                                                        num_traj=N)
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
                    prediction = np.array(model.predict(np.hstack(dat)).detach())
                else:
                    if env == 'lorenz':
                        prediction = model.predict(np.array(currents[key]))
                        prediction = np.array(prediction.detach())
                    else:
                        acts = actions[:, i - 1, :]
                        prediction = model.predict(np.hstack((currents[key], acts)))
                        prediction = np.array(prediction.detach())
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

    MSEs = {key: np.square(states[:, :, ind_dict[key]] - predictions[key]).mean(axis=2)[:, 1:] for key in predictions}

    MSEscaled = {}
    for key in predictions:
        # scaling of error
        # if env == 'crazyflie':
        #     # ind_dict[key] = [0,1,3,4,5]
        #     ind_dict[key] = [0, 1, 3, 4]
        if t_range < np.shape(states)[1]:
            l = t_range
        else:
            l = np.shape(states)[1]

        min_states = np.min(states[:, :l, ind_dict[key]], axis=(0, 1))
        max_states = np.ptp(states[:, :l, ind_dict[key]], axis=(0, 1))
        where_zero =  np.where(max_states==0)

        idx_no_zero = np.delete(ind_dict[key], where_zero)

        scaled_states = (states[:, :l, ind_dict[key]] - min_states) / max_states
        scaled_pred = (predictions[key][:, :, :] - min_states) / max_states

        # MSEscaled[key] = np.square(scaled_states[:,:,idx_no_zero] - scaled_pred[:,:,idx_no_zero]).mean(axis=2)[:, 1:]
        # MSEscaled[key] = MSEs[key]
        MSEscaled[key] = np.square(scaled_states - scaled_pred).mean(axis=2)[:, 1:]

    return MSEscaled, predictions, variances


if __name__ == '__main__':
    sys.exit(contpred())
