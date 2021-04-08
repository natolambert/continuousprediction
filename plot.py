import os
import sys
import glob

import hydra
import numpy as np
import plotly.graph_objects as go
import plotly

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from dotmap import DotMap

import logging

log = logging.getLogger(__name__)

setup = False
label_dict, color_dict, color_dict_plotly, marker_dict, marker_dict_plotly = None, None, None, None, None


###########
# Helpers #
###########


def setup_plotting(models):
    """
    Populates the necessary dictionaries for plotting. Must run this before using most
    plotting functions (those that require the dictionary above.

    Parameters:
        models: a dictionary of models of the form {key (string): model (DynamicsModel)}
    """

    global label_dict, color_dict, color_dict_plotly, marker_dict, marker_dict_plotly, setup

    setup = True

    keyset = {(key[0] if type(key) == tuple else key) for key in models}

    # temp for one-step plot
    # models['pe'].cfg =  models['t'].cfg

    label_dict = {(key[0] if type(key) == tuple else key): models[key].cfg.model.plotting.label for key in models}
    color_dict = {(key[0] if type(key) == tuple else key): models[key].cfg.model.plotting.color for key in models}
    color_dict_plotly = {(key[0] if type(key) == tuple else key): models[key].cfg.model.plotting.color_plotly for key in
                         models}
    marker_dict = {(key[0] if type(key) == tuple else key): models[key].cfg.model.plotting.marker for key in models}
    marker_dict_plotly = {(key[0] if type(key) == tuple else key): models[key].cfg.model.plotting.marker_plotly for key
                          in models}

    # label_dict['zero'] = 'Only Zeros'
    # color_dict['zero'] = '#db0b3f'
    # color_dict_plotly['zero'] = 'rgb(219,11,63)'
    # marker_dict['zero'] = 'o'
    # marker_dict_plotly['zero'] = 'cross'


def find_latest_checkpoint(cfg):
    '''
    Try to find the latest checkpoint in the log directory if cfg.checkpoint
    is not provided (usually through the command line).
    '''
    # same path as in save_log method, but with {} replaced to wildcard *
    checkpoint_paths = os.path.join(os.getcwd(),
                                    cfg.checkpoint_file.replace("{}", "*"))

    # use glob to find files (returned a list)
    files = glob.glob(checkpoint_paths)

    # If we cannot find one (empty file list), then do nothing and return
    if not files:
        return None

    # find the one with maximum last modified time (getmtime). Don't sort
    last_modified_file = max(files, key=os.path.getmtime)

    return last_modified_file


def get_helper(dict1, dict2, key, default):
    dic = dict1 or dict2
    out = dic.get(key)
    return out or default


############
# Plotters #
############
def plot_ss(states, actions, save=False):
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                        # subplot_titles=("Position", "Action - Torques"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ar[:, 0], name='state0',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[::25], y=ar[:, 0][::25], name='state0', mode='markers',
                             marker=dict(color='firebrick', size=50, symbol="circle-open-dot")), row=1, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=ar[:,1], name='state1',
    #                          line=dict(color='royalblue', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=ar[:, 1], name='state1',
                             line=dict(color='green', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[::25], y=ar[:, 1][::25], name='state2', mode='markers',
                             marker=dict(color='green', size=50, symbol="hourglass-open")), row=1, col=1)

    fig.add_trace(go.Scatter(x=xs, y=ar[:, 2], name='state2',
                             line=dict(color='black', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[::25], y=ar[:, 2][::25], name='state2', mode='markers',
                             marker=dict(color='black', size=50, symbol="x-open-dot")), row=1, col=1)

    # fig.add_trace(go.Scatter(x=xs, y=ar[:, 3], name='state3', #
    #                          line=dict(color='black', width=4)), row=1, col=1)

    fig.update_layout(  # title='Position of Cartpole Task',
        xaxis_title='Timestep',
        yaxis_title='Normalize State',
        plot_bgcolor='white',
        showlegend=False,
        font=dict(family='Times New Roman', size=50, color='#000000'),
        height=800,
        width=1500,
        margin=dict(r=0, l=0, b=10, t=1),
        xaxis=dict(
            range=[0, 200],
            showline=False,
            showgrid=False,
            showticklabels=True, ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True, ),
    )
    fig.show()
    if save: fig.write_image("traj.pdf")


def plot_cp(states, actions, save=False):
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                        # subplot_titles=("Position", "Action - Torques"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ar[:, 0], name='state0',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[::25], y=ar[:, 0][::25], name='state0', mode='markers',
                             marker=dict(color='firebrick', size=50, symbol="circle-open-dot")), row=1, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=ar[:,1], name='state1',
    #                          line=dict(color='royalblue', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=ar[:, 2], name='state2',
                             line=dict(color='green', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[::25], y=ar[:, 2][::25], name='state2', mode='markers',
                             marker=dict(color='green', size=50, symbol="hourglass-open")), row=1, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=ar[:, 3], name='state3', #
    #                          line=dict(color='black', width=4)), row=1, col=1)

    fig.update_layout(  # title='Position of Cartpole Task',
        xaxis_title='Timestep',
        yaxis_title='Normalize State',
        plot_bgcolor='white',
        showlegend=False,
        font=dict(family='Times New Roman', size=50, color='#000000'),
        height=800,
        width=1500,
        margin=dict(r=0, l=0, b=10, t=1),
        xaxis=dict(
            range=[0, 200],
            showline=False,
            showgrid=False,
            showticklabels=True, ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True, ),
    )
    fig.show()
    if save: fig.write_image("traj.pdf")


def plot_cf(states, actions):
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    r = np.rad2deg(ar[:, 3])
    p = np.rad2deg(ar[:, 4])
    y = ar[:, 5]

    # actions = np.stack(actions)

    fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                        subplot_titles=("Position", "Action - Torques"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=r, name='roll',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=p, name='pitch',
                             line=dict(color='royalblue', width=4)), row=1, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=y, name='yaw',
    #                          line=dict(color='green', width=4)), row=1, col=1)

    # fig.add_trace(go.Scatter(x=xs, y=actions[:, 0], name='M1',
    #                          line=dict(color='firebrick', width=4)), row=2, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=actions[:, 1], name='M2',
    #                          line=dict(color='royalblue', width=4)), row=2, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=actions[:, 2], name='M3',
    #                          line=dict(color='green', width=4)), row=2, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=actions[:, 3], name='M4',
    #                          line=dict(color='orange', width=4)), row=2, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=actions[:, 4], name='M5',
    #                          line=dict(color='black', width=4)), row=2, col=1)

    fig.update_layout(title='Position of Crazyflie Task',
                      xaxis_title='Timestep',
                      yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      )
    fig.show()


def plot_reacher(states, actions):
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    X = ar[:, -3]
    Y = ar[:, -2]
    Z = ar[:, -1]

    actions = np.stack(actions)

    fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                        subplot_titles=("Position", "Action - Torques"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=X, name='X',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=Y, name='Y',
                             line=dict(color='royalblue', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=Z, name='Z',
                             line=dict(color='green', width=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=xs, y=actions[:, 0], name='M1',
                             line=dict(color='firebrick', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 1], name='M2',
                             line=dict(color='royalblue', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 2], name='M3',
                             line=dict(color='green', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 3], name='M4',
                             line=dict(color='orange', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 4], name='M5',
                             line=dict(color='black', width=4)), row=2, col=1)

    fig.update_layout(title='Position of Reacher Task',
                      xaxis_title='Timestep',
                      yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      )
    fig.show()


def plot_states_dist(data):
    states = []
    for seq in data:
        states.append(seq.states)
    state_data = np.stack(states)

    traces = []
    for i in range(np.shape(state_data)[-1]):
        s = state_data[:, :, i]
        s_mean = np.mean(s, axis=0)
        trc, x, y = generate_errorbar_traces(s)

        layout = dict(  # title=title if title else f"Average Error over Run",
            xaxis={'title': 'Prediction Step'},  # 2e-9, 5
            yaxis={'title': f"State distribution {i}"},  # 'range': [np.log10(20e-6), np.log10(5)]},
            xaxis_showgrid=False, yaxis_showgrid=False,
            font=dict(family='Times New Roman', size=50, color='#000000'),
            height=800,
            width=1500,
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(r=0, l=0, b=10, t=1),

            legend={'x': .01, 'y': .98, 'bgcolor': 'rgba(50, 50, 50, .03)',
                    'font': dict(family='Times New Roman', size=30, color='#000000')}
        )

        fig = {
            'data': trc,
            # 'layout': layout
        }

        import plotly.io as pio
        fig = go.Figure(fig)
        fig.update_layout(layout)
        fig.show()
        fig.write_image(f"state_{i}" + ".pdf")


def generate_errorbar_traces(ys, xs=None, percentiles='66+95', color=None, name=None):
    if xs is None:
        xs = [list(range(len(y))) for y in ys]

    minX = min([len(x) for x in xs])

    xs = [x[:minX] for x in xs]
    ys = [y[:minX] for y in ys]

    assert all([(len(y) == len(ys[0])) for y in ys]), \
        'Y should be the same size for all traces'

    assert all([(x == xs[0]) for x in xs]), \
        'X should be the same for all traces'

    y = np.array(ys)

    def median_percentile(data, des_percentiles='66+95'):
        # median = np.nanmedian(data, axis=0)
        median = np.median(data, axis=0)
        out = np.array(list(map(int, des_percentiles.split("+"))))
        for i in range(out.size):
            assert 0 <= out[i] <= 100, 'Percentile must be >0 <100; instead is %f' % out[i]
        list_percentiles = np.empty((2 * out.size,), dtype=out.dtype)
        list_percentiles[0::2] = out  # Compute the percentile
        list_percentiles[1::2] = 100 - out  # Compute also the mirror percentile
        # percentiles = np.nanpercentile(data, list_percentiles, axis=0)
        percentiles = np.percentile(data, list_percentiles, axis=0)
        return [median, percentiles]

    out = median_percentile(y, des_percentiles=percentiles)
    ymed = out[0]
    # yavg = np.median(y, 0)

    err_traces = [
        dict(x=xs[0], y=ymed.tolist(), mode='lines', name=name, type='scatter', legendgroup=f"group-{name}",
             line=dict(color=color, width=4))]

    intensity = .3  # .15 #
    ''' 
    interval = scipy.stats.norm.interval(percentile/100, loc=y, scale=np.sqrt(variance))
    interval = np.nan_to_num(interval)  # Fix stupid case of norm.interval(0) returning nan
    '''

    for i, p_str in enumerate(percentiles.split("+")):
        p = int(p_str)
        high = out[1][2 * i, :]
        low = out[1][2 * i + 1, :]

        err_traces.append(dict(
            x=xs[0] + xs[0][::-1], type='scatter',
            y=(high).tolist() + (low).tolist()[::-1],
            fill='toself',
            fillcolor=(color[:-1] + str(f", {intensity})")).replace('rgb', 'rgba')
            if color is not None else None,
            line=dict(color='rgba(0,0,0,0.0)'),
            legendgroup=f"group-{name}",
            showlegend=False,
            name=name + str(f"_std{p}") if name is not None else None,
        ), )
        intensity -= .1

    return err_traces, xs, ys


def plot_states(ground_truth, predictions, variances=None, idx_plot=None, plot_avg=True, save_loc=None, show=True):
    """
    Plots the states given in predictions against the groundtruth. Predictions
    is a dictionary mapping model types to predictions
    """
    assert setup, "Must run setup_plotting before this function"

    num = np.shape(ground_truth)[0]
    dx = np.shape(ground_truth)[1]
    import matplotlib

    # font = {'size': 18, 'family': 'serif', 'serif': ['Times']}
    font = {'size': 18, 'family': 'Times New Roman'}  # , 'serif': ['Times']}
    matplotlib.rc('font', **font)

    if idx_plot is None:
        idx_plot = list(range(dx))

    if False:
        fig, ax = plt.subplots()
        plt.title("Predictions Averaged")
        plt.xlabel("Timestep")
        plt.ylabel("Average State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        gt = np.zeros(ground_truth[:, 0:1].shape)
        for i in idx_plot:
            gt = np.hstack((gt, ground_truth[:, i:i + 1]))
        gt_avg = np.average(gt[:, 1:], axis=1)
        plt.plot(gt_avg, c='k', label='Groundtruth')

        for key in predictions:
            pred = predictions[key]
            p = np.zeros(pred[:, 0:1].shape)
            for i in idx_plot:
                p = np.hstack((p, pred[:, i:i + 1]))
            p_avg = np.average(p[:, 1:], axis=1)
            # chopped = [(x if abs(x) < 50 else float("nan")) for x in p_avg]
            chopped = p_avg
            plt.plot(chopped, c=color_dict[key], label=label_dict[key], markersize=10, marker=marker_dict[key],
                     markevery=50)
        # plt.ylim(-.5, 1.5)
        plt.legend()
        if save_loc:
            fig.set_size_inches(7.5, 4)
            plt.savefig(save_loc + "-avg_states.pdf")
        if show:
            plt.show()
        else:
            plt.close()

    for i in idx_plot:
        fig, ax = plt.subplots()
        gt = ground_truth[:, i]
        # plt.title("Predictions on one dimension")
        plt.xlabel("Timestep")
        plt.ylabel("State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot(gt, c='k', label='gt')
        for key in predictions:
            # print(key)
            pred = predictions[key][:, i]

            if key == 'lstm_d':
                color_dict[key] = '#800080'
                marker_dict[key] = "*"

            # chopped = np.maximum(np.minimum(pred, 3), -3)  # to keep it from messing up graphs when it diverges
            chopped = [(x if abs(x) < 150 else np.nan) for x in pred]
            # chopped = pred
            plt.plot(chopped, c=color_dict[key], label=key,  # label_dict[key],
                     markersize=10, marker=marker_dict[key],
                     markevery=50)

            # chopped[0] = chopped[0].item()
            if variances is not None:
                err_every = 20
                start = np.random.randint(10)
                chopped = np.array(chopped)
                v = 2 * np.sqrt(np.array(variances[key][:, i])) ** 2
                # v = np.array([(x if abs(x) < 10 else 10) for x in v])
                v = np.array([(x if abs(x) < 150 else 150) for x in v])
                # v = np.array([(x if abs(x) < 10 else np.nan) for x in v])
                # plt.errorbar(x=np.arange(len(chopped))[start+1::err_every], y=chopped[start+1::err_every],
                #              yerr=v[start::err_every], c=color_dict[key])
                plt.fill_between(np.arange(len(chopped))[1:], chopped[1:] + v, chopped[1:] - v,
                                 facecolor=color_dict[key], alpha=0.5)

        plt.legend(ncol=2, fontsize=16)
        # if i > 9:
        #     plt.ylim(-3,3)
        # else:
        #     plt.ylim(-1.5,1.5)

        if save_loc:
            fig.set_size_inches(7.5, 4)
            plt.savefig(save_loc + "-state%d.pdf" % i, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def plot_loss(train_logs, test_logs, cfg, save_loc=None, show=False, title=None):
    """

    Parameters:
        logs: a list of lists of loss values, one list for each net in the model
        s: the string describing the model, ie 'd' or 'tpe'
    """
    assert setup, "Must run setup_plotting before this function"

    fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                        # subplot_titles=("Position", "Action - Torques"),
                                        vertical_spacing=.15)  # go.Figure()
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]

    markers = [
        "cross-open-dot",
        "circle-open-dot",
        "x-open-dot",
        "triangle-up-open-dot",
        "y-down-open",
        "diamond-open-dot",
        "hourglass-open",
        "hash-open-dot",
        "star-open-dot",
        "square-open-dot",
    ]

    def add_line(fig, log, type, ind=-1):
        if ind == -1:
            name = type
        else:
            name = type + str(ind)
        if type == 'Test':
            fig.add_trace(go.Scatter(x=np.arange(len(log)).tolist(), y=log,
                                     name=name, legendgroup=type,
                                     line=dict(color=colors[ind], width=4),
                                     marker=dict(color=colors[ind], symbol=markers[ind], size=16)),
                          row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=np.arange(len(log)).tolist(), y=log,
                                     name=name, legendgroup=type,
                                     line=dict(color=colors[ind], width=4, dash='dash'),
                                     marker=dict(color=colors[ind], symbol=markers[ind], size=16)),
                          row=1, col=1)
        return fig

    if len(np.shape(train_logs)) > 1:
        # ENSEMBLE
        for i, (train, test) in enumerate(zip(train_logs, test_logs)):
            fig = add_line(fig, train, type="Train", ind=i)
            fig = add_line(fig, test, type="Test", ind=i)
    else:
        # BASE
        fig = add_line(fig, train_logs, type="Train", ind=-1)
        fig = add_line(fig, test_logs, type="Test", ind=-1)

    fig.update_layout(font=dict(
        family="Times New Roman, Times, serif",
        size=24,
        color="black"
    ),
        title='Training Plot ' + cfg.model.str,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        plot_bgcolor='white',
        width=1000,
        height=1000,
        margin=dict(l=10, r=0, b=10),
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True, ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True, ),
    )
    if show: fig.show()
    fig.write_image(save_loc + ".png")


def add_marker(err_traces, color=[], symbol=None, skip=None):
    mark_every = 50
    size = 50
    l = len(err_traces[0]['x'])
    skip = np.random.randint(mark_every - 10) + 15
    # skip = np.random.randint(mark_every)
    if skip is not None:
        size_list = [0] * skip + [size] + [0] * (mark_every - skip)
    else:
        size_list = [size] + [0] * (mark_every)
    repeat = int(l / mark_every)
    size_list = size_list * repeat
    line = err_traces[0]
    line['mode'] = 'lines+markers'
    line['marker'] = dict(
        color=line['line']['color'],
        size=size_list,
        symbol="x" if symbol is None else symbol,
        line=dict(width=3,
                  color='rgba(1,1,1,1)')
    )
    err_traces[0] = line
    return err_traces


# def plot_mse_err(mse_batch, save_loc=None, show=True, log_scale=True, title=None, y_min=.05, y_max=1e4, legend=False):
def plot_mse_err(mse_batch, save_loc=None, show=True, log_scale=True, title=None, y_min=.01, y_max=1e7,
                     legend=False):

    assert setup, "Must run setup_plotting before this function"

    arrays = []
    keys = [k for k in mse_batch[0].keys()]
    for k in keys:
        temp = []
        for data in mse_batch:
            temp.append(data[k])
        arrays.append(np.stack(temp))


    colors_temp = ['rgb(12, 7, 134)', 'rgb(64, 3, 156)', 'rgb(106, 0, 167)',
                   'rgb(143, 13, 163)', 'rgb(176, 42, 143)', 'rgb(203, 71, 119)', 'rgb(224, 100, 97)',
                   'rgb(242, 132, 75)', 'rgb(252, 166, 53)', 'rgb(252, 206, 37)']
    traces_plot = []
    for n, (ar, k) in enumerate(zip(arrays, keys)):
        # temp
        # if n > 1:
        #     continue
        tr, xs, ys = generate_errorbar_traces(ar, xs=[np.arange(1, np.shape(ar)[1] + 1).tolist()], percentiles='66+90',
                                              color=color_dict_plotly[k],
                                              name=label_dict[k]+str(n))
        w_marker = []
        # for t in tr:
        m = add_marker(tr, color=color_dict_plotly[k], symbol=marker_dict_plotly[k], skip=30)
        # w_marker.append(m)
        [traces_plot.append(t) for t in m]

    layout = dict(  # title=title if title else f"Average Error over Run",
        xaxis={'title': 'Prediction Step'},  # 2e-9, 5
        yaxis={'title': 'Mean Squared Error', 'range': [np.log10(20e-6), np.log10(10)]},# 25]}, #
        # yaxis={'title': 'Mean Squared Error', 'range': [np.log10(.01), np.log10(10000)]},# 25]}, #
        # yaxis={'title': 'Mean Squared Error', 'range': [np.log10(0.1), np.log10(10000000000000)]},# 25]}, #
        # [np.log10(y_min), np.log10(y_max)]},
        yaxis_type="log",
        xaxis_showgrid=False, yaxis_showgrid=False,
        font=dict(family='Times New Roman', size=50, color='#000000'),
        height=800,
        width=1500,
        plot_bgcolor='white',
        showlegend=legend,
        margin=dict(r=0, l=0, b=10, t=1),

        legend={'x': .01, 'y': .98, 'bgcolor': 'rgba(50, 50, 50, .03)',
                'font': dict(family='Times New Roman', size=30, color='#000000')}
    )

    fig = {
        'data': traces_plot,
        # 'layout': layout
    }

    import plotly.io as pio
    fig = go.Figure(fig)
    fig.update_layout(layout)
    if show: fig.show()
    fig.write_image(save_loc + ".pdf")

    return fig


def plot_mse(MSEs, log_scale=True, title=None, custom_colors=None, custom_labels=None,
             custom_markers=None, save_loc=None, show=True):
    """
    Plots MSE graphs for the sequences given given

    Parameters:
    ------------
    MSEs: a dictionary mapping model type key to an array of MSEs
    """
    assert setup, "Must run setup_plotting before this function"

    fig, ax = plt.subplots()
    title = title or "%s MSE for a variety of models" % ('Log ' if log_scale else '')
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel('Mean Square Error')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for key in MSEs:
        mse = MSEs[key]
        color = get_helper(custom_colors, color_dict, key, None)
        label = get_helper(custom_labels, label_dict, key, None)
        marker = get_helper(custom_markers, marker_dict, key, 'o')
        if log_scale:
            plt.semilogy(mse, color=color, label=label, marker=marker, markevery=50)
        else:
            plt.plot(mse, color=color, label=label, marker=marker, markevery=50)
    plt.legend()
    if save_loc:
        plt.savefig(save_loc)
    if show:
        plt.show()
    else:
        plt.close()


def plot_lorenz(data, cfg, predictions=None):
    assert setup, "Must run setup_plotting before this function"

    import plotly.graph_objects as go

    fig = go.Figure()

    for dat in data:
        x, y, z = dat.states[:, 0], dat.states[:, 1], dat.states[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            # color=(1, c[i], 0),
            marker=dict(
                size=2,
                color=np.arange(len(x)),
                colorscale='Viridis',
            ),
            line=dict(
                color='black',
                width=4
            ),
        ))

    color_scales_dict = {'t': 'Inferno',
                         'd': 'Magma',
                         'p': 'Plasma',
                         'tp': 'Blackbody',
                         'te': 'Electric',
                         'de': 'Hot',
                         'pe': 'Jet',
                         'tpe': 'Plotly3'}

    if predictions is not None:
        for key, p in predictions.items():
            if len(np.shape(p)) == 3:
                fig.add_trace(go.Scatter3d(x=p[0, :, 0], y=p[0, :, 1], z=p[0, :, 2],
                                           name=label_dict[key], legendgroup=key,
                                           marker=dict(
                                               size=1,
                                               color=np.arange(len(x)),
                                               colorscale=color_scales_dict[key],
                                           ),
                                           line=dict(
                                               color=color_dict_plotly[key],
                                               width=1
                                           ),
                                           ))
            else:
                fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                           name=label_dict[key], legendgroup=key,
                                           marker=dict(
                                               size=1,
                                               color=np.arange(len(x)),
                                               colorscale=color_scales_dict[key],
                                           ),
                                           line=dict(
                                               color=color_dict_plotly[key],
                                               width=1
                                           ),
                                           ))

    fig.update_layout(
        width=1500,
        height=800,
        autosize=False,
        showlegend=True if predictions is not None else False,
        font=dict(
            family="Times New Roman, Times, serif",
            size=18,
            color="black"
        ),
        scene_camera=dict(eye=dict(x=1.5 * -.1, y=1.5 * 1.5, z=1.5 * .25)),
        scene=dict(
            # xaxis=dict(nticks=4, range=[-100, 100], ),
            # yaxis=dict(nticks=4, range=[-100, 100], ),
            # zaxis=dict(nticks=4, range=[-100, 100], ),
            xaxis=dict(nticks=5, range=[-40, 40],
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgb(128, 128, 128)",
                       showbackground=True,
                       zerolinecolor="rgb(0, 0, 0)",
                       ),
            yaxis=dict(nticks=5, range=[-60, 60],
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgb(128, 128, 128)",
                       showbackground=True,
                       zerolinecolor="rgb(0, 0, 0)",
                       ),
            zaxis=dict(nticks=5, range=[-40, 75],
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgb(128, 128, 128)",
                       showbackground=True,
                       zerolinecolor="rgb(0, 0, 0)",
                       ),
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
            aspectmode='manual'
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.show()
    fig.write_image(os.getcwd() + "/lorenz.png")


def plot_sorted(deltas_gt, deltas_pred, idx_plot=None, save_loc=None, show=True):
    assert setup, "Must run setup_plotting before this function"

    # deltas_gt = {key: np.reshape(deltas_gt[key], (-1, deltas_gt[key].shape[-1])) for key in deltas_gt}
    # deltas_pred = {key: np.reshape(deltas_pred[key], (-1, deltas_pred[key].shape[-1])) for key in deltas_pred}

    if idx_plot is None:
        idx_plot = list(range(5))

    for key in deltas_gt:
        for idx in idx_plot:
            # Sorting
            gt = deltas_gt[key][:, :, idx].ravel()
            pred = deltas_pred[key][:, :, idx].ravel()
            zipped = list(zip(gt, pred))
            zipped.sort()
            gt_plot = []
            pred_plot = []
            for gt_i, pred_i in zipped:
                gt_plot.append(gt_i)
                pred_plot.append(pred_i)

            # Plotting
            fig, ax = plt.subplots()
            plt.title("Sorted Predictions - %s - Dimension %d" % (label_dict[key], idx))
            plt.ylabel("Delta Predictions")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.plot(gt_plot, c='k', label='Groundtruth')
            # for key in delts:
            #     # plt.plot(preds[key], label=label_dict[key], c=color_dict[key])
            #     plt.scatter(np.arange(len(delts[key])), delts[key], c=color_dict[key],
            #                 label=label_dict[key], marker=marker_dict[key], s=3)
            plt.scatter(np.arange(len(pred_plot)), pred_plot, c=color_dict[key], marker=marker_dict[key], s=3)

            plt.ylim(min(np.min(gt) * .8, np.min(gt) * 1.2), max(np.max(gt) * .8, np.max(gt) * 1.2))
            plt.legend()

            if save_loc:
                plt.savefig(save_loc + "-state%d-%s.pdf" % (idx, key))
            if show:
                plt.show()
            else:
                plt.close()

    # Debug: plot unsorted
    for key in deltas_gt:
        for idx in idx_plot:
            gt = deltas_gt[key][:, :, idx].ravel()
            pred = deltas_pred[key][:, :, idx].ravel()

            fig, ax = plt.subplots()
            plt.title("Unsorted Predictions - Dimension %d" % idx)
            plt.ylabel("Delta Predictions")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.plot(gt, c='k', label='Groundtruth')
            plt.plot(pred, label=label_dict[key], c=color_dict[key])

            plt.ylim(min(np.min(gt) * .8, np.min(gt) * 1.2), max(np.max(gt) * .8, np.max(gt) * 1.2))
            plt.legend()

            if save_loc:
                plt.savefig(save_loc + "-unsorted-state%d.pdf" % idx)
            if show:
                plt.show()
            else:
                plt.close()


def plot_evaluations(data, x, ylabel=None, xlabel=None, title=None, log_scale=False, save_loc=None, show=True):
    """
    Plots plots for sample efficiency tests

    data: dictionary of arrays of eval values
    """
    assert setup, "Must run setup_plotting before this function"

    fig, ax = plt.subplots()
    plt.title(title or "Trajectory prediction evalutaions")
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if log_scale:
        plt.yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for key in data:
        plt.plot(x, data[key].squeeze(), color=color_dict[key], label=label_dict[key], marker=marker_dict[key])
    plt.legend()
    if save_loc:
        plt.savefig(save_loc)
    if show:
        plt.show()
    else:
        plt.close()


def plot_evaluations_3d(data, x, y, ylabel=None, xlabel=None, zlabel=None, title=None, log_scale=False, save_loc=None,
                        show=True):
    """
    Plots the data using a heatmap, which is a nice way of doing 3D plots

    Parameters:
        Data: a dictionary of 2D arrays of shale (len(y), len(x))
        x: the labels to go on the x axis
        y: the labels to go on the y axis
    """
    assert setup, "Must run setup_plotting before this function"

    # X = np.tile(x, len(y)).reshape(len(y), -1).T
    # Y = np.tile(y, len(x)).reshape(len(x), -1)

    # if log_scale:
    #     data = {key: np.log(data[key]) for key in data}

    cmap = 'magma'

    images = []
    dats = []
    fig, axs = plt.subplots(1, len(data), figsize=(10, 5))
    for i, key in list(enumerate(data)):
        axs[i].set_title(label_dict[key])

        dat = data[key]
        dat = np.nan_to_num(dat)
        im = axs[i].imshow(data[key], cmap=cmap, origin='lower')

        ind_x = np.arange(1, len(x), 2) if len(x) > 10 else np.arange(len(x))
        ind_y = np.arange(1, len(y), 2) if len(y) > 10 else np.arange(len(y))
        ticks_x = np.array(x)[ind_x]
        ticks_y = np.array(y)[ind_y]

        axs[i].set_xticks(ind_x)
        axs[i].set_yticks(ind_y)
        axs[i].set_xticklabels(ticks_x)
        axs[i].set_yticklabels(ticks_y)

        images.append(im)
        dats.append(dat)

    cmin = np.min(np.array(dats))
    cmax = np.max(np.array(dats))
    if log_scale:
        norm = colors.LogNorm(vmin=cmin, vmax=cmax)
    else:
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.1)
    if zlabel:
        cbar.ax.set_ylabel(zlabel)

    if xlabel:
        fig.text(0.5, 0.04, xlabel, ha='center')
    if ylabel:
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

    if save_loc:
        plt.savefig("%s.pdf" % save_loc)
    if show:
        plt.show()
    else:
        plt.close()
    # for key in data:
    #     fig, ax = plt.subplots()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     plt.title(label_dict[key])
    #
    #     dat = data[key]
    #     dat = np.nan_to_num(dat)
    #     im = plt.imshow(dat, origin='lower')
    #
    #     if ylabel:
    #         # yLabel = ax.set_ylabel(ylabel)
    #         plt.ylabel(ylabel)
    #     if xlabel:
    #         # xLabel = ax.set_xlabel(xlabel)
    #         plt.xlabel(xlabel)
    #     cbar = ax.figure.colorbar(im)
    #     if zlabel:
    #         cbar.ax.set_ylabel(zlabel)
    #     # if zlabel:
    #     #     zLabel = ax.set_zlabel(zlabel)
    #     # ax.view_init(elev=40, azim=-130)
    #
    #     ax.set_xticks(np.arange(len(x)))
    #     ax.set_yticks(np.arange(len(y)))
    #     ax.set_xticklabels(x)
    #     ax.set_yticklabels(y)
    #
    #     if save_loc:
    #         plt.savefig("%s_%s.pdf" % (save_loc, key))
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()


@hydra.main(config_path='config-plot.yaml')
def plot(cfg):
    pass


if __name__ == '__main__':
    sys.exit(plot())
