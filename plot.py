import os
import sys
import glob

import hydra
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *

import logging

log = logging.getLogger(__name__)


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
        median = np.nanmedian(data, axis=0)
        out = np.array(list(map(int, des_percentiles.split("+"))))
        for i in range(out.size):
            assert 0 <= out[i] <= 100, 'Percentile must be >0 <100; instead is %f' % out[i]
        list_percentiles = np.empty((2 * out.size,), dtype=out.dtype)
        list_percentiles[0::2] = out  # Compute the percentile
        list_percentiles[1::2] = 100 - out  # Compute also the mirror percentile
        percentiles = np.nanpercentile(data, list_percentiles, axis=0)
        return [median, percentiles]

    out = median_percentile(y, des_percentiles=percentiles)
    ymed = out[0]
    # yavg = np.median(y, 0)

    err_traces = [
        dict(x=xs[0], y=ymed.tolist(), mode='lines', name=name, type='line', legendgroup=f"group-{name}",
             line=dict(color=color, width=4))]

    intensity = .3
    '''
    interval = scipy.stats.norm.interval(percentile/100, loc=y, scale=np.sqrt(variance))
    interval = np.nan_to_num(interval)  # Fix stupid case of norm.interval(0) returning nan
    '''

    for i, p_str in enumerate(percentiles.split("+")):
        p = int(p_str)
        high = out[1][2 * i, :]
        low = out[1][2 * i + 1, :]

        err_traces.append(dict(
            x=xs[0] + xs[0][::-1], type='line',
            y=(high).tolist() + (low).tolist()[::-1],
            fill='toself',
            fillcolor=(color[:-1] + str(f", {intensity})")).replace('rgb', 'rgba')
            if color is not None else None,
            line=dict(color='transparent'),
            legendgroup=f"group-{name}",
            showlegend=False,
            name=name + str(f"_std{p}") if name is not None else None,
        ), )
        intensity -= .1

    return err_traces, xs, ys


@hydra.main(config_path='config-plot.yaml')
def plot(cfg):
    data = Data([new_trace1])

    plot_url = py.plot(data, filename='append plot', fileopt='append')


if __name__ == '__main__':
    sys.exit(plot())
