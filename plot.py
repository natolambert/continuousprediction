import os
import sys
import glob

import hydra


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
