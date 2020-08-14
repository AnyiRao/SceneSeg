import json
import os
import os.path as osp
import pdb
import pickle
import shutil

import numpy as np

import torch


def mkdir_ifmiss(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_folder_list(checked_directory,log_fn):
    checked_list = os.listdir(checked_directory)
    with open(log_fn, "w") as f:
        for item in checked_list:
            f.write(item+"\n")


def strcal(shotid,num):
    return str(int(shotid)+num).zfill(4)


def read_json(json_fn):
    with open(json_fn,"r") as f:
            json_dict = json.load(f)
    return  json_dict


def write_json(json_fn,json_dict):
    with open(json_fn,"w") as f:
        json.dump(json_dict,f,indent=4)


def read_pkl(pkl_fn):
    with open(pkl_fn,"rb") as f:
        pkl_contents = pickle.load(f)
    return pkl_contents


def write_pkl(pkl_fn,pkl):
    with open(pkl_fn,"wb") as f:
        pickle.dump(pkl,f)


def read_txt_list(txt_fn):
    with open(txt_fn,"r") as f:
            txt_list = f.read().splitlines()
    return txt_list


def write_txt_list(txt_fn,txt_list):
    with open(txt_fn,"w") as f:
        for item in txt_list:
            f.write("{}\n".format(item))


def timecode_to_frames(timecode,framerate):
    return int(int(timecode.split(",")[1])*0.001*framerate) + sum(f * int(t) for f,t in zip((3600*framerate, 60*framerate, framerate), timecode.split(",")[0].split(':')))


def frames_to_timecode(frames,framerate):
    ms = "{0:.3f}".format((frames % framerate)/framerate).split(".")[1]
    return '{0:02d}:{1:02d}:{2:02d},{3:s}'.format(int(frames / (3600*framerate)),
                                            int(frames / (60*framerate) % 60),
                                            int(frames / framerate % 60),
                                            ms)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
