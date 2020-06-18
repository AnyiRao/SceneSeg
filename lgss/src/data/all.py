from __future__ import print_function

import random
import sys
from multiprocessing import Manager, Pool, Process
sys.path.append(".")
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utilis import read_json, read_pkl, read_txt_list, strcal
from utilis.package import *


class Preprocessor(data.Dataset):
    def __init__(self, cfg, listIDs, data_dict):
        self.shot_num = cfg.shot_num
        self.data_root = cfg.data_root
        self.listIDs = listIDs
        self.data_dict = data_dict
        self.shot_boundary_range = range(-cfg.shot_num//2+1,cfg.shot_num//2+1)
        self.mode = cfg.dataset.mode
        assert(len(self.mode) > 0)
    
    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        ID_list = self.listIDs[index]
        if isinstance(ID_list, (tuple, list)):
            place_feats, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            for ID in ID_list:
                place_feat, cast_feat, act_feat, aud_feat, label = self._get_single_item(ID)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)
            if 'cast' in self.mode:
                cast_feats = torch.stack(cast_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            labels = np.array(labels)
            return place_feats, cast_feats, act_feats, aud_feats, labels
        else:
            return self._get_single_item(ID_list)

    def _get_single_item(self, ID):
        imdbid = ID['imdbid']
        shotid = ID['shotid']
        label = self.data_dict["annos_dict"].get(imdbid).get(shotid)
        aud_feats, place_feats = [], []
        cast_feats, act_feats = [], []
        if 'place' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.npy'.format(strcal(shotid, ind))
                path = osp.join(self.data_root, 'place_feat/{}'.format(imdbid), name)
                place_feat = np.load(path)
                place_feats.append(torch.from_numpy(place_feat).float())
        if 'cast' in self.mode:
            for ind in self.shot_boundary_range:
                cast_feat_raw = self.data_dict["casts_dict"].get(imdbid).get(strcal(shotid, ind))
                cast_feat = np.mean(cast_feat_raw, axis=0)
                cast_feats.append(torch.from_numpy(cast_feat).float())
        if 'act' in self.mode:
            for ind in self.shot_boundary_range:
                act_feat = self.data_dict["acts_dict"].get(imdbid).get(strcal(shotid, ind))
                act_feats.append(torch.from_numpy(act_feat).float())
        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.npy'.format(strcal(shotid,ind))
                path = osp.join(
                    self.data_root, 'aud_feat/{}'.format(imdbid), name)
                aud_feat = np.load(path)
                aud_feats.append(torch.from_numpy(aud_feat).float())

        if len(place_feats) > 0:
            place_feats = torch.stack(place_feats)
        if len(cast_feats) > 0:
            cast_feats = torch.stack(cast_feats)
        if len(act_feats) > 0:
            act_feats = torch.stack(act_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, cast_feats, act_feats, aud_feats, label


def data_partition(cfg, imdbidlist_json, annos_dict):
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2

    idxs = []
    for mode in ['train', 'test', 'val']:
        one_mode_idxs = []
        for imdbid in imdbidlist_json[mode]:
            anno_dict = annos_dict[imdbid]
            shotid_list = sorted(anno_dict.keys())
            shotid_tmp = 0
            for shotid in shotid_list:
                if int(shotid) < shotid_tmp+seq_len_half:
                    continue
                shotid_tmp = int(shotid)+seq_len_half
                one_idxs = []
                for idx in range(-seq_len_half+1, seq_len_half+1):
                    one_idxs.append({'imdbid':imdbid, 'shotid': strcal(shotid, idx)})
                one_mode_idxs.append(one_idxs)
        idxs.append(one_mode_idxs)

    partition = {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[1]
    partition['val'] = idxs[2]
    return partition


def data_pre_one(cfg, imdbid, acts_dict, casts_dict, annos_dict, annos_valid_dict):
    data_root = cfg.data_root
    label_fn = osp.join(data_root,'label318')
    place_feat_fn = osp.join(data_root, 'place_feat')
    win_len = cfg.seq_len+cfg.shot_num # - 1

    files = os.listdir(osp.join(place_feat_fn, imdbid))
    all_shot_place_feat = [int(x.split('.')[0].split('_')[1]) for x in files]

    anno_fn = '{}/{}.txt'.format(label_fn, imdbid)
    anno_dict = get_anno_dict(anno_fn)
    annos_dict.update({imdbid: anno_dict})
    # get anno_valid_dict
    anno_valid_dict = anno_dict.copy()
    shotids = [int(x) for x in anno_valid_dict.keys()]
    to_be_del = []
    for shotid in shotids:
        del_flag = False
        for idx in range(-(win_len)//2+1, win_len//2+1):
            if ((shotid + idx) not in all_shot_place_feat) or \
                 anno_dict.get(str(shotid+idx).zfill(4)) is None:
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)
    for shotid in to_be_del:
        del anno_valid_dict[str(shotid).zfill(4)]
    annos_valid_dict.update({imdbid: anno_valid_dict})
    #########################
    act_feat_fn = osp.join(data_root, "act_feat/{}.pkl".format(imdbid))
    acts_dict.update({imdbid: read_pkl(act_feat_fn)})

    cast_feat_fn = osp.join(data_root, "cast_feat/{}.pkl".format(imdbid))
    casts_dict.update({imdbid: read_pkl(cast_feat_fn)})


def data_pre(cfg):
    data_root = cfg.data_root
    imdbidlist_json = osp.join(data_root, 'meta/split318.json')
    imdbidlist_json = read_json(imdbidlist_json)
    imdbidlist = imdbidlist_json['all']

    mgr = Manager()
    acts_dict_raw = mgr.dict()
    casts_dict_raw = mgr.dict()
    annos_dict_raw = mgr.dict()
    annos_valid_dict_raw = mgr.dict()
    jobs = [Process(
        target=data_pre_one,
        args=(cfg, imdbid, acts_dict_raw, casts_dict_raw, annos_dict_raw, annos_valid_dict_raw))
        for imdbid in imdbidlist]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    annos_dict, annos_valid_dict = {}, {}
    acts_dict, casts_dict = {}, {}
    for key, value in annos_dict_raw.items():
        annos_dict.update({key: value})
    for key, value in annos_valid_dict_raw.items():
        annos_valid_dict.update({key: value})
    for key, value in acts_dict_raw.items():
        acts_dict.update({key: value})
    for key, value in casts_dict_raw.items():
        casts_dict.update({key: value})

    return imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict


def get_anno_dict(anno_fn):
    contents = read_txt_list(anno_fn)
    anno_dict = {}
    for content in contents:
        shotid = content.split(' ')[0]
        value = int(content.split(' ')[1])
        if value >= 0:
            anno_dict.update({shotid: value})
        elif value == -1:
            anno_dict.update({shotid: 1})
    return anno_dict


def main():
    from mmcv import Config
    cfg = Config.fromfile("./config/test/all.py")
    imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict = data_pre(cfg)
    partition = data_partition(cfg, imdbidlist_json, annos_valid_dict)
    data_dict = {"annos_dict": annos_dict,
                 "casts_dict": casts_dict,
                 "acts_dict": acts_dict}
    if 0:
        place_feats, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
        listIDs = partition['train']
        for index, _ in enumerate(listIDs):
            ID_list = listIDs[index]
            for ID in ID_list:
                if _get_single_item(ID, cfg, data_dict) is None:
                    continue
                place_feat, cast_feat, act_feat, aud_feat, label = _get_single_item(ID, cfg, data_dict)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
        pdb.set_trace()

    batch_size = cfg.batch_size
    testSet = Preprocessor(cfg, partition['test'], data_dict)
    test_loader = DataLoader(testSet, batch_size=batch_size, \
                shuffle=False, **cfg.data_loader_kwargs)

    dataloader = test_loader
    for batch_idx, (data_place,data_cast,data_act,data_aud,target) in enumerate(dataloader):
        data_place = data_place.cuda()if 'place'in cfg.dataset.mode else []
        data_cast  = data_cast.cuda() if 'cast' in cfg.dataset.mode else []
        data_act   = data_act.cuda()  if 'act' in cfg.dataset.mode else [] 
        data_aud   = data_aud.cuda()  if 'aud' in cfg.dataset.mode else []
        # print (data_cast.shape)
        # print (data_cast.shape,data_act.shape)
        print (data_place.shape,data_cast.shape,data_act.shape,data_aud.shape,target.shape)
        print (batch_idx,target.shape)
        # if i_batch > 1:
        #     break
        # pdb.set_trace()
    pdb.set_trace()


if __name__ == '__main__':
    main()
