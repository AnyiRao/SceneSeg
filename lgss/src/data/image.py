from __future__ import print_function

import random
import sys
sys.path.append(".")
from multiprocessing import Manager, Pool, Process

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


normalizer = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalizer])


class Preprocessor(data.Dataset):
    def __init__(self, cfg, listIDs, data_dict):
        self.shot_num = cfg.shot_num
        self.data_root = cfg.data_root
        self.listIDs = listIDs
        self.data_dict = data_dict
        self.shot_boundary_range = range(-cfg.shot_num//2+1, cfg.shot_num//2+1)
        self.mode = cfg.dataset.mode
        self.transform = transformer
        assert(len(self.mode) > 0)

    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        ID_list = self.listIDs[index]
        if isinstance(ID_list, (tuple, list)):
            imgs, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            for ID in ID_list:
                img, label = self._get_single_item(ID)
                imgs.append(img)
                labels.append(label)
            if 'image' in self.mode:
                imgs = torch.stack(imgs)
            labels = np.array(labels)
            return imgs, cast_feats, act_feats, aud_feats, labels
        else:
            return self._get_single_item(ID_list)

    def _get_single_item(self, ID):
        imdbid = ID['imdbid']
        shotid = ID['shotid']
        label = self.data_dict["annos_dict"].get(imdbid).get(shotid)
        imgs = []
        if 'image' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.jpg'.format(strcal(shotid, ind))
                path = osp.join(
                    self.data_root, 'keyf_240p/{}'.format(imdbid), name)
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        return imgs, label


def data_partition(cfg, imdbidlist_json, annos_dict):
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2

    idxs = []
    for mode in ['train','test','val']:
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
                    one_idxs.append({'imdbid': imdbid, 'shotid': strcal(shotid, idx)})
                one_mode_idxs.append(one_idxs)
        idxs.append(one_mode_idxs)

    partition= {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[1]
    partition['val'] = idxs[2]
    return partition


def data_pre_one(cfg, imdbid, acts_dict, casts_dict, annos_dict, annos_valid_dict):
    data_root = cfg.data_root
    label_fn = osp.join(data_root, 'label318')
    img_dir_fn = osp.join(data_root, 'keyf_240p')
    win_len = cfg.seq_len + cfg.shot_num # - 1

    files = os.listdir(osp.join(img_dir_fn, imdbid))
    all_shot_img = [int(x.split('.')[0].split('_')[1]) for x in files]

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
            if ((shotid + idx) not in all_shot_img) or \
                 anno_dict.get(str(shotid+idx).zfill(4)) is None:
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)
    for shotid in to_be_del:
        del anno_valid_dict[str(shotid).zfill(4)]
    annos_valid_dict.update({imdbid: anno_valid_dict})


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

    return imdbidlist_json, annos_dict, annos_valid_dict


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
    cfg = Config.fromfile("./config/image.py")

    imdbidlist_json, annos_dict, annos_valid_dict = data_pre(cfg)
    partition = data_partition(cfg, imdbidlist_json, annos_valid_dict)
    data_dict = {"annos_dict": annos_dict}

    batch_size = cfg.batch_size
    testSet = Preprocessor(cfg, partition['test'], data_dict)
    test_loader = DataLoader(
                testSet, batch_size=batch_size,
                shuffle=False, **cfg.data_loader_kwargs)

    dataloader = test_loader
    for batch_idx, (data_place, data_cast, data_act, data_aud, target) in enumerate(dataloader):
        print(data_place.shape)  # bs, seq_len, shot_num, 3, 224, 224
        print(batch_idx,target.shape)
        # if batch_idx > 1:
        #     break
        # pdb.set_trace()
    pdb.set_trace()


if __name__ == '__main__':
    main()
