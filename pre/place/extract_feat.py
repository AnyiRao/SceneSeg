from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import multiprocessing
import numpy as np
import pickle
import pdb
from PIL import Image
import sys
import json
import time
from pprint import pprint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

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

class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.base = models.resnet50(pretrained=pretrained)
    
    def forward(self, x):
        for name, module in self.base._modules.items():
            x = module(x)
            # print(name, x.size())
            if name == 'avgpool':
                x = x.view(x.size(0), -1)
                feature = x.clone()
        return feature, x

class Extractor(object):
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model
        # pprint(self.model.module)

    def extract_feature(self, data_loader, print_summary=True):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        scores = OrderedDict()

        end = time.time()
        for i, (imgs, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = self.model(imgs)

            for fname, feat, score in zip(fnames, outputs[0], outputs[1]):
                features[fname] = feat.cpu().data
                scores[fname] = score.cpu().data

            batch_time.update(time.time() - end)
            end = time.time()

            if print_summary:
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(
                        i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg))
        return features, scores


class Preprocessor(object):
    def __init__(self, dataset, images_path, default_size, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = osp.join(self.images_path, fname)
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, fname


def get_data(video_id, img_path, batch_size, workers):

    dataset = os.listdir(img_path)
    if len(dataset) % batch_size < 8:
        for i in range(8 - len(dataset) % batch_size):
            dataset.append(dataset[-1])

    # data transforms
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    data_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    data_loader = DataLoader(
        Preprocessor(dataset, img_path, default_size=(256, 256), transform=data_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, data_loader


def get_img_folder(data_root, video_id):
    img_folder = osp.join(data_root, video_id)
    if osp.isdir(img_folder):
        return img_folder
    else:
        print('No such movie: {}'.format(video_id))
        return None


def main(args):
    print(args)
    cudnn.benchmark = True
    # create model
    model = ResNet50(pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    # create and extractor
    extractor = Extractor(model)
    
    if args.list_file is None:
        video_list = sorted(os.listdir(args.source_img_path))
    else:
        video_list = [x for x in open(args.list_file)] 
    video_list = [i.split(".m")[0] for i in video_list] ## to remove suffix .mp4 .mov etc. if applicable
    video_list = video_list[args.st:args.ed]
    print('****** Total {} videos ******'.format(len(video_list)))

    for idx_m, video_id in enumerate(video_list):
        print('****** {}, {} / {}, {} ******'.format(datetime.now(), idx_m+1, len(video_list), video_id))
        save_path = osp.join(args.save_path, video_id)
        os.makedirs(save_path,exist_ok=True)
        img_path = get_img_folder(args.source_img_path, video_id)
        if not osp.isdir(img_path):
            print('Cannot find images!')

        feat_save_name = osp.join(save_path, 'feat.pkl')
        score_save_name = osp.join(save_path, 'score.pkl')
        if osp.isfile(feat_save_name) and osp.isfile(score_save_name):
            print('{}, {} exist.'.format(datetime.now(), video_id))
            continue
        # create data loaders
        dataset, data_loader = get_data(video_id, img_path, args.batch_size, args.workers)

        # extract feature
        try:
            print('{}, extracting features...'.format(datetime.now()))
            feat_dict, score_dict = extractor.extract_feature(data_loader, print_summary=False)
            for key, item in feat_dict.items():
                item = to_numpy(item)
                os.makedirs(osp.join(args.save_feat_path,video_id),exist_ok = True)
                img_ind = key.split("_")[-1].split(".jpg")[0]
                if args.save_one_frame_feat:
                    if img_ind is "1":
                        shot_ind = key.split("_")[1]
                        save_fn = osp.join(args.save_feat_path,video_id,"shot_{}.npy".format(shot_ind))
                        np.save(save_fn,item)
                    else:
                        continue
                else:
                    save_fn = osp.join(args.save_feat_path,video_id,"{}.npy".format(key.split(".jpg")[0]))
                    np.save(save_fn,item)
                        
            print('{}, saving...'.format(datetime.now()))
            with open(feat_save_name, 'wb') as f:
                pickle.dump(feat_dict, f)
            with open(score_save_name, 'wb') as f:
                pickle.dump(score_dict, f)
        except Exception as e:
            print('{} error! {}'.format(video_id, e))
        print('\n')


if __name__ == '__main__':
    data_root = "data/demo"
    parser = argparse.ArgumentParser("Place feature using ResNet50 with ImageNet pretrain")
    parser.add_argument('--save-one-frame-feat', action="store_true")
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('--list_file', type=str, default=osp.join(data_root,'meta/list_test.txt'),
                        help='The list of videos to be processed,\
                        in the form of xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n \
                                     or xxxx0\nxxxx1\nxxxx2\n')
    parser.add_argument('--source_img_path', type=str,default=osp.join(data_root,'shot_keyf'))
    parser.add_argument('--save_path',type=str,default=osp.join(data_root,'place_feat_raw'))
    parser.add_argument('--save_feat_path',type=str,default=osp.join(data_root,'place_feat'))
    parser.add_argument('--st', type=int, default=0, help='start number') 
    parser.add_argument('--ed', type=int, default=9999999, help='end number')
    args = parser.parse_args()
    main(args)
