# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import numpy as np
import json
import glob
import torch

from shapely.geometry import Polygon, box
from os.path import join, realpath, dirname, normpath


## for OTB benchmark
def load_dataset(dataset):
    # buffer controls whether load all images
    info = {}

    if 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), '../dataset', dataset)
        json_path = join(realpath(dirname(__file__)), '../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect'])-[1, 1, 0, 0]
            info[v]['name'] = v

    elif 'VOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3],
                                      gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3], gt[:, 0] + gt[:, 2], gt[:, 1]))

            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    else:
        print('Not support now, edit for other dataset youself...')
        exit()

    return info


## for a single video
def load_video(video):
    # buffer controls whether load all images
    info = {}
    info[video] = {}

    base_path = normpath(join(realpath(dirname(__file__)), '../dataset', video))

    # ground truth
    gt_path = join(base_path, 'groundtruth_rect.txt')
    gt = np.loadtxt(gt_path, delimiter=',')
    gt = gt - [1, 1, 0, 0]   # OTB for python (if video not from OTB, please delete it)

    # img file name
    img_path = join(base_path, 'img', '*jpg')
    image_files = sorted(glob.glob(img_path))

    # info summary
    info[video]['name'] = video
    info[video]['image_files'] = [normpath(join(base_path, 'img', im_f)) for im_f in image_files]
    info[video]['gt'] = gt

    return info


## for loading pretrained model
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(len(missing_keys)))
    print('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# others
def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return [rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]]  # 0-index


def get_min_max_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def judge_overlap(poly, rect):
    xy = poly.reshape(-1, 2)
    polygon_shape = Polygon(xy)
    gridcell_shape = box(rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
    # The intersection
    overlap = polygon_shape.intersection(gridcell_shape).area
    return True if overlap > 1 else False