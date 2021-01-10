import random
import sys
import os
from multiprocessing import Manager, Pool, Process
# from PIL import Image
from more_itertools import chunked
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os.path as osp
import numpy as np
from utils.utils import read_json, read_pkl, read_txt_list, strcal
from utils.torch_utils import load_checkpoint, to_numpy
from utils.dataset_utils import pred2scene, scene2video
from multiprocessing import Manager, Pool, Process  # 多进程处理
from mmcv import Config
import argparse
import mbss


def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', default='config.py',
                        help='config file path')
    args = parser.parse_args()
    return args


class Preprocessor(data.Dataset):
    def __init__(self, cfg, list_ids, data_dict):
        self.shot_num = cfg.shot_num
        self.data_root = cfg.data_root
        self.list_ids = list_ids
        self.data_dict = data_dict
        self.shot_boundary_range = range(-cfg.shot_num //
                                         2 + 1, cfg.shot_num // 2 + 1)
        self.mode = cfg.dataset.mode
        assert (len(self.mode) > 0)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id_list = self.list_ids[index]
        if isinstance(id_list, (tuple, list)):
            place_feats, cast_feats, act_feats, aud_feats, sub_feats = [], [], [], [], []
            for id in id_list:
                place_feat, cast_feat, act_feat, aud_feat, sub_feat = self._get_single_item(
                    id)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                sub_feats.append(sub_feat)
            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)
            if 'cast' in self.mode:
                cast_feats = torch.stack(cast_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            if 'sub' in self.mode:
                sub_feats = torch.stack(sub_feats)
            return place_feats, cast_feats, act_feats, aud_feats, sub_feats
        else:
            return self._get_single_item(id_list)

    def _get_single_item(self, id):
        videoid = id['videoid']
        shotid = id['shotid']
        aud_feats, place_feats, cast_feats, act_feats = [], [], [], []
        sub_feats = []
        if 'place' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.npy'.format(strcal(shotid, ind))
                path = osp.join(
                    self.data_root, 'place_feat/{}'.format(videoid), name)
                place_feat = np.load(path)
                place_feats.append(torch.from_numpy(place_feat).float())
        if 'cast' in self.mode:
            for ind in self.shot_boundary_range:
                cast_feat_raw = self.data_dict["casts_dict"].get(
                    videoid).get(strcal(shotid, ind))
                # cast_feat = np.mean(cast_feat_raw, axis=0)
                cast_feats.append(torch.from_numpy(cast_feat_raw).float())
        if 'act' in self.mode:
            for ind in self.shot_boundary_range:
                # act_feat = self.data_dict["acts_dict"].get(videoid).get(strcal(shotid, ind))
                act_feat = self.data_dict["acts_dict"].get(videoid)[
                    int(shotid)+ind]
                # print(len(act_feat))
                # print('act_feat',act_feat[0]['feat'].shape)
                if act_feat:
                    test = act_feat[0]['feat']
                    # exchange dimension 2048 into 512
                    b = [sum(x) / len(x) for x in chunked(test, 4)]
                    act_feats.append(torch.Tensor(b))
                    # act_feats.append(torch.from_numpy(
                    #     act_feat[0]['feat']).float())
                    # print(b)
                else:
                    act_feats.append(
                        torch.Tensor([0]*512).float())
        if 'sub' in self.mode:
            for ind in self.shot_boundary_range:
                sub_feat = self.data_dict["subs_dict"].get(
                    videoid).get(strcal(shotid, ind))
                sub_feats.append(torch.from_numpy(sub_feat).float())
        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.npy'.format(strcal(shotid, ind))
                path = osp.join(
                    self.data_root, 'aud_feat/{}'.format(videoid), name)
                aud_feat = np.load(path)
                aud_feats.append(torch.from_numpy(aud_feat).float())
        if len(place_feats) > 0:
            place_feats = torch.stack(place_feats)
        if len(cast_feats) > 0:
            cast_feats = torch.stack(cast_feats)
        if len(act_feats) > 0:
            act_feats = torch.stack(act_feats)
        if len(sub_feats) > 0:
            sub_feats = torch.stack(sub_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, cast_feats, act_feats, aud_feats, sub_feats


def data_preprocess_one(cfg, video_id, acts_dict, casts_dict, subs_dict):
    data_root = cfg.data_root
    # place_feat_path = osp.join(data_root, 'place_feat')
    # files = os.listdir(osp.join(place_feat_path, video_id))
    # # 获取地点特征
    # all_shot_place_feat = [int(x.split('.')[0].split('_')[1]) for x in files]
    # 获取含有place特征的镜头
    # 当前id周围的窗口的镜头中，如果有一个镜头不在place_feat的key中，或者不再anno的key中，那么就不需要这样的镜头
    # 论文中该值设置为4
    # 获取动作特征
    print('\033[1;36m 正在加载动作特征...\033[0m')
    act_feat_path = osp.join(data_root, 'action_feat/{}.pkl'.format(video_id))
    acts_dict.update({video_id: read_pkl(act_feat_path)})
    # 获取人物特征
    print('\033[1;36m 正在加载演员特征...\033[0m')
    cast_feat_path = osp.join(data_root, "cast_feat/{}.pkl".format(video_id))
    casts_dict.update({video_id: read_pkl(cast_feat_path)})
    print('\033[1;36m 正在加载字幕特征...\033[0m')
    sub_feat_path = osp.join(data_root, "sub_feat/{}.pkl".format(video_id))
    subs_dict.update({video_id: read_pkl(sub_feat_path)})


def data_preprocess(cfg):
    data_root = cfg.data_root
    with open(osp.join(data_root, 'info/video_list.txt'), 'r') as fv:
        video_list = [os.path.splitext(i)[0] for i in fv.readlines()]
    # 获取全部的movie id
    mgr = Manager()
    acts_dict_raw = mgr.dict()
    casts_dict_raw = mgr.dict()
    subs_dict_raw = mgr.dict()
    jobs = [Process(
        target=data_preprocess_one,
        args=(cfg, video_id, acts_dict_raw, casts_dict_raw, subs_dict_raw)) for video_id in
        video_list]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    acts_dict, casts_dict = {}, {}
    subs_dict = {}
    for key, value in acts_dict_raw.items():
        acts_dict.update({key: value})
    for key, value in casts_dict_raw.items():
        casts_dict.update({key: value})
    for key, value in subs_dict_raw.items():
        subs_dict.update({key: value})
    return video_list, casts_dict, acts_dict, subs_dict


def data_partition(cfg, video_list, shots_dict):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2
    # 以序列长度为一组，按顺序选择每个电影镜头
    # 如序列长度为10，电影共55个镜头 则 [01,01,...,09],[10,11,...,19],...,[41,42,...49]
    # 其余的镜头舍弃 对下一个电影进行重复的操作
    idxs = []
    win_len = (cfg.seq_len + cfg.shot_num) // 2
    for videoid in video_list:
        shot_max = shots_dict[videoid]
        shotid_tmp = 0
        # 对每个电影镜头进行遍历
        for shotid in range(shot_max):
            if int(shotid) < shotid_tmp + seq_len_half:
                continue
            shotid_tmp = int(shotid) + seq_len_half
            one_idxs = []
            if int(shotid) < shot_max-win_len:
                for idx in range(-seq_len_half + 1, seq_len_half + 1):

                    one_idxs.append(
                        {'videoid': videoid, 'shotid': strcal(shotid, idx)})
                idxs.append(one_idxs)
    return idxs


def get_each_video_shots(data_root):
    with open(osp.join(data_root, 'info/video_list.txt'), 'r') as fv:
        video_list = [os.path.splitext(i)[0] for i in fv.readlines()]
    video_max_shots = {}
    for video_id in video_list:
        with open(osp.join(data_root, 'shot_txt', video_id+'.txt'), 'r') as fl:
            count = len(fl.readlines())
        video_max_shots[video_id] = count
    return video_max_shots


def get_preprocessed_data_test(cfg_path):
    cfg = Config.fromfile(cfg_path)
    shots_dict = get_each_video_shots(cfg.data_root)
    video_list, casts_dict, acts_dict, subs_dict = data_preprocess(cfg)
    data_dict = {
        'shots_dict': shots_dict,
        'casts_dict': casts_dict,
        'acts_dict': acts_dict,
        'subs_dict': subs_dict
    }
    list_ids = data_partition(cfg, video_list, shots_dict)
    infer_video_set = Preprocessor(cfg, list_ids, data_dict)
    a = infer_video_set[0]
    # print(a)
    return infer_video_set


def video_infer(cfg, model, infer_loader):
    model.eval()
    test_loss = 0
    gt1, gt0, all_gt = 0, 0, 0
    prob_raw, gts_raw = [], []
    preds = []
    batch_num = 0
    with torch.no_grad():
        for data_place, data_cast, data_act, data_aud, data_sub in infer_loader:
            batch_num += 1
            data_place = data_place.cuda() if 'place' in cfg.dataset.mode else []
            data_cast = data_cast.cuda() if 'cast' in cfg.dataset.mode else []
            data_act = data_act.cuda() if 'act' in cfg.dataset.mode else []
            data_aud = data_aud.cuda() if 'aud' in cfg.dataset.mode else []
            data_sub = data_sub.cuda() if 'sub' in cfg.dataset.mode else []
            output = model(data_place, data_cast, data_act, data_aud, data_sub)
            output = output.view(-1, 2)

            output = F.softmax(output, dim=1)
            prob = output[:, 1]

            prob_raw.append(to_numpy(prob))

            # prediction = np.nan_to_num(
            #     prob.squeeze().cpu().detach().numpy()) > 0.5

        for x in prob_raw:
            preds.extend(x.tolist())

        return preds


def save_pred_seq(cfg, loader, preds):
    if not os.path.exists(osp.join(cfg.data_root, 'pred')):
        os.mkdir(osp.join(cfg.data_root, 'pred'))
    for threshold in np.arange(0, 1.01, 0.1):
        pred_fn = osp.join(cfg.data_root,
                           'pred/pred_{:.2f}.txt'.format(threshold))
        n = min(len(loader.dataset.list_ids), len(preds)//cfg.seq_len)
        tmp = np.array(preds, np.float32)
        tmp = (tmp > threshold).astype(np.int32)
        with open(pred_fn, "w") as f:
            for i in range(n):
                for j in range(cfg.seq_len):
                    one_idx = loader.dataset.list_ids[i][j]
                    f.write('{} {} {}\n'.format(
                        one_idx['videoid'], one_idx['shotid'], tmp[i*cfg.seq_len+j]))


if __name__ == '__main__':
    args = parse_args()
    infer_video_set = get_preprocessed_data_test(args.config)

    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

    infer_loader = DataLoader(
        infer_video_set, batch_size=cfg.batch_size,
        shuffle=False, **cfg.data_loader_kwargs)

    # load model
    model = mbss.movie_bert.__dict__[cfg.model.name](cfg).cuda()
    model = nn.DataParallel(model)

    # load the parameters of models
    checkpoint = load_checkpoint(
        osp.join(cfg.logger.logs_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    preds = video_infer(cfg, model, infer_loader)
    save_pred_seq(cfg, infer_loader, preds)
    # _, scene_list = pred2scene(cfg)
    # scene2video(cfg, scene_list)
