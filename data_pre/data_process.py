import os

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os.path as osp
from utils.utils import read_json, read_pkl, read_txt_list, strcal
from multiprocessing import Manager, Pool, Process  # 多进程处理
from mmcv import Config
import logging


class Preprocessor(data.Dataset):
    def __init__(self, cfg, list_ids, data_dict):
        self.shot_num = cfg.shot_num
        self.data_root = cfg.data_root
        self.list_ids = list_ids
        self.data_dict = data_dict
        self.shot_boundary_range = range(-cfg.shot_num // 2 + 1, cfg.shot_num // 2 + 1)
        self.mode = cfg.dataset.mode
        assert (len(self.mode) > 0)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        id_list = self.list_ids[index]
        if isinstance(id_list, (tuple, list)):
            place_feats, cast_feats, act_feats, aud_feats,sub_feats, labels = [], [], [], [], [],[]
            for id in id_list:
                place_feat, cast_feat, act_feat, aud_feat,sub_feat, label = self._get_single_item(id)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                sub_feats.append(sub_feat)
                labels.append(label)
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
            labels = np.array(labels)
            return place_feats, cast_feats, act_feats, aud_feats,sub_feats, labels
        else:
            return self._get_single_item(id_list)

    def _get_single_item(self, id):
        imdbid = id['imdbid']
        shotid = id['shotid']
        label = self.data_dict["annos_dict"].get(imdbid).get(shotid)
        aud_feats, place_feats, cast_feats, act_feats = [], [], [], []
        sub_feats = []
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
        if 'sub' in self.mode:
            for ind in self.shot_boundary_range:
                sub_feat = self.data_dict["subs_dict"].get(imdbid).get(strcal(shotid, ind))
                sub_feats.append(torch.from_numpy(sub_feat).float())
        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:
                name = 'shot_{}.npy'.format(strcal(shotid, ind))
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
        if len(sub_feats) > 0:
            sub_feats = torch.stack(sub_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, cast_feats, act_feats, aud_feats,sub_feats, label


# TODO:-1的表示镜头的什么
def get_anno_dict(anno_path):
    contents = read_txt_list(anno_path)
    anno_dict = {}
    for content in contents:
        shotid = content.split(' ')[0]
        value = int(content.split(' ')[1])
        if value >= 0:
            anno_dict.update({shotid: value})
        elif value == -1:
            anno_dict.update({shotid: 1})
    return anno_dict


def data_preprocess_one(cfg, imdbid, acts_dict, casts_dict,subs_dict, annos_dict, annos_valid_dict):
    data_root = cfg.data_root
    label_path = osp.join(data_root, 'label318')
    place_feat_path = osp.join(data_root, 'place_feat')
    # 滑动窗口的大小
    win_len = cfg.seq_len + cfg.shot_num
    files = os.listdir(osp.join(place_feat_path, imdbid))
    # 获取地点特征
    print('place features is loading...')
    all_shot_place_feat = [int(x.split('.')[0].split('_')[1]) for x in files]
    # 获取标签
    
    anno_path = '{}/{}.txt'.format(label_path, imdbid)
    anno_dict = get_anno_dict(anno_path)
    annos_dict.update({imdbid: anno_dict})
    anno_valid_dict = anno_dict.copy()
    # 获取含有place特征的镜头
    # 当前id周围的窗口的镜头中，如果有一个镜头不在place_feat的key中，或者不再anno的key中，那么就不需要这样的镜头,并更新labels
    # 论文中该值设置为4
    shotids = [int(x) for x in anno_valid_dict.keys()]
    to_be_del = []
    for shotid in shotids:
        del_flag = False
        for idx in range(-(win_len) // 2 + 1, win_len // 2 + 1):
            if ((shotid + idx) not in all_shot_place_feat) or \
                    anno_dict.get(str(shotid+idx).zfill(4)) is None:
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)
    for shotid in to_be_del:
        del anno_valid_dict[str(shotid).zfill(4)]
    # 更新可用的镜头
    # print('正在更新有用的镜头，当前窗口大小是{}'.format(win_len))
    print('labels is loading...')
    annos_valid_dict.update({imdbid: anno_valid_dict})
    # 获取动作特征
    print('action features is loading...')
    act_feat_path = osp.join(data_root, 'act_feat/{}.pkl'.format(imdbid))
    acts_dict.update({imdbid: read_pkl(act_feat_path)})
    # 获取人物特征
    print('cast features is loading...')
    cast_feat_path = osp.join(data_root, "cast_feat/{}.pkl".format(imdbid))
    casts_dict.update({imdbid: read_pkl(cast_feat_path)})
    print('subtitle features is loading...')
    sub_feat_path = osp.join(data_root, "sub_feat/{}.pkl".format(imdbid))
    subs_dict.update({imdbid: read_pkl(sub_feat_path)})


def data_preprocess(cfg):
    data_root = cfg.data_root
    imdbid_list_json_path = osp.join(data_root, 'meta/split318.json')
    imdbid_list_json = read_json(imdbid_list_json_path)
    # 获取全部的movie id
    imdbid_list = imdbid_list_json['all']
    mgr = Manager()
    acts_dict_raw = mgr.dict()
    casts_dict_raw = mgr.dict()
    subs_dict_raw = mgr.dict()
    annos_dict_raw = mgr.dict()
    annos_valid_dict_raw = mgr.dict()
    # 分配多进程工作
    jobs = [Process(
        target=data_preprocess_one,
        args=(cfg, imdbid, acts_dict_raw, casts_dict_raw,subs_dict_raw, annos_dict_raw, annos_valid_dict_raw)) for imdbid in
        imdbid_list]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    annos_dict, annos_valid_dict, acts_dict, casts_dict = {}, {}, {}, {}
    subs_dict={}
    for key, value in annos_dict_raw.items():
        annos_dict.update({key: value})
    for key, value in annos_valid_dict_raw.items():
        annos_valid_dict.update({key: value})
    for key, value in acts_dict_raw.items():
        acts_dict.update({key: value})
    for key, value in casts_dict_raw.items():
        casts_dict.update({key: value})
    for key, value in subs_dict_raw.items():
        subs_dict.update({key: value})
    return imdbid_list_json, annos_dict, annos_valid_dict, casts_dict, acts_dict,subs_dict


def data_partition(cfg, imdbid_list_json, annos_dict):
    assert (cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len // 2
    idxs = []
    for mode in ['train', 'test', 'val']:
        one_mode_idxs = []
        # 以序列长度为一组，按顺序选择每个电影镜头
        # 如序列长度为10，电影共55个镜头 则 [01,01,...,09],[10,11,...,19],...,[41,42,...49]
        # 其余的镜头舍弃 对下一个电影进行重复的操作
        for imdbid in imdbid_list_json[mode]:
            anno_dict = annos_dict[imdbid]
            shotid_list = sorted(anno_dict.keys())
            shotid_tmp = 0
            # 对每个imdbid的电影镜头进行遍历
            for shotid in shotid_list:
                if int(shotid) < shotid_tmp + seq_len_half:
                    continue
                if mode == 'train':
                    shotid_tmp = int(shotid) + cfg.sample_stride - seq_len_half
                else:
                    shotid_tmp = int(shotid) + seq_len_half
                # shotid_tmp = cfg.sample_stride
                
                one_idxs = []
                for idx in range(-seq_len_half + 1, seq_len_half + 1):
                    one_idxs.append({'imdbid': imdbid, 'shotid': strcal(shotid, idx)})
                one_mode_idxs.append(one_idxs)
        idxs.append(one_mode_idxs)
    partition = {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[1]
    partition['val'] = idxs[2]
    return partition


def get_preprocessed_data_test(cfg_path):
    cfg = Config.fromfile(cfg_path)
    imdbid_list_json, annos_dict, annos_valid_dict, casts_dict, acts_dict,subs_dict = data_preprocess(cfg)
    partition = data_partition(cfg,imdbid_list_json,annos_valid_dict)
    if partition:
        print('Test of get preprocessed data is successful !')


if __name__ == '__main__':
    moviebert_dir = osp.dirname(osp.dirname(__file__))
    cfg_path = osp.join(moviebert_dir, 'config.py')
    get_preprocessed_data_test(cfg_path)
