import argparse
import os
import os.path as osp
import mmcv
from movienet.tools.utils import read_movie_list
import tempfile
# from movienet.tools.action_extractor.src.dataset import ActionDataset
from movienet.tools.action_extractor.src.video import (VideoFileBackend,
                                                       VideoMMCVBackend,
                                                       VideoDecordBackend)
from movienet.tools.detector.parallel_persondetector import (
    ParallelPersonDetector)
from movienet.tools.action_extractor.action_extract_manager import (
    ActionExtractManager)
from movienet.tools.action_extractor.action_extractor import (
    ParallelActionExtractor)
import torch
import shutil


def main(args):
    movie_ids = read_movie_list(args.video_list)
    movie_ids = [i.split('.m')[0] for i in movie_ids]
    manager = ActionExtractManager()
    if args.detect:
        assert not args.tracklet_dir
        if args.temp_dir is None:
            args.temp_dir = tempfile.mkdtemp()
            print(f'create temp dir {args.temp_dir}')
        args.tracklet_dir = args.temp_dir
        mmcv.mkdir_or_exist(args.temp_dir)
        detector = ParallelPersonDetector(
            args.arch,
            args.detector_cfg,
            args.detector_weight,
            gpu_ids=list(range(args.ngpu)))
        for movie_id in movie_ids:
            shot_file = osp.join(args.data_root_path, 'shot_txt', f"{movie_id}.txt")
            # One could replace VideoFileBackend to other type of backends
            # like VideoMMCVBackend.
            # video = VideoFileBackend(
            #     'twolevel',
            #     osp.join(args.movienet_root, 'frame', movie_id),
            #     shot_file=shot_file)
            video = VideoMMCVBackend(
                osp.join(args.videos_path, f"{movie_id}.mp4"))
            tracklets = manager.run_detect(
                detector, video, shot_file, imgs_per_gpu=args.imgs_per_gpu)
            tracklet_file = osp.join(args.temp_dir, f"{movie_id}.pkl")
            mmcv.dump(tracklets, tracklet_file)
        del detector
        torch.cuda.empty_cache()

    extractor = ParallelActionExtractor(
        args.extractor_cfg,
        args.extractor_weight,
        gpu_ids=list(range(args.ngpu)))
    mmcv.mkdir_or_exist(args.save_dir)

    for movie_id in movie_ids:
        shot_file = osp.join(args.data_root_path, 'shot_txt', f"{movie_id}.txt")
        # video = VideoFileBackend(
        #     'twolevel',
        #     osp.join(args.movienet_root, 'frame', movie_id),
        #     shot_file=shot_file)
        video = VideoMMCVBackend(
            osp.join(args.videos_path, f"{movie_id}.mp4"))

        tracklet_file = osp.join(args.tracklet_dir, f"{movie_id}.pkl")
        result = manager.run_extract(extractor, video, shot_file,
                                     tracklet_file)
        mmcv.dump(result, osp.join(args.save_dir, f"{movie_id}.pkl"))

    if args.detect:
        shutil.rmtree(args.temp_dir)
        print(f'deleted temp dir {args.temp_dir}')


if __name__ == '__main__':
    my_experiment_dir = 'data'
    root_path = osp.dirname((osp.dirname(osp.dirname(__file__))))
    myexp_path = osp.join(root_path, my_experiment_dir)

    parser = argparse.ArgumentParser('Extract Action Feature')
    parser.add_argument('--video_list', type=str, default=osp.join(myexp_path, 'info','video_list.txt'))
    parser.add_argument('--save_dir', type=str, default=osp.join(myexp_path, 'action_feat'))
    parser.add_argument('--data_root_path', type=str, default=myexp_path, help="all videos path")
    parser.add_argument('--videos_path', type=str, default=osp.join(root_path,'videos'), help="all videos path")
    parser.add_argument('--imgs_per_gpu', type=int, default=1)
    parser.add_argument(
        '--detector_cfg',
        type=str,
        default=osp.join(root_path,
                         'pre/pretrained_models/cascade_rcnn_x101_64x4d_fpn.json'))
    parser.add_argument(
        '--extractor_cfg',
        type=str,
        default=osp.join(root_path,
                         'pre/pretrained_models/ava_fast_rcnn_nl_r50_c4_1x_kinetics.py'))
    parser.add_argument(
        '--arch',
        type=str,
        choices=['rcnn', 'retina'],
        default='rcnn',
        help='architechture of the detector')
    parser.add_argument(
        '--detector_weight',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/cascade_rcnn_x101_64x4d_fpn.pth'),
        help='the weight of the model')
    parser.add_argument(
        '--extractor_weight',
        type=str,
        default=osp.join(root_path,
                         'pre/pretrained_models/ava_fast_rcnn_nl_r50_c4_1x_kinetics.pth'),
        help='the weight of the model')
    parser.add_argument('--tracklet_dir', default=None, type=str)
    parser.add_argument('--temp_dir', default=osp.join(myexp_path, 'temp_files'), type=str)
    parser.add_argument('--ngpu', type=int, default=1)

    parser.add_argument('--detect', action='store_true',default=True)

    args = parser.parse_args()

    main(args)
