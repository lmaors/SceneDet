import pysrt
import os.path as osp
import os
import argparse
import json
import math
import pandas as pd

def args_parser():
    parser = argparse.ArgumentParser('subtitle to feature of bert')
    parser.add_argument('--subtitle_path', type=str, default='subtitles')
    parser.add_argument('--video_shot_info_path', type=str, default='data/shot_txt')
    return parser.parse_args()


def load_subtitles(args, sub_id, id_fps):
    # 获取s

    sub_srt_object = pysrt.open(osp.join(args.subtitle_path, sub_id + '.en.srt'))
    sub_info = []
    for i in sub_srt_object:
        start_time = i.start.ordinal
        start_frame = math.ceil(start_time * id_fps / 1000)
        end_time = i.end.ordinal
        end_frame = math.floor(end_time * id_fps / 1000)
        text = i.text_without_tags.replace('- ', ' ').strip().replace("\n", "")
        sub_info.append(dict(
            start_frame=str(start_frame),
            end_frame=str(end_frame),
            subtitle=text
        ))
    return sub_info


def get_shot_subtitle(shot_info, sub_info):
    # 两个列表的位置
    shot_posi = 0
    for sub in sub_info:
        sub_sf = sub['start_frame']
        sub_ef = sub['end_frame']
        for shot in shot_info[shot_posi:]:
            shot_sf = shot['start_frame']
            shot_ef = shot['end_frame']
            if (int(shot_ef) < int(sub_sf)) or (int(shot_sf) > int(sub_sf)):
                shot_posi += 1
                continue
            else:
                shot['subtitle'] += sub['subtitle']
                break
    return shot_info



if __name__ == '__main__':
    args = args_parser()
    # print(load_subtitles(args))
     
    root_path = osp.dirname(osp.dirname(osp.dirname(__file__)))
    data_path = osp.join(root_path,'data')
    movie_ids_list = os.listdir(osp.join(data_path,'shot_stats'))
    fps_dict = {}
    for i in movie_ids_list:
        with open(osp.join(data_path,'shot_stats',i), "r") as file:
            fps_dict[i.split('.cs')[0]] = float(file.readline().split(',')[-1].strip())

    # movie_ids_list = [i.split('.csv')[0] for i in movie_ids_list]
    
    subtitle_path = args.subtitle_path

    sub_ids_list = [i.split('.en.srt')[0] for i in os.listdir(subtitle_path)]


    movies_subs_dict = dict()
    for id in sorted(sub_ids_list):
        id_fps = fps_dict[id]
        # 获取当前电影字幕信息
        sub_info_list = load_subtitles(args, id, id_fps)
        # 获取了镜头信息
        shot_info_dict = dict()
        shot_info_list = list()
        with open(osp.join(args.video_shot_info_path, id + '.txt'), 'r') as f:
            for shot_id, line in enumerate(f.readlines()):
                start_frame = line.split(' ')[0]
                end_frame = line.split(' ')[1]
                text = ''
                shot_info_dict[str(shot_id).zfill(4)] = dict(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    subtitle=text
                )
                shot_info_list.append(dict(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    subtitle=text
                ))
        # 将镜头信息与字幕信息进行处理，得到镜头字幕信息
        shot_subtitle = get_shot_subtitle(shot_info_list, sub_info_list)
        movies_subs_dict[id] = shot_subtitle
        print('the shot\'s subtitle of movie {} has processed...'.format(id))
    if not os.path.exists(osp.join(data_path,'info')):
        os.mkdir(osp.join(data_path,'info'))
    with open(osp.join(data_path,'info/shot_sub.json'), 'w') as f:
        json.dump(movies_subs_dict,f)
        print('json has been written.')