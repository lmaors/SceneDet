import os.path as osp
import os
import cv2
import subprocess
from tqdm import tqdm
import argparse

def reduce_scene_list(video_file):
    video_id = os.path.splitext(video_file)[0]
    with open('data/shot_txt/{}.txt'.format(video_id), 'r') as ft:
        shot_list = [i.strip() for i in ft.readlines()]
        num_shots = len(shot_list)
        shot_frame_dict = {}
        for n, shot in enumerate(shot_list):
            frames_each_frames = shot.split(' ')
            shot_id = str(n+1).zfill(4)
            shot_frame_dict[shot_id] = (
                frames_each_frames[0], frames_each_frames[1])
        # print(shot_frame_dict)

    with open('data/pred/pred_0.50.txt', 'r') as fp:
        pred_list = [i.strip() for i in fp.readlines() if video_id in i]

        pred_dict = {}
        for i in pred_list:
            shot = i.split(' ')[1]
            pred_value = i.split(' ')[2]
            # print(pred_value)
            pred_dict[shot] = pred_value
        #
        if len(pred_dict) < len(shot_frame_dict):
            for i in range(len(pred_list), len(shot_frame_dict)):
                pred_dict[str(i).zfill(4)] = '0'
            pred_dict[str(len(shot_frame_dict)).zfill(4)] = '1'
    # print(pred_dict)
    # print(len(pred_dict))
    # print(pred_shots)

    scenes_pair_shots = []
    scenes_pair_frames = []
    start_shot = 1
    end_shot = 1
    for n, m in enumerate(sorted(pred_dict.keys())):
        i = pred_dict[m]
        if i == '1':
            end_shot = n+1

            scenes_pair_shots.append(
                (str(start_shot).zfill(4), str(end_shot).zfill(4)))
            shot_start_framme = shot_frame_dict[str(start_shot).zfill(4)][0]
            shot_end_frame = shot_frame_dict[str(end_shot).zfill(4)][1]
            scenes_pair_frames.append((shot_start_framme, shot_end_frame))
            start_shot = n + 2

    # print(scenes_pair_shots)
    # print(scenes_pair_frames)
    return scenes_pair_frames




def scene2video(args, scene_list, video_file):
    print("...scene list to videos process of {}".format(video_file))
    # 获取 OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # source_movie_fn = '{}.mp4'.format(
    #     osp.join(cfg.data_root, "video", cfg.video_name))
    # vcap = cv2.VideoCapture(source_movie_fn)
    # if int(major_ver)  < 3 :
    #     fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    # else :
    #     fps = vcap.get(cv2.CAP_PROP_FPS)
    #     print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    source_movie_fn = 'videos/{}'.format(video_file)
    vcap = cv2.VideoCapture(source_movie_fn)
    fps = vcap.get(cv2.CAP_PROP_FPS)  # video.fps
    # out_video_dir_fn = osp.join(cfg.data_root, "scene_video", cfg.video_name)
    # mkdir_ifmiss(out_video_dir_fn)
    out_video_dir_fn = osp.join('scenes_result',video_file)
    if not os.path.exists(out_video_dir_fn):
        os.mkdir(out_video_dir_fn)
        
    for scene_ind, scene_item in tqdm(enumerate(scene_list)):
        scene = str(scene_ind).zfill(4)
        start_frame = int(scene_item[0])
        end_frame = int(scene_item[1])

        start_time, end_time = start_frame/fps, end_frame/fps
        duration_time = end_time - start_time
        out_video_fn = osp.join(out_video_dir_fn, "scene_{}.mp4".format(scene))
        if osp.exists(out_video_fn):
            continue
        call_list = ['ffmpeg']
        call_list += ['-v', 'quiet']
        call_list += [
            '-y',
            '-ss',
            str(start_time),
            '-t',
            str(duration_time),
            '-i',
            source_movie_fn]
        call_list += ['-map_chapters', '-1']
        call_list += [out_video_fn]
        subprocess.call(call_list)
        print("...scene videos has been saved in {}".format(out_video_fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scene2video')
    parser.add_argument('--save_scene_path',type=str,default='scenes_result')
    parser.add_argument('--videos_path',type=str,default='videos')
    args = parser.parse_args()
    with open('data/info/video_list.txt', 'r') as fs:
        video_list = [i.strip() for i in fs.readlines()]
    for video_file in video_list:
        print('processing video {}'.format(video_file))
        scene_list = reduce_scene_list(video_file)
        # print(scene_list)
        scene2video(args,scene_list,video_file)
    
