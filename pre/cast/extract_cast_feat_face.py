import argparse
import os
import os.path as osp
import pickle
import cv2
# from movienet.tools import PersonDetector,PersonExtractor
from movienet.tools import FaceDetector, FaceExtractor
import threading
import numpy as np
from collections import defaultdict

def list_duplicates(seq,src):
    tally = defaultdict(list)
    # tally = {}
    for i,item in enumerate(seq):
        tally[item].append(src[i])

    return tally



if __name__ == '__main__':
    my_experiment_dir = 'data'
    root_path = osp.dirname((osp.dirname(osp.dirname(__file__))))
    myexp_path = osp.join(root_path, my_experiment_dir)
    parser = argparse.ArgumentParser('Detect Person')
    parser.add_argument('--videos_list',type=str,default=osp.join(myexp_path,'info','video_list.txt'))
    parser.add_argument(
        '--key_frames_path', type=str, default=osp.join(myexp_path,'shot_keyf'))
    parser.add_argument(
        '--save_faces', type=str, default=True)
    parser.add_argument(
        '--save_faces_path', type=str, default=osp.join(myexp_path,'face_detected'))
    parser.add_argument(
        '--fc_cfg',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/mtcnn.json'))
    parser.add_argument(
        '--fc_detect_weight',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/mtcnn.pth'))
    parser.add_argument(
        '--fc_extract_weight',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/irv1_vggface2.pth'))
    args = parser.parse_args()
    
    if not os.path.exists(args.save_faces_path):
        os.makedirs(args.save_faces_path)
    
    assert osp.isfile(args.fc_cfg)
    assert osp.isfile(args.fc_detect_weight)
    assert osp.isfile(args.fc_extract_weight)
    
    fc_detector = FaceDetector(args.fc_cfg, args.fc_detect_weight)
    fc_extractor = FaceExtractor(args.fc_extract_weight, gpu=0)
    
    video_ids_list = [osp.splitext(i)[0] for i in open(args.videos_list,'r').readlines()]
    for i in video_ids_list:
        cur_id_shots_path = osp.join(args.key_frames_path,i)
        # select the 2-th keyframes ,others disboard
        key_frames = [i for i in os.listdir(cur_id_shots_path) if '_1.jpg' in i] 
        video_cast_feat = defaultdict(dict)
        for frame in sorted(key_frames):
            im = cv2.imread(osp.join(cur_id_shots_path,frame))
            ## detect and crop face
            faces, _ = fc_detector.detect(im)
            frame_id = frame.split('_')[1]
            if args.save_faces:
                if not os.path.exists(osp.join(args.save_faces_path,i)):
                    os.mkdir(osp.join(args.save_faces_path,i))
                face_imgs = fc_detector.crop_face(im, faces,save_dir=osp.join(args.save_faces_path,i),save_prefix='shot_'+frame_id+'_face')
            face_imgs = fc_detector.crop_face(im, faces)
            if face_imgs:
                length = len(face_imgs)
                video_cast_feat[frame_id] = np.zeros((512,),dtype=np.float32)
                for fc_img in face_imgs:
                    fc_feat = fc_extractor.extract(fc_img) / length
                    fc_feat /= np.linalg.norm(fc_feat)
                    video_cast_feat[frame_id] += fc_feat
            else:
                video_cast_feat[frame_id] = np.zeros((512,),dtype=np.float32)
            print('videoid: ',i,' shot: ',frame, ' has been processed.')
        
        if not os.path.exists(osp.join(myexp_path,'cast_feat')):
                os.makedirs(osp.join(myexp_path,'cast_feat'))
        with open(osp.join(myexp_path,'cast_feat',i +'.pkl'),'wb') as f:
            pickle.dump(video_cast_feat, f)
        print('have processed the video {}'.format(i))





