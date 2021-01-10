import argparse
import os
import os.path as osp
import pickle
import cv2
from movienet.tools import PersonDetector,PersonExtractor
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
        '--save_path', type=str, default=osp.join(myexp_path,'cast_detected'), help='path to save result')

    parser.add_argument(
        '--arch',
        type=str,
        choices=['rcnn', 'retina'],
        default='rcnn',
        help='architechture of the detector')
    parser.add_argument(
        '--cfg',
        type=str,
        default=osp.join(root_path,
                         'pre/pretrained_models/cascade_rcnn_x101_64x4d_fpn.json'),
        help='the config file of the model')
    parser.add_argument(
        '--weight',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/cascade_rcnn_x101_64x4d_fpn.pth'),
        help='the weight of the model')
    parser.add_argument(
        '--extractor_weight',
        type=str,
        default=osp.join(root_path, 'pre/pretrained_models/resnet50_csm.pth'),
        help='the weight of the model')
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
    
    assert osp.isfile(args.cfg)
    assert osp.isfile(args.weight)

    detector = PersonDetector(args.arch, args.cfg, args.weight)
    extractor = PersonExtractor(args.extractor_weight, gpu=0)

    fc_detector = FaceDetector(args.fc_cfg, args.fc_detect_weight)
    fc_extractor = FaceExtractor(args.fc_extract_weight, gpu=0)
    
    videos_list = [x.strip().split('.')[0] for x in open(args.videos_list)]  # 考虑到其他格式的video
    # videos_list = [x.strip().split('.m')[0] for x in open(args.videos_list)]  
    if videos_list:
        for video in videos_list:

            imglist = os.listdir(osp.join(myexp_path,'shot_keyf',video))
            if not os.path.exists(osp.join(args.save_path,video)):
                os.makedirs(osp.join(args.save_path,video))
            for n,img in enumerate(imglist):
                save_prefix = img.split('.j')[0]
                img = osp.join(osp.join(myexp_path,'shot_keyf',video),img)
                persons = detector.detect(img, show=False, conf_thr=0.9)
                im = cv2.imread(img)
                # assert persons.shape[0] == 2
                print('in the {} image,{} persons detected!'.format(n+1,persons.shape[0]))
                
                person_imgs = detector.crop_person(im, persons,save_dir=osp.join(args.save_path,video),save_prefix=save_prefix)
                print('the persons {} in the image {} persons have processeded!'.format(persons.shape[0],n+1))




    # extract person feature
    for video in videos_list:
        persons_in_shots = os.listdir(osp.join(myexp_path,'cast_detected',video))
        persons_shots_list = [i.split('_')[1] for i in persons_in_shots]
        shot_persons_dict =list_duplicates(persons_shots_list,persons_in_shots)
        video_shots_list = list(set(os.listdir(osp.join(myexp_path,'shot_keyf',video))))
        video_shots_list = [i.split('_')[1] for i in video_shots_list]
        video_cast_feat = defaultdict(dict)
        for shot in sorted(video_shots_list):
            if shot_persons_dict[shot]:
                persons = shot_persons_dict[shot]
                video_cast_feat[shot] = np.zeros((256,),dtype=np.float32)
                for person in persons:
                    img = osp.join(osp.join(myexp_path,'cast_detected',video),person)
                    im = cv2.imread(img)
                    
                    im_feat = extractor.extract(im)
                    im_feat /= np.linalg.norm(im_feat)
                    video_cast_feat[shot] += im_feat
       
            else:
                video_cast_feat[shot] = np.zeros((256,),dtype=np.float32)
        if not os.path.exists(osp.join(myexp_path,'cast_feat')):
            os.makedirs(osp.join(myexp_path,'cast_feat'))
        with open(osp.join(myexp_path,'cast_feat',video+'.pkl'),'wb') as f:
            pickle.dump(video_cast_feat, f)
        print('have processed the video {}'.format(video))




