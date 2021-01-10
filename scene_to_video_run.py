import subprocess


def subprocess_run():
    # tensorflow2.0 python interpreter used to process sentence embedding with albert
    tf2_python = '/home/lcc/pyenvs/tensorflow2/bin/python'
    # pytorch1.5 python interpreter
    th_python = '/home/lcc/pyenvs/th15cuda102/bin/python'

    # detect shots default settings: --save_keyf True, --save_keyf_txt True,--split_video True
    print('\033[1;36m Detecting shots ...\033[0m')
    subprocess.run([th_python, 'pre/ShotDetect/shotdetect_p.py',
                    '--source_path', 'videos', '--save_data_root_path', 'data'])

    # # extract places features default settings: --save-one-frame-feat True
    print('\033[1;36m Extracting places features ...\033[0m')
    subprocess.run([th_python, 'pre/place/extract_place_feat.py'])

    print('\033[1;36m Extracting actions features ...\033[0m')
    subprocess.run([th_python, 'pre/action/extract_action_feat.py'])

    print('\033[1;36m Extracting casts features ...\033[0m')
    subprocess.run([th_python, 'pre/cast/extract_cast_feat_face.py'])

    print('\033[1;36m Extracting audios features ...\033[0m')
    subprocess.run([th_python, 'pre/audio/extract_audio_feat.py'])

    # currently, the type of subtitle is support for srt, and is necessary
    print(
        '\033[1;36m Exchange original subtitle into shot-subtitle clips ...\033[0m')
    subprocess.run([th_python, 'pre/subtitle/subtitle2shotsubtitle.py'])

    print('\033[1;36m Extracting subtitles features with albert model...\033[0m')
    subprocess.run([tf2_python, 'pre/subtitle/extract_sub_feat.py'])

    print('\033[1;32m All features have been processed.\033[0m')

    print('\033[1;32m Inferencing ...\033[0m')
    subprocess.run([th_python, 'inference.py'])

    print('\033[1;32m Scenes to videos ...\033[0m')
    subprocess.run([th_python, 'pred_list_to_scenes.py'])


subprocess_run()
