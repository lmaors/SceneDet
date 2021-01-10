import nltk
# from nltk.tokenize import WordPunctTokenizer
# from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import numpy as np
import os
import os.path as osp
import json
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
root_path = osp.dirname(osp.dirname(osp.dirname(__file__)))
data_path = osp.join(root_path,'data')
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(
    base_dir, 'pretrained_models/albert_base/albert_config.json')
checkpoint_path = os.path.join(
    base_dir, 'pretrained_models/albert_base/model.ckpt-best')
dict_path = os.path.join(
    base_dir, 'pretrained_models/albert_base/vocab.txt')
#加载punkt句子分割器

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path, checkpoint_path, model='albert')  # 建立模型，加载权重
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

token_ids, segment_ids = tokenizer.encode(['i like china','i love my teacher.'])
token_ids, segment_ids = to_array([token_ids], [segment_ids])

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def extract_feat(imdbid,sub_info):
    file_name = imdbid + '.pkl'
    movie_feat = dict()
    for shot_id,sub in enumerate(sub_info):
        sub_str =  sub['subtitle'].lower()
        # #对句子进行分割
        if sub_str:
            sentences = sen_tokenizer.tokenize(sub_str)
            # 编码
            token_ids, segment_ids = tokenizer.encode(sentences)
            token_ids, segment_ids = to_array([token_ids], [segment_ids])
            outs = model.predict([token_ids, segment_ids])
            outs = outs.squeeze(0)
            sub_feat = normalization(np.sum(outs,axis=0))
        else:
            sub_feat =  np.zeros([768,],dtype=np.float32)
        movie_feat[str(shot_id).zfill(4)] = sub_feat
    if not os.path.exists(osp.join(data_path,'sub_feat')):
        os.mkdir(osp.join(data_path,'sub_feat'))
    with open(data_path+"/sub_feat/{}".format(file_name), 'wb') as fw:
        pickle.dump(movie_feat,fw)
    print('the sub feat movie {} has written'.format(imdbid))
    return movie_feat


if __name__ == '__main__':
 
    with open(data_path+'/info/'+'shot_sub.json', 'r') as load_f:
        load_dict = json.load(load_f)
        for k,v in sorted(load_dict.items()):
            movie_feat = extract_feat(k,v)
