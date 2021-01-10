from data_pre.data_process import Preprocessor,data_preprocess,data_partition


def get_data(cfg):
    imdbid_list_json,annos_dict,annos_valid_dict,casts_dict,acts_dict,subs_dict = data_preprocess(cfg)
    partition = data_partition(cfg,imdbid_list_json,annos_valid_dict)
    data_dict = {
        'annos_dict':annos_dict,
        'casts_dict':casts_dict,
        'acts_dict':acts_dict,
        'subs_dict':subs_dict
    }
    train_set = Preprocessor(cfg, partition['train'], data_dict)
    test_set = Preprocessor(cfg, partition['test'], data_dict)
    val_set = Preprocessor(cfg, partition['val'], data_dict)
    return train_set,test_set,val_set