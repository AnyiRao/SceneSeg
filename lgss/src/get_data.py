import sys
sys.path.append("..")
sys.path.append("../..")
from utilis import to_numpy
from utilis.package import *


def get_data(cfg):
    if cfg.dataset.name == "all":
        from .data.all import Preprocessor, data_pre, data_partition
        imdbidlist_json, annos_dict, annos_valid_dict, casts_dict, acts_dict = data_pre(cfg)
        partition = data_partition(cfg, imdbidlist_json, annos_valid_dict)
        data_dict = {"annos_dict": annos_dict, "casts_dict":casts_dict,\
                    "acts_dict":acts_dict}
        trainSet = Preprocessor(cfg, partition['train'], data_dict)
        testSet = Preprocessor(cfg, partition['test'], data_dict)
        valSet = Preprocessor(cfg, partition['val'], data_dict)
        return trainSet, testSet, valSet

    elif cfg.dataset.name == "image":
        from .data.image import Preprocessor, data_pre, data_partition
        imdbidlist_json, annos_dict, annos_valid_dict = data_pre(cfg)
        partition = data_partition(cfg, imdbidlist_json, annos_valid_dict)
        data_dict = {"annos_dict": annos_dict}
        trainSet = Preprocessor(cfg, partition['train'], data_dict)
        testSet = Preprocessor(cfg, partition['test'], data_dict)
        valSet = Preprocessor(cfg, partition['val'], data_dict)
        return trainSet, testSet, valSet

    elif cfg.dataset.name == "demo":
        from .data.demo import Preprocessor, data_pre, data_partition
        valid_shotids = data_pre(cfg)
        partition = data_partition(cfg, valid_shotids)
        trainSet = Preprocessor(cfg, partition['train'])
        testSet = Preprocessor(cfg, partition['test'])
        valSet = Preprocessor(cfg, partition['val'])
        return trainSet, testSet, valSet

    elif cfg.dataset.name == "custom":
        '''
        specify custom preprocessor
        '''
        pass
