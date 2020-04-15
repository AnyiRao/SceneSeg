from . import mkdir_ifmiss,strcal,get_pair_list,getRatio
from .package import *

def get_result_dict(metric_dict):
     with open(metric_dict['pred_path'],"r") as f:
          content = f.read().splitlines()
     result_dict = {}
     for item in content:
          imdbid = item.split(" ")[0]
          shotid = item.split(" ")[1]
          gt   = item.split(" ")[2]
          pred = item.split(" ")[3]
          if result_dict.get(imdbid) is None:
               result_dict[imdbid] = {}
          result_dict[imdbid][shotid] = {'pred':pred,'gt':gt}
     return result_dict

def get_scene_list(pair_list,shot_list):
     scene_list = []
     if pair_list is None:
          return None
     for item in pair_list:
          start = int(shot_list[int(item[0])].split(" ")[0])
          end   = int(shot_list[int(item[-1])].split(" ")[1])
          scene_list.append((start,end))
     return scene_list

def cal_miou(gt_scene_list,pred_scene_list):
     mious = []
     for gt_scene_item in gt_scene_list:
          rats = []
          for pred_scene_item in pred_scene_list:
               rat = getRatio(pred_scene_item,gt_scene_item)
               rats.append(rat)
          mious.append(np.max(rats))  
     miou = np.mean(mious) 
     return miou

def cal_MIOU(cfg,threshold=0.5):
     metric_dict = {
          "pred_path":osp.join(cfg.logger.logs_dir,'pred/pred_{:.2f}.txt'.format(threshold)),
          "shot_path":cfg.shot_frm_path
     }
     result_dict = get_result_dict(metric_dict)
     Mious = []
     for imdbid, result_dict_one in result_dict.items():
          shot_fn = "{}/{}.txt".format(metric_dict.get('shot_path'),imdbid)
          with open (shot_fn,"r") as f:
               shot_list = f.read().splitlines()

          gt_dict_one,pred_dict_one = {}, {}
          for shotid, item in result_dict_one.items():
               gt_dict_one.update({shotid:int(item.get('gt'))})
               pred_dict_one.update({shotid:int(item.get('pred'))})
          gt_pair_list = get_pair_list(gt_dict_one)
          pred_pair_list = get_pair_list(pred_dict_one)
          if pred_pair_list is None:
               Mious.append(0)
               continue
          gt_scene_list = get_scene_list(gt_pair_list,shot_list)
          pred_scene_list = get_scene_list(pred_pair_list,shot_list)
          if gt_scene_list is None or pred_scene_list is None:
               return None
          miou1 = cal_miou(gt_scene_list,pred_scene_list)
          miou2 = cal_miou(pred_scene_list,gt_scene_list)   
          Mious.append(np.mean([miou1,miou2]))
     print("Miou: ",np.mean(Mious))
     return np.mean(Mious)
