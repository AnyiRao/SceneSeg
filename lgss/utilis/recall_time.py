from utilis import mkdir_ifmiss,strcal
from utilis.package import *
from sklearn.metrics import recall_score

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

def cal_Recall_time(cfg,recall_time=3,threshold=0.3):
     metric_dict = {
          "pred_path":osp.join(cfg.logger.logs_dir,'pred/pred_{:.2f}.txt'.format(threshold)),
          "shot_path":cfg.shot_frm_path,
          "recall_duration":5,
          "recall_time":recall_time
     }

     result_dict = get_result_dict(metric_dict)
     recall = []
     for imdbid, result_dict_one in result_dict.items():
          shot_fn = "{}/{}.txt".format(metric_dict.get('shot_path'),imdbid)
          with open (shot_fn,"r") as f:
               shot_list = f.read().splitlines()

          cont_one, total_one = 0, 0
          for shotid, item in result_dict_one.items():
               gt = item.get('gt')
               shot_time = int(shot_list[int(shotid)].split(' ')[1])
               if gt != '1':
                    continue
               total_one +=1
               for ind in range(0-metric_dict.get('recall_duration'),1+metric_dict.get('recall_duration')):
                    shotid_cp = strcal(shotid,ind)
                    if int(shotid_cp) < 0 or int(shotid_cp) >= len(shot_list):
                         continue
                    shot_time_cp = int(shot_list[int(shotid_cp)].split(' ')[1])
                    item_cp = result_dict_one.get(shotid_cp)
                    if item_cp is None:
                         continue
                    else:
                         pred = item_cp.get('pred')
                         gap_time = np.abs(shot_time_cp-shot_time)/24
                         if gt == pred and gap_time < metric_dict.get('recall_time'):
                              cont_one +=1
                              break
          recall_one = cont_one/(total_one+1e-5)
          recall.append(recall_one)
     print('Recall_at_{}: '.format(metric_dict.get('recall_time')),np.mean(recall))
     return np.mean(recall)
     
def cal_Recall(cfg,threshold=0.3):
     metric_dict = {
          "pred_path":osp.join(cfg.logger.logs_dir,'pred/pred_{:.2f}.txt'.format(threshold)),
          "shot_path":cfg.shot_frm_path
     }
     result_dict = get_result_dict(metric_dict)
     recall = []
     for imdbid, result_dict_one in result_dict.items():
          preds,gts = [],[]
          for shotid, item in result_dict_one.items():
               pred = int(item.get('pred'))
               gt   = int(item.get('gt'))
               preds.append(pred)
               gts.append(gt)
          recall_one = recall_score(gts, preds, average='binary')  
          recall.append(recall_one)
     print('Recall: ',np.mean(recall))
     return np.mean(recall)
