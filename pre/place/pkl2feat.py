import sys
sys.path.append("../")
from utilis.package import *
from utilis import mkdir_ifmiss,read_pkl,to_numpy
import multiprocessing
import datetime

def run(args,video_id):
     feat_pkl = read_pkl(osp.join(args.source_path,video_id,"feat.pkl"))
     for key, item in feat_pkl.items():
          item = to_numpy(item)
          save_fn = osp.join(args.dest_path,video_id,"{}.npy".format(key.split(".jpg")[0]))
          np.save(save_fn,item)

if __name__ == '__main__':
     data_root = "../../data/scene318"
     parser = argparse.ArgumentParser()
     parser.add_argument('-nw','--num_workers', type=int,default=10,help='number of processors.')
     parser.add_argument('-dp','--dest_path', type=str,  default=osp.join(data_root,"npy_feat3"))
     parser.add_argument('-sp','--source_path', type=str,default=osp.join(data_root,"img_feat3"))
     args = parser.parse_args()

     video_list = os.listdir(args.source_path)
     # for video_id in video_list:
     #      print(video_id)
     #      mkdir_ifmiss(osp.join(args.dest_path,video_id))
     #      run(args,video_id)
     startTime = datetime.datetime.now() 
     pool = multiprocessing.Pool(processes=args.num_workers) 
     for video_id in video_list:
          mkdir_ifmiss(osp.join(args.dest_path,video_id))
          # if (osp.exists(osp.join(args.dest_path,video_id))):
          #      # print ("Already: ",imdbid)
          #      continue
          #run(args,video_id)
          #pdb.set_trace()
          pool.apply_async(run, (args,video_id,) )
     pool.close() 
     pool.join() 
     print(datetime.datetime.now() - startTime)
