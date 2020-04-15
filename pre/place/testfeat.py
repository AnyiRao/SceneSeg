import sys
sys.path.append("../")
from utilis import mkdir_ifmiss,read_txt_list,read_json
from utilis.package import *
import mmcv
import cv2
import librosa
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import subprocess

if __name__=='__main__':
    data_root = "../../data/scene318"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int,default=8,help='number of processors.')
    parser.add_argument('--source_video_path',type=str,default=osp.join(data_root,"shot_video"))
    parser.add_argument('--source_wav_path', type=str,default=osp.join(data_root, "audio_wav"))
    parser.add_argument('--dest_stft_path',  type=str,default=osp.join(data_root, "stft_feat"))
    parser.add_argument('--dest_map_path',   type=str,default=osp.join(data_root, "stft_map"))
    parser.add_argument('--duration_time',type=float,default=0.2)
    args = parser.parse_args()


    npy1 = np.load(osp.join("/DATA/scene/data/npy_feat/tt1340800","shot_1008_img_0.npy"))
    print(npy1.shape)
    npy2 = np.load(osp.join("/DATA/scene/movienet/data/scene318/npy_feat3_176/tt0116282","shot_0008_img_0.npy"))
    print(npy2.shape)
    pdb.set_trace()
