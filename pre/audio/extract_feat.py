import sys
sys.path.append("../")
import argparse
import os
import os.path as osp
from datetime import datetime
import multiprocessing
import numpy as np
import pickle
import pdb
import subprocess
import librosa

global parallel_cnt
global parallel_num
parallel_cnt = 0

def call_back(rst):
    global parallel_cnt
    global parallel_num
    parallel_cnt += 1
    if parallel_cnt % 100 == 0:
        print('{}, {:5d} / {:5d} done!'.format(datetime.now(), parallel_cnt, parallel_num))

def run_mp42wav(args,video_id,shot_id):
    source_movie_fn = osp.join(args.source_video_path,video_id,"{}.mp4".format(shot_id))
    out_video_fn    = osp.join(args.save_wav_path,    video_id,"{}.wav".format(shot_id))
    if not args.replace_old and osp.exists(out_video_fn):
        return 0
    call_list  = ['ffmpeg']
    call_list += ['-v', 'quiet']
    call_list += [
        '-i',
        source_movie_fn,
        '-f',
        'wav']
    call_list += ['-map_chapters', '-1'] #remove meta stream
    call_list += [out_video_fn]
    subprocess.call(call_list)
    if not osp.exists(out_video_fn):
        wav_np = np.zeros((16000*4),np.float32)
        librosa.output.write_wav(out_video_fn,wav_np,sr=16000)
        print(video_id,shot_id,"not exist")

def run_wav2stft(args,video_id,shot_id):
    k = 3  # sample episode num
    time_unit = 3  # unit: second
    feat_path = osp.join(args.save_stft_path, video_id, '{}.npy'.format(shot_id))
    if args.replace_old and osp.exists(feat_path):
        return 0
    data, fs = librosa.core.load(osp.join(args.save_wav_path,video_id,"{}.wav".format(shot_id)), sr=16000)
    # normalize
    mean = (data.max() + data.min()) / 2
    span = (data.max() - data.min()) / 2
    if span < 1e-6:
        span = 1
    data = (data - mean) / span  # range: [-1,1]
    
    D = librosa.core.stft(data, n_fft=512)
    freq = np.abs(D)
    freq = librosa.core.amplitude_to_db(freq)
    
    # tile
    rate = freq.shape[1] / (len(data) / fs)
    thr = int(np.ceil(time_unit * rate / k * (k + 1)))
    copy_ = freq.copy()
    while freq.shape[1]<thr:
        tmp = copy_.copy()
        freq = np.concatenate((freq, tmp), axis=1)

    if freq.shape[1] <=90:
        print(video_id,shot_id,freq.shape)

    # sample
    n = freq.shape[1]
    milestone = [x[0] for x in np.array_split(np.arange(n), k+1)[1:] ]
    span = 15
    stft_img = []
    for i in range(k):
        stft_img.append(freq[:, milestone[i]-span:milestone[i]+span])
    freq = np.concatenate(stft_img, axis=1)
    if freq.shape[1] != 90:
        print(video_id,shot_id,freq.shape)
    np.save(feat_path, freq)

def run(args,video_id,shot_id):
    run_mp42wav(args,video_id,shot_id)
    run_wav2stft(args,video_id,shot_id)


def main(args):
    print(args)
    os.makedirs(args.save_wav_path,exist_ok = True)
    os.makedirs(args.save_stft_path,exist_ok = True)

    if args.list_file is None:
        video_list = sorted(os.listdir(args.source_video_path))
    else:
        video_list = [x.strip() for x in open(args.list_file)] 
    video_list = [i.split(".m")[0] for i in video_list] ## to remove suffix .mp4 .mov etc. if applicable

    # pdb.set_trace()
    global parallel_num
    parallel_num = 0
    for video_id in video_list:
        shot_id_mp4_list = sorted(os.listdir(osp.join(args.source_video_path,video_id)))
        for shot_id_mp4 in shot_id_mp4_list:
            parallel_num +=1
    
    pool = multiprocessing.Pool(processes=args.num_workers) 
    for video_id in video_list:
        shot_id_mp4_list = os.listdir(osp.join(args.source_video_path,video_id))
        os.makedirs(osp.join(args.save_wav_path,video_id),exist_ok = True)
        os.makedirs(osp.join(args.save_stft_path,video_id),exist_ok = True)
        for shot_id_mp4 in shot_id_mp4_list:
            shot_id = shot_id_mp4.split(".m")[0]
            # run(args,video_id,shot_id)
            # pdb.set_trace()
            pool.apply_async(run, (args,video_id,shot_id) , callback=call_back)
    pool.close() 
    pool.join()

if __name__ == '__main__':
    data_root = "data/demo"
    parser = argparse.ArgumentParser("Audio feature using stft")
    parser.add_argument('--replace_old', action="store_true",help='rewrite exisiting wav and feature')
    parser.add_argument('-nw','--num_workers', type=int,default=8,help='number of processors.')
    parser.add_argument('--list_file', type=str, default=osp.join(data_root,'meta/list_test.txt'),
                        help='The list of videos to be processed,\
                        in the form of xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n \
                                     or xxxx0\nxxxx1\nxxxx2\n')
    parser.add_argument('--source_video_path',type=str,default=osp.join(data_root,"shot_split_video"))
    parser.add_argument('--save_wav_path',    type=str,default=osp.join(data_root,"aud_wav"))
    parser.add_argument('--save_stft_path',   type=str,default=osp.join(data_root,"aud_feat"))
    parser.add_argument('--duration_time',type=float,default=0.2)
    args = parser.parse_args()
    main(args)