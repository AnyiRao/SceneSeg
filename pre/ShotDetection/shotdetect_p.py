'''
Parallel detect shot according to a list
'''

from __future__ import print_function

import argparse
import multiprocessing
import os
import os.path as osp
import pdb
from datetime import datetime

from shotdetect.detectors.average_detector import AverageDetector
from shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV
from shotdetect.keyf_img_saver import generate_images, generate_images_txt
from shotdetect.shot_manager import ShotManager
from shotdetect.stats_manager import StatsManager
from shotdetect.video_manager import VideoManager
from shotdetect.video_splitter import split_video_ffmpeg

global parallel_cnt
global parallel_num
parallel_cnt = 0


def main(args, video_path, data_root):
    stats_file_folder_path = osp.join(data_root, "shot_stats")
    os.makedirs(stats_file_folder_path, exist_ok=True)

    video_prefix = video_path.split(".")[0].split("/")[-1]
    stats_file_path = osp.join(stats_file_folder_path, '{}.csv'.format(video_prefix))
    # print(video_path)
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our shotManager and pass it our StatsManager.
    shot_manager = ShotManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    if args.avg_sample:
        shot_manager.add_detector(AverageDetector(shot_length=50))
    else:
        shot_manager.add_detector(ContentDetectorHSVLUV(threshold=20))
    base_timecode = video_manager.get_base_timecode()

    shot_list = []

    try:
        # If stats file exists, load it.
        if osp.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set begin and end time
        if args.begin_time is not None:
            start_time = base_timecode + args.begin_time
            end_time = base_timecode + args.end_time
            video_manager.set_duration(start_time=start_time, end_time=end_time)
        elif args.begin_frame is not None:
            start_frame = base_timecode + args.begin_frame
            end_frame = base_timecode + args.end_frame
            video_manager.set_duration(start_time=start_frame, end_time=end_frame)
            pass
        # Set downscale factor to improve processing speed.
        if args.keep_resolution:
            video_manager.set_downscale_factor(1)
        else:
            video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform shot detection on video_manager.
        shot_manager.detect_shots(frame_source=video_manager)

        # Obtain list of detected shots.
        shot_list = shot_manager.get_shot_list(base_timecode)
        # Each shot is a tuple of (start, end) FrameTimecodes.
        # Save keyf img for each shot
        if args.save_keyf:
            output_dir = osp.join(data_root, "shot_keyf", video_prefix)
            generate_images(video_manager, shot_list, output_dir, num_images=3)

        # Save keyf txt of frame ind
        if args.save_keyf_txt:
            output_dir = osp.join(data_root, "shot_txt", "{}.txt".format(video_prefix))
            os.makedirs(osp.join(data_root, 'shot_txt'), exist_ok=True)
            generate_images_txt(shot_list, output_dir, num_images=5)

        # Split video into shot video
        if args.split_video:
            output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=True)

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)
    finally:
        video_manager.release()

    return shot_list


def call_back(rst):
    global parallel_cnt
    global parallel_num
    parallel_cnt += 1
    if parallel_cnt % 1 == 0:
        print('{}, {:5d} / {:5d} done!'.format(datetime.now(), parallel_cnt, parallel_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parallel ShotDetect")
    parser.add_argument('--num_workers', type=int, default=2, help='number of processors.')
    parser.add_argument('--source_path', type=str,
                        default=osp.join("../../data/video"),
                        help="path to the videos to be processed, please use absolute path")
    parser.add_argument('--list_file', type=str, 
                        default="../../data/meta.txt",
                        help='The list of videos to be processed,\
                        in the form of xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n')
    parser.add_argument('--save_data_root_path', type=str,
                        default="../../data",
                        help="path to the saved data, please use absolute path")
    parser.add_argument('--save_keyf',       action="store_true")
    parser.add_argument('--save_keyf_txt',   action="store_true")
    parser.add_argument('--split_video',     action="store_true")
    parser.add_argument('--keep_resolution', action="store_true")
    parser.add_argument('--avg_sample',      action="store_true")
    parser.add_argument('--begin_time',  type=float, default=None,  help="float: timecode")
    parser.add_argument('--end_time',    type=float, default=120.0, help="float: timecode")
    parser.add_argument('--begin_frame', type=int,   default=None,  help="int: frame")
    parser.add_argument('--end_frame',   type=int,   default=1000,  help="int: frame")
    args = parser.parse_args()

    if args.list_file is None:
        video_list = sorted(os.listdir(args.source_path))
    else:
        video_list = [x.strip() for x in open(args.list_file)]

    parallel_num = len(video_list)
    pool = multiprocessing.Pool(processes=args.num_workers)
    for video_id in video_list:
        video_path = osp.abspath(osp.join(args.source_path, f"{video_id}.mp4"))
        # uncommnet the following line and turn to non-parallel mode if wish to debug
        # main(args, video_path, args.save_data_root_path) 
        pool.apply_async(main, args=(args, video_path, args.save_data_root_path), callback=call_back)
    pool.close()
    pool.join()
