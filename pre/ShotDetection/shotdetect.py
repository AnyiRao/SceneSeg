from __future__ import print_function

import argparse
import os
import os.path as osp

from shotdetect.detectors.average_detector import AverageDetector
from shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV
from shotdetect.keyf_img_saver import generate_images, generate_images_txt
from shotdetect.shot_manager import ShotManager
from shotdetect.stats_manager import StatsManager
from shotdetect.video_manager import VideoManager
from shotdetect.video_splitter import split_video_ffmpeg


def main(args, data_root):
    video_path = osp.abspath(args.video_path)
    video_prefix = video_path.split(".")[0].split("/")[-1]
    stats_file_folder_path = osp.join(data_root, "shot_stats")
    os.makedirs(stats_file_folder_path, exist_ok=True)

    stats_file_path = osp.join(stats_file_folder_path, '{}.csv'.format(video_prefix))
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
        if args.print_result:
            print('List of shots obtained:')
            for i, shot in enumerate(shot_list):
                print(
                    'Shot %4d: Start %s / Frame %d, End %s / Frame %d' % (
                        i,
                        shot[0].get_timecode(), shot[0].get_frames(),
                        shot[1].get_timecode(), shot[1].get_frames(),))
        # Save keyf img for each shot
        if args.save_keyf:
            output_dir = osp.join(data_root, "shot_keyf", video_prefix)
            generate_images(video_manager, shot_list, output_dir, num_images=3)

        # Save keyf txt of frame ind
        if args.save_keyf_txt:
            output_dir = osp.join(data_root, "shot_txt", "{}.txt".format(video_prefix))
            os.makedirs(osp.join(data_root, 'shot_txt'), exist_ok=True)
            generate_images_txt(shot_list, output_dir)

        # Split video into shot video
        if args.split_video:
            output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=False)

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)
    finally:
        video_manager.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_path', type=str,
                        default=osp.join("../data/demo", "video/demo.mp4"),
                        help="path to the video to be processed")
    parser.add_argument('--save_data_root_path', type=str,
                        default="../data/demo",
                        help="path to the saved data")
    parser.add_argument('--print_result',    action="store_true")
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
    main(args, args.save_data_root_path)
