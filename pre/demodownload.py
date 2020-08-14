import os
import os.path as osp
import shutil

from pytube import YouTube


def main():
    os.makedirs("../data/demo", exist_ok = True)
    os.makedirs("../data/demo/video", exist_ok = True)
    video_save_path = "../data/demo/video"
    yt = YouTube('https://www.youtube.com/watch?v=rT22nYLaVbo')
    yt.streams.get_highest_resolution().download(video_save_path)
    shutil.move(osp.join(video_save_path,os.listdir(video_save_path)[0]),osp.join(video_save_path,"demo.mp4"))


if __name__ == '__main__':
    main()
