from pytube import YouTube
import shutil
import os.path as osp
import os
def main():
    os.makedirs("../data/demo", exist_ok = True)
    os.makedirs("../data/demo/video", exist_ok = True)
    video_save_path = "../data/demo/video"
    yt = YouTube('https://www.youtube.com/watch?v=BaqBhLkDuto')
    yt.streams.get_highest_resolution().download(video_save_path)
    shutil.move(osp.join(video_save_path,"{}.mp4".format(yt.title)),osp.join(video_save_path,"demo.mp4"))
    # yt = YouTube('http://youtube.com/watch?v=9bZkp7q19f0')
    # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download()
if __name__ == '__main__':
    main()
