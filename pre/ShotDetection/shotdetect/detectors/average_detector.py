import pdb

import cv2
import numpy as np

from shotdetect.shot_detector import shotDetector


class AverageDetector(shotDetector):
    """Average sampling around
    noraml [-1, 1] * 0.2 * self.shot_length_fix + self.shot_length_fix
    """

    def __init__(self, shot_length=50, random=False):
        super(AverageDetector, self).__init__()
        self.random = random
        self.random_ratio = 0.2
        self.shot_length_fix = shot_length

        if random:
            self.shot_length = int(self.shot_length_fix + (np.random.rand() - 0.5) * 2 * self.random_ratio * self.shot_length_fix)
        else:
            self.shot_length = shot_length

        # self.min_shot_len = min_shot_len  # minimum length of any given shot, in frames
        self.last_frame = None
        self.last_frame_num = 0
        self.last_shot_cut = 0
        self._metric_keys = ['frame_num']

    def process_frame(self, frame_num, frame_img):
        """
        Args:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (np.ndarray) to perform shot
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base shotDetector class) returns True.

        Returns:
            List[int]: List of frames where shot cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        cut_list = []
        metric_keys = self._metric_keys
        _unused = ''
        # print(frame_num, cut_list, frame_num / self.shot_length == 0)
        # pdb.set_trace()
        if self.last_frame is not None:
            self.stats_manager.set_metrics(frame_num, {
                        metric_keys[0]: frame_num,
                        })

            # if self.last_shot_cut is None or (
            #         (frame_num - self.last_shot_cut) >= self.min_shot_len):
            if frame_num - self.last_shot_cut >= self.shot_length and frame_num != 0:
                cut_list.append(frame_num)
                self.last_shot_cut = frame_num
                if self.random:
                    self.shot_length = int(self.shot_length_fix + (np.random.rand() - 0.5) * 2 * self.random_ratio * self.shot_length_fix)
                else:
                    self.shot_length = self.shot_length_fix
                assert self.shot_length > 0
        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num+1, metric_keys)):
            self.last_frame = _unused
        else:
            self.last_frame = frame_img.copy()
        # if len(cut_list) > 0:
        #     print(frame_num, cut_list)
        #     pdb.set_trace()
        return cut_list
