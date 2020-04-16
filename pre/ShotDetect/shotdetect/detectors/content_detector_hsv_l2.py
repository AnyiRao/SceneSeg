import numpy as np
import cv2
import pdb
from shotdetect.shot_detector import shotDetector


class ContentDetectorHSVL2(shotDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content shots still using HSV information, use the DissolveDetector.
    """

    def __init__(self, threshold=30.0, min_shot_len=15):
        super(ContentDetectorHSVL2, self).__init__()
        self.hsv_threshold = threshold
        self.delta_hsv_gap_threshold = 10
        self.rgb_threshold = 50
        self.hsv_weight = 3
        self.min_shot_len = min_shot_len  # minimum length of any given shot, in frames
        self.last_frame = None
        self.last_shot_cut = None
        self.last_hsv = None
        self._metric_keys = ['hsv_content_val', 'delta_hsv_hue', 'delta_hsv_sat', 'delta_hsv_lum','rgb_content_val', 'delta_rgb_hue', 'delta_rgb_sat', 'delta_rgb_lum']
        self.cli_name = 'detect-content'
        self.last_rgb = None

    def process_frame(self, frame_num, frame_img):
        # type: (int, np.ndarray) -> List[int]
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
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

        if self.last_frame is not None:
            # Change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
            delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v = 0.0, 0.0, 0.0, 0.0
            delta_rgb_avg, delta_rgb_h, delta_rgb_s, delta_rgb_v = 0.0, 0.0, 0.0, 0.0
            
            if (self.stats_manager is not None and
                    self.stats_manager.metrics_exist(frame_num, metric_keys)):
                delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v, delta_rgb_avg, delta_rgb_h, delta_rgb_s, delta_rgb_v = self.stats_manager.get_metrics(
                    frame_num, metric_keys)

            else:
                num_pixels = frame_img.shape[0] * frame_img.shape[1]
                curr_rgb = cv2.split(frame_img)
                curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
                last_hsv = self.last_hsv
                last_rgb = self.last_rgb
                if not last_hsv:
                    last_hsv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))
                    last_rgb = cv2.split(self.last_frame)

                delta_hsv = [0, 0, 0, 0]
                for i in range(3):
                    num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                    curr_hsv[i] = curr_hsv[i].astype(np.int32)
                    last_hsv[i] = last_hsv[i].astype(np.int32)
                    delta_hsv[i] = np.sum(
                        np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
                delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
                delta_hsv_h, delta_hsv_s, delta_hsv_v, delta_hsv_avg = delta_hsv

                delta_rgb =  [0, 0, 0, 0]
                for i in range(3):
                    num_pixels = curr_rgb[i].shape[0] * curr_rgb[i].shape[1]
                    curr_rgb[i] = curr_rgb[i].astype(np.int32)
                    last_rgb[i] = last_rgb[i].astype(np.int32)
                    delta_rgb[i] = np.sum(
                        np.abs(curr_rgb[i] - last_rgb[i])) / float(num_pixels)
                delta_rgb[3] = sum(delta_rgb[0:3]) / 3.0
                delta_rgb_h, delta_rgb_s, delta_rgb_v, delta_rgb_avg = delta_rgb

                if self.stats_manager is not None:
                    self.stats_manager.set_metrics(frame_num, {
                        metric_keys[0]: delta_hsv_avg,
                        metric_keys[1]: delta_hsv_h,
                        metric_keys[2]: delta_hsv_s,
                        metric_keys[3]: delta_hsv_v,
                        metric_keys[0+4]: delta_rgb_avg,
                        metric_keys[1+4]: delta_rgb_h,
                        metric_keys[2+4]: delta_rgb_s,
                        metric_keys[3+4]: delta_rgb_v,
                        })

                self.last_hsv = curr_hsv
                self.last_rgb = curr_rgb
            # pdb.set_trace()
            if delta_hsv_avg >= self.hsv_threshold and delta_hsv_avg - self.hsv_threshold >= self.delta_hsv_gap_threshold:
                print(frame_num,delta_hsv_avg,delta_rgb_avg)
                if self.last_shot_cut is None or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num
            elif delta_hsv_avg >= self.hsv_threshold and delta_hsv_avg - self.hsv_threshold < self.delta_hsv_gap_threshold \
                and delta_rgb_avg + self.hsv_weight * (delta_hsv_avg - self.hsv_threshold) > self.rgb_threshold:
                print(frame_num,delta_hsv_avg,delta_rgb_avg)
                if self.last_shot_cut is None or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num    

            if self.last_frame is not None and self.last_frame is not _unused:
                del self.last_frame

        # If we have the next frame computed, don't copy the current frame
        # into last_frame since we won't use it on the next call anyways.
        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num+1, metric_keys)):
            self.last_frame = _unused
        else:
            self.last_frame = frame_img.copy()
        # if len(cut_list) > 0:
        #     print(frame_num,cut_list)
        return cut_list


    #def post_process(self, frame_num):
    #    """ Not used for ContentDetector, as unlike ThresholdDetector, cuts
    #    are always written as they are found.
    #    """
    #    return []

