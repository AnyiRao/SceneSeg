import numpy
import cv2
import pdb
from shotdetect.shot_detector import shotDetector


class ContentDetector(shotDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content shots still using HSV information, use the DissolveDetector.
    """

    def __init__(self, threshold=32.0, min_shot_len=15):
        super(ContentDetector, self).__init__()
        self.threshold = threshold
        self.min_shot_len = min_shot_len  # minimum length of any given shot, in frames
        self.last_frame = None
        self.last_shot_cut = None
        self.last_hsv = None
        self._metric_keys = ['content_val', 'delta_hue', 'delta_sat', 'delta_lum']
        self.cli_name = 'detect-content'


    def process_frame(self, frame_num, frame_img):
        # type: (int, numpy.ndarray) -> List[int]
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (numpy.ndarray) to perform shot
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
            delta_hsv_avg, delta_h, delta_s, delta_v = 0.0, 0.0, 0.0, 0.0

            if (self.stats_manager is not None and
                    self.stats_manager.metrics_exist(frame_num, metric_keys)):
                delta_hsv_avg, delta_h, delta_s, delta_v = self.stats_manager.get_metrics(
                    frame_num, metric_keys)

            else:
                num_pixels = frame_img.shape[0] * frame_img.shape[1]
                curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
                last_hsv = self.last_hsv
                if not last_hsv:
                    last_hsv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))

                delta_hsv = [0, 0, 0, 0]
                for i in range(3):
                    num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                    curr_hsv[i] = curr_hsv[i].astype(numpy.int32)
                    last_hsv[i] = last_hsv[i].astype(numpy.int32)
                    delta_hsv[i] = numpy.sum(
                        numpy.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
                delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
                delta_h, delta_s, delta_v, delta_hsv_avg = delta_hsv

                if self.stats_manager is not None:
                    self.stats_manager.set_metrics(frame_num, {
                        metric_keys[0]: delta_hsv_avg,
                        metric_keys[1]: delta_h,
                        metric_keys[2]: delta_s,
                        metric_keys[3]: delta_v})

                self.last_hsv = curr_hsv
            # pdb.set_trace()
            if delta_hsv_avg >= self.threshold:
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

