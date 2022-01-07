# The codes below partially refer to the PySceneDetect. According
# to its BSD 3-Clause License, we keep the following.
#
#          PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2021 Brandon Castellano <http://www.bcastell.com>.

import numpy

from shotdetect.shot_detector import shotDetector


def compute_frame_average(frame):
    """Computes the average pixel value/intensity for all pixels in a frame.

    The value is computed by adding up the 8-bit R, G, and B values for
    each pixel, and dividing by the number of pixels multiplied by 3.

    Returns:
        Floating point value representing average pixel intensity.
    """
    num_pixel_values = float(
        frame.shape[0] * frame.shape[1] * frame.shape[2])
    avg_pixel_value = numpy.sum(frame[:, :, :]) / num_pixel_values
    return avg_pixel_value


class ThresholdDetector(shotDetector):
    """Detects fast cuts/slow fades in from and out to a given threshold level.

    Detects both fast cuts and slow fades so long as an appropriate threshold
    is chosen (especially taking into account the minimum grey/black level).

    Attributes:
        threshold:  8-bit intensity value that each pixel value (R, G, and B)
            must be <= to in order to trigger a fade in/out.
        min_percent:  Float between 0.0 and 1.0 which represents the minimum
            percent of pixels in a frame that must meet the threshold value in
            order to trigger a fade in/out.
        min_shot_len:  Unsigned integer greater than 0 representing the
            minimum length, in frames, of a shot (or subsequent shot cut).
        fade_bias:  Float between -1.0 and +1.0 representing the percentage of
            timecode skew for the start of a shot (-1.0 causing a cut at the
            fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be
            right at the position where the threshold is passed).
        add_final_shot:  Boolean indicating if the video ends on a fade-out to
            generate an additional shot at this timecode.
        block_size:  Number of rows in the image to sum per iteration (can be
            tuned to increase performance in some cases; should be computed
            programmatically in the future).
    """
    def __init__(self, threshold=12, min_percent=0.95, min_shot_len=15,
                 fade_bias=0.0, add_final_shot=False, block_size=8):
        """Initializes threshold-based shot detector object."""

        super(ThresholdDetector, self).__init__()
        self.threshold = int(threshold)
        self.fade_bias = fade_bias
        self.min_percent = min_percent
        self.min_shot_len = min_shot_len
        self.last_frame_avg = None
        self.last_shot_cut = None
        # Whether to add an additional shot or not when ending on a fade out
        # (as cuts are only added on fade ins; see post_process() for details).
        self.add_final_shot = add_final_shot
        # Where the last fade (threshold crossing) was detected.
        self.last_fade = {
            'frame': 0,         # frame number where the last detected fade is
            'type': None        # type of fade, can be either 'in' or 'out'
        }
        self.block_size = block_size
        self._metric_keys = ['delta_rgb']
        self.cli_name = 'detect-threshold'

    def frame_under_threshold(self, frame):
        """Check if the frame is below (true) or above (false) the threshold.

        Instead of using the average, we check all pixel values (R, G, and B)
        meet the given threshold (within the minimum percent).  This ensures
        that the threshold is not exceeded while maintaining some tolerance for
        compression and noise.

        This is the algorithm used for absolute mode of the threshold detector.

        Returns:
            Boolean, True if the number of pixels whose R, G, and B values are
            all <= the threshold is within min_percent pixels, or False if not.
        """
        # First we compute the minimum number of pixels that need to meet the
        # threshold. Internally, we check for values greater than the threshold
        # as it's more likely that a given frame contains actual content. This
        # is done in blocks of rows, so in many cases we only have to check a
        # small portion of the frame instead of inspecting every single pixel.
        num_pixel_values = float(frame.shape[0] * frame.shape[1] * frame.shape[2])
        min_pixels = int(num_pixel_values * (1.0 - self.min_percent))

        curr_frame_amt = 0
        curr_frame_row = 0

        while curr_frame_row < frame.shape[0]:
            # Add and total the number of individual pixel values (R, G, and B)
            # in the current row block that exceed the threshold.
            curr_frame_amt += int(numpy.sum(
                frame[curr_frame_row : curr_frame_row + self.block_size, :, :] > self.threshold))
            # If we've already exceeded the most pixels allowed to be above the
            # threshold, we can skip processing the rest of the pixels.
            if curr_frame_amt > min_pixels:
                return False
            curr_frame_row += self.block_size
        return True

    def process_frame(self, frame_num, frame_img):
        # type: (int, Optional[numpy.ndarray]) -> List[int]
        """
        Args:
            frame_num (int): Frame number of frame that is being passed.
            frame_img (numpy.ndarray or None): Decoded frame image (numpy.ndarray) to perform
                shot detection with. Can be None *only* if the self.is_processing_required()
                method (inhereted from the base shotDetector class) returns True.
        Returns:
            List[int]: List of frames where shot cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        # Compare the # of pixels under threshold in current_frame & last_frame.
        # If absolute value of pixel intensity delta is above the threshold,
        # then we trigger a new shot cut/break.

        # List of cuts to return.
        cut_list = []

        # The metric used here to detect shot breaks is the percent of pixels
        # less than or equal to the threshold; however, since this differs on
        # user-supplied values, we supply the average pixel intensity as this
        # frame metric instead (to assist with manually selecting a threshold)
        frame_avg = 0.0

        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num, self._metric_keys)):
            frame_avg = self.stats_manager.get_metrics(frame_num, self._metric_keys)[0]
        else:
            frame_avg = compute_frame_average(frame_img)
            if self.stats_manager is not None:
                self.stats_manager.set_metrics(frame_num, {
                    self._metric_keys[0]: frame_avg})

        if self.last_frame_avg is not None:
            if self.last_fade['type'] == 'in' and self.frame_under_threshold(frame_img):
                # Just faded out of a shot, wait for next fade in.
                self.last_fade['type'] = 'out'
                self.last_fade['frame'] = frame_num
            elif self.last_fade['type'] == 'out' and not self.frame_under_threshold(frame_img):
                # Just faded into a new shot, compute timecode for the shot
                # split based on the fade bias.
                f_in = frame_num
                f_out = self.last_fade['frame']
                f_split = int((f_in + f_out + int(self.fade_bias * (f_in - f_out))) / 2)
                # Only add the shot if min_shot_len frames have passed.
                if self.last_shot_cut is None or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len):
                    cut_list.append(f_split)
                    self.last_shot_cut = frame_num
                self.last_fade['type'] = 'in'
                self.last_fade['frame'] = frame_num
        else:
            self.last_fade['frame'] = 0
            if self.frame_under_threshold(frame_img):
                self.last_fade['type'] = 'out'
            else:
                self.last_fade['type'] = 'in'
        # Before returning, we keep track of the last frame average (can also
        # be used to compute fades independently of the last fade type).
        self.last_frame_avg = frame_avg
        return cut_list

    def post_process(self, frame_num):
        """Writes a final shot cut if the last detected fade was a fade-out.

        Only writes the shot cut if add_final_shot is true, and the last fade
        that was detected was a fade-out.  There is no bias applied to this cut
        (since there is no corresponding fade-in) so it will be located at the
        exact frame where the fade-out crossed the detection threshold.
        """

        # If the last fade detected was a fade out, we add a corresponding new
        # shot break to indicate the end of the shot.  This is only done for
        # fade-outs, as a shot cut is already added when a fade-in is found.
        cut_times = []
        if self.last_fade['type'] == 'out' and self.add_final_shot and (
                self.last_shot_cut is None or
                (frame_num - self.last_shot_cut) >= self.min_shot_len):
            cut_times.append(self.last_fade['frame'])
        return cut_times
