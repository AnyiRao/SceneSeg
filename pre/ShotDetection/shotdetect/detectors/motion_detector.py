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
import cv2
import numpy


from shotdetect.shot_detector import shotDetector

class MotionDetector(shotDetector):
    """Detects motion events in shots containing a static background.

    Uses background subtraction followed by noise removal (via morphological
    opening) to generate a frame score compared against the set threshold.

    Attributes:
        threshold:  floating point value compared to each frame's score, which
            represents average intensity change per pixel (lower values are
            more sensitive to motion changes).  Default 0.5, must be > 0.0.
        num_frames_post_shot:  Number of frames to include in each motion
            event after the frame score falls below the threshold, adding any
            subsequent motion events to the same shot.
        kernel_size:  Size of morphological opening kernel for noise removal.
            Setting to -1 (default) will auto-compute based on video resolution
            (typically 3 for SD, 5-7 for HD). Must be an odd integer > 1.
    """
    def __init__(self, threshold = 0.50, num_frames_post_shot = 30,
                 kernel_size = -1):
        """Initializes motion-based shot detector object."""
        # Requires porting to v0.5 API.
        raise NotImplementedError()

        self.threshold = float(threshold)
        self.num_frames_post_shot = int(num_frames_post_shot)

        self.kernel_size = int(kernel_size)
        if self.kernel_size < 0:
            # Set kernel size when process_frame first runs based on
            # video resolution (480p = 3x3, 720p = 5x5, 1080p = 7x7).
            pass

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2( 
            detectShadows = False )

        self.last_frame_score = 0.0

        self.in_motion_event = False
        self.first_motion_frame_index = -1
        self.last_motion_frame_index = -1
        self.cli_name = 'detect-motion'
        return

    def process_frame(self, frame_num, frame_img, frame_metrics, shot_list):

        # Value to return indiciating if a shot cut was found or not.
        cut_detected = False

        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_frame = self.bg_subtractor.apply(frame_grayscale)

        kernel = numpy.ones((self.kernel_size, self.kernel_size), numpy.uint8)
        filtered_frame = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        frame_score = numpy.sum(filtered_frame) / float( 
            filtered_frame.shape[0] * filtered_frame.shape[1] )

        return cut_detected

    def post_process(self, shot_list, frame_num):
        """Writes the last shot if the video ends while in a motion event.
        """

        # If the last fade detected was a fade out, we add a corresponding new
        # shot break to indicate the end of the shot.  This is only done for
        # fade-outs, as a shot cut is already added when a fade-in is found.

        if self.in_motion_event:
            # Write new shot based on first and last motion event frames.
            pass
        return self.in_motion_event


