# Shot Detection

![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python) ![mmcv](https://img.shields.io/badge/mmcv-%3E%3D0.4.0-green)

## Easy-to-use
```
cd ShotDetect
python shotdetect.py # to process a single video
python shotdetect_p.py # to process a list of videos in parallel
```

## Introduction
Shot detection from videos
with useful portals for long complicated videos, e.g., [movies](http://movienet.site/) scenarios.
The repo is based on [PySceneDetect](py.scenedetect.com), which is under BSD 3-Clause License.

## Features
- Parallel processing.
- Keyframe saving.
- Optimal detector that is tested on movie/tv epsoides scenarios, e.g., HSV-LUV joint model.
- Average sampler.

## Citation
```
@inproceedings{rao2020local,
title={A Local-to-Global Approach to Multi-modal Movie Scene Segmentation},
author={Rao, Anyi and Xu, Linning and Xiong, Yu and Xu, Guodong and Huang, Qingqiu and Zhou, Bolei and Lin, Dahua},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2020}
}

@misc{brandon2018,
author = {Brandon Castellano},
title = {PySceneDetect: Intelligent scene cut detection and video splitting tool},
year = 2018,
howpublished = {\url{https://pyscenedetect.readthedocs.io/en/latest/}},
}
```