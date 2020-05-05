## Installation and Preparation

![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.0.0-Orange?logo=pytorch) ![mmcv](https://img.shields.io/badge/mmcv-%3E%3D0.4.0-green)


### Requirements
SceneSeg is currently very easy to install before the introduction of **feature extractor**

- Python 3.6+
- PyTorch 1.0 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

a. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```sh
conda install pytorch torchvision -c pytorch
```

b. Clone the SceneSeg repository.

```sh
git clone https://github.com/AnyiRao/SceneSeg.git
cd SceneSeg
```
c. Install Python Packages

```sh
pip install -r docs/requirements.txt
```

### Folder Structure
```sh
|-data ## the data_root for experiments
|-run  ## to store experiments
|-pre  ## for preprocess
|-lgss
|   |-src
|   |-config
|   |-utilis
|   |-run.py
|   |-gen_csv.py
```
### Download Pretrained Models
Pretrained models are at [Google Drive](https://drive.google.com/open?id=1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz). Part of the models are currently supported. We will also be very likely to put at OneDrive or BaiduNetDisk.
Please follow the provided folder structure and put it under ``run``.

### Prepare Datasets for Scene318
Dataset are at [Google Drive](https://drive.google.com/open?id=1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz). Some of them are currently supported.
Put ``label318`` ``shot_movie318`` ``meta/split318.json``, 
intermediate features ``place_feat cast_feat act_feat aud_feat`` under ``data``

```label318``` is the scene transit (1) or not (0) label for each shot. ```shot_movie318``` is the shot and frame correspondence to recover the time of each scene.

### Prepare Your Own Dataset
If you run our demo, you don't need to prepare. Just run our [three lines](./GETTING_STARTED.md#demo)

If you wish to use the full functions of the method, you may need to organize your data as ours.