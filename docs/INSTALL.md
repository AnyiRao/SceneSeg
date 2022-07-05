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
Pretrained models are at [Google Drive](https://drive.google.com/open?id=1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz).
Please follow the provided folder structure and put it under ``run``.

### Prepare Datasets for Scene318
Annotation and metadata are at [Google Drive](https://drive.google.com/open?id=1F-uqCKnhtSdQKcDUiL3dRcLOrAxHargz). The full movienet annotation including, debut year, cast, director info and so on can be downloaded from [OpenDataLab](https://opendatalab.com/MovieNet/download)

The intermediate features ``place_feat, cast_feat, act_feat, aud_feat`` are located at [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155113941_link_cuhk_edu_hk/Eu52Dx-I5M5KhAQKXnwg2oQBV4icIe_zpziOQkqAo1_5XA?e=9hYk28)
 due to the limited free size of Google Drive. 
They can be extracted with unzip e.g., ``unzip act_feat.zip``. 

Put ``label318`` ``shot_movie318`` ``meta/split318.json``, 
intermediate features ``place_feat, cast_feat, act_feat, aud_feat`` under ``data``. 

#### Explanation
```label318``` is the scene transit (**1**) or not (**0**) label for each shot. ```shot_movie318``` is the shot and frame correspondence to recover the time of each scene. They will be automatically 
handled by the processing codes e.g., the ``data_pre`` function in ``src/data/all.py``
### Prepare Your Own Dataset
If you run our demo, you don't need to prepare. Just run our [three lines](./GETTING_STARTED.md#demo).

If you wish to use the method's full functions, you may need to organize your data like ours.
