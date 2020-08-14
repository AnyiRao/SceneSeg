## Getting Started
### Running

```sh
cd lgss
python run.py config/xxx.py 
```

```run.py``` is the main process, which is encapsulated and controlled by ```config``` files.
Every running process creates a run folder containing experiments logs. 
And we prepare a script ```gen_csv.py``` to automaticly generate a csv file with desired setting and results records.

#### Test with pretrained models
Please set up the config file in ```run``` folders and set ``trainFlag`` as ``False`` and ``testFlag`` as ``True``.

```sh
python run.py ../run/xxx/xxx.py 
 ```

This will automatically load the best model in the ``experiment_name`` folder.

If you wish to use a specific model, please use ``resume`` to include the model path.

#### Use different semantic elements
Modify the config file set dataset name as ``all`` and change the ``mode`` according to the preference
```python
dataset = dict(
    name = "all",
    mode=['place','cast','act','aud'],
)
 ```


### Preprocessing
It is able to follow the following codes to process the data. Remember to read the ``argparase`` to choose an ideal setting.

```sh
cd pre
python ShotDetect/shotdetect.py # Cut shot 
python place/extrac_feat.py     # Extract place feature
python audio/extrac_feat.py     # Extract audio feature
 ```

And the full feature extraction is updated in [movienet-tools](https://github.com/movienet/movienet-tools)

### Demo
#### Preparation
- [pytube](https://github.com/nficano/pytube) is to download YouTube video. Install with ```pip install pytube3 --upgrade```
- FFMPEG is to cut scene video and it is usually installed by your OS

#### Run
```sh
cd pre
python demodownload.py ## Download a YouTube video with pytube
python ShotDetect/shotdetect.py --print_result --save_keyf --save_keyf_txt ## Cut shot 
cd ../lgss
python run.py config/demo.py ## Cut scene 
 ```

#### Notice
The video link in the ``pre/demodownload.py`` might be invalid as time goes, and it may change to your own.

The demo code only use the image place feature for simplicity and casues inferior performance. It may change the threshold here to have a slight modification. The higher the threshold, the less scene it will generate. ``scene_dict, scene_list = pred2scene(cfg, threshold=0.8)``

```pre/ShotDetect``` is developed based on [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/). The shot detector is optimized to suit for movie.
**Parallel shot detection** ``shotdetect_p.py`` is also included for future usage.