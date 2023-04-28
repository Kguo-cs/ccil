This repo is the implementation of the following paper:

** Ke Guo, Wei Jing, Junbo Chen, Jia Pan. CCIL: Context-conditioned imitation learning for urban driving. RSS, 2023**

## Lyft Dataset
Download Lyft's [Python software kit](https://github.com/woven-planet/l5kit). 
Download the [Lyft Motion Prediction Dataset](https://level-5.global/download/); only the files in ```Training Dataset(8.4GB), validation Dataset (8.2GB), Aerial Map and Semantic Map``` are needed. 
Store all files in a single folder to match this structure: https://woven-planet.github.io/l5kit/dataset.html.

## nuPlan Dataset
Download nuPlan's [Python software kit](https://github.com/motional/nuplan-devkit). 
Download the [nuPlan-v1.0 Dataset](https://www.nuscenes.org/nuplan#download); only the files in ```Maps, Val Split, Test Split and Train Split for Las Vegas City``` are needed. 
Store all files in a single folder to match this structure: https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md.

### Training
Run ```train.py``` to learn the planner. You need to specify the model name ```--model_name``` and the file paths to dataset ```--data_root```. Leave other arguments vacant to use the default setting.
```shell
python train.py \
--name lyft \
--data_root /path/to/lyft/data \
```

### Closed-loop testing
Run ```eval.py``` to do closed-loop testing. You need tospecify the model name ```--model_name``` and the file paths to dataset ```--data_root```. Leave other arguments vacant to use the default setting.
```shell
python eval.py \
--name lyft \
--data_root /path/to/lyft/data \
```

