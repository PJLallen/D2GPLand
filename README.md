# D2GPLand [MICCAI'24]
<div align=center>
<img src="assets/Figure1.png"  height=230 width=750>
</div>


Official Implementation of MICCAI 2024 paper "[Depth-Driven Geometric Prompt Learning for Laparoscopic Liver Landmark Detection](https://arxiv.org/abs/2406.17858)"

[Jialun Pei*](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en), Ruize Cui*, Yaoqian Li, [Weixin Si](https://scholar.google.com/citations?user=E4efwTgAAAAJ&hl=zh-CN&oi=ao), [Jing Qin](https://harry-qinjing.github.io/), and [Pheng-Ann Heng](https://scholar.google.com/citations?user=OFdytjoAAAAJ&hl=zh-CN)

[[Paper]](https://arxiv.org/abs/2406.17858); [[Chinese Version]](); [[Official Version]]()

**Contact:** jwxsics@gmail.com, peijialun@gmail.com


<div align=center>
<img src="assets/Overview.jpg"  height=400 width=700>
</div>


## Environment preparation
The code is tested on python 3.9.19, pytorch 2.0.1, and CUDA 11.7, change the versions below to your desired ones.
1. Clone repository:
```shell
git clone https://github.com/PJLallen/D2GPLand.git

cd D2GPLand
```
   
2. Set up anaconda environment:
```shell
# Create D2GPLand anaconda environment from YAML.file
conda env create -f D2GPLand.yaml
# Activate environment
conda activate D2GPLand
```

## Dataset preparation

### Download the datasets and annotation files
- Currently we only release the test set of the L3D dataset: 
### Register datasets
Change the path of the datasets as:
```shell
DATASET_ROOT = 'D2GPLand/L3D/'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train/')
TEST_PATH = os.path.join(DATASET_ROOT, 'Test/')
VAL_PATH = os.path.join(DATASET_ROOT, 'Val/')
```
## Pre-trained paths
We provide the weights of our D2GPLand with SAM-b and ResNet-34 as backbones: [Google Drive](https://drive.google.com/drive/folders/1Mll-izyMLoCnTxfW5LOJhzaThipnUSg0?usp=drive_link)

## Usage
### Train

```shell
python train.py --data_path {PATH_TO_DATASET} \
--batch_size 4 --lr 1e-4 --decay_lr 1e-6 --epoch 60
```

Please replace {PATH_TO_DATASET} to your own dataset dir

### Eval

```shell
python test.py --model_path {PATH_TO_THE_MODEL_WEIGHTS} \
  --prototype_path {PATH_TO_THE_PROTOTYPE_WEIGHTS} \
  --data_path {PATH_TO_DATASET}
```

- `{PATH_TO_THE_MODEL_WEIGHTS}`: please put the pre-trained model weights here
- `{PATH_TO_THE_PROTOTYPE_WEIGHTS}`: please put the pre-trained prototype weights here
- `{PATH_TO_DATASET}`: please put the dataset dir here
  
## Acknowledgement
This work is based on:

- [AdelaiDepth](https://github.com/aim-uofa/AdelaiDepth)
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

Thanks them for their great work!

## Citation

If this helps you, please cite this work:

```
@inproceedings{pei2024land,
  title={Depth-Driven Geometric Prompt Learning for Laparoscopic Liver Landmark Detection},
  author={Pei, Jialun and Cui, Ruize and Li, Yaoqian and Si, Weixin and Qin, Jing and Heng, Pheng-Ann},
  booktitle={MICCAI},
  year={2024},
}
```

