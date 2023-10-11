## Installation

1. Environment Setup
```bash
conda create -n detr python=3.8 -y 

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install 'mmengine==0.8.4'
mim install 'mmcv==2.0.0rc4'
mim install 'mmdet==3.0.0rc5'
pip install setuptools==59.5.0
pip install -v -e .
```
2. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/

3. Downloads the [language model and config](https://drive.google.com/drive/folders/1Lya4cSzh62S8shMjGXUWAndAReaIEMzZ?usp=sharing) to /text
## Get Started

1. Please Download nuScenes dataset to data/nuscnes and run 
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
which should look like:
|--nuscnes
  |--maps
  |--samples
  |--sweeps
  |--v1.0-test
  |--v1.0-trainval
  |--nuscenes_gt_database
  |--nuscenes_dbinfos_train.pkl
  |--nuscenes_infos_test.pkl
  |--nuscenes_infos_val.pkl
  |--nuscenes_infos_train.pkl

2. For example, to train DETR3D on 2 GPUs with bert encoder and half data, please use

```bash
bash tools/dist_train.sh projects/DETR3D/configs/detr3d_r50_bert_gridmask_halfdata.py 2 --cfg-options load_from=pretrained/fcos3d.pth
```