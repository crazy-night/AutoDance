# **AutoDance**

This project is an simple implement of MHformer and aims at transferring dancing video to vmd files.
本项目是对MHFormer的实际应用，用于识别单人舞蹈视频并生成vmd文件。请输入单人全身视频以保证效果，视频第一帧必须出现人物，生成的人物动作依据h36m格式，共有17个关键点，没有表情、手部和足部动作。



## Installation
Please check your environment before installation. Nvidia GPU is highly recommanded.
请在安装前检查你的硬件，推荐英伟达显卡并查看你的型号是否支持cuda11.1
Install anaconda,cuda,cudnn first.
请先安装Anaconda，更新NVIDIA驱动以及安装cuda和cudnn。

- Create a conda environment: 
- 创建虚拟环境：
```conda create -n mhformer python=3.9```
- Activate the environment:
- 激活虚拟环境：
```conda activate mhformer```
- Install PyTorch 1.9.1, Torchvision 0.10.1 and cuda11.1 following the [official instructions](https://pytorch.org/):
- 按照[官方文档](https://pytorch.org/)安装cuda11.1的pytorch 1.9.1和torchvision 0.10.1（其余版本请自行解决）：
```pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html```
- Install packages: 
- 安装相应的包：
```pip3 install -r requirements.txt```
- Install mmpose following the [official instructions](https://mmpose.readthedocs.io/zh-cn/dev-1.x/installation.html). Start from Step.3:
- 按照[官方文档](https://mmpose.readthedocs.io/zh-cn/dev-1.x/installation.html)从第三步开始安装mmpose：
```pip install -U openmim```
```mim install mmengine```
```mim install "mmcv>=2.0.1"```
```mim install "mmdet>=3.1.0"```
```mim install "mmpose>=1.1.0"```
```mim install 'mmpretrain>=1.0.0'```
  

## Download pretrained model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing) (from MHFormer), please download it and put it in the './checkpoint/pretrained' directory.
MHFormer提供了预训练模型，请在[此处](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing)下载，并放到'./checkpoint/pretrained'目录下。


## Demo
MMPose will automately download the pretrained models. Please leave enough place.
mmpose会自动下载所需的模型，请在c盘留下足够的空间。（默认的vitpose目前准确率最高，速度较慢，约占2G）
Then, you need to put your in-the-wild videos in the './demo/video' directory. 
然后你需要把自己的视频放在'./demo/video'目录下。
Run the command like below:
在命令行中执行类似命令，示例如下：
```bash
python demo/vis.py --video sample_video.mp4 --model vitpose-h
```

- P.S.
- 注意
Please ignore the warning messages from mmengine.
请忽略来自mmengine的警告。
The output will be saved in the './demo/output' directory.
输出文件将会保存在'./demo/output'文件夹下。
You can change the model according to the [official instructions](https://mmpose.readthedocs.io/zh-cn/dev-1.x/model_zoo/body_2d_keypoint.html)
你可以根据mmpose的[官方文档](https://mmpose.readthedocs.io/zh-cn/dev-1.x/model_zoo/body_2d_keypoint.html)，更改所使用的模型。


##

## Acknowledgement

My code is extended from the following repositories. I thank the authors for releasing the codes. 

- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [OpenMMD](https://github.com/peterljq/OpenMMD)
- [MMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x)
## Licence

This project is licensed under the terms of the MIT license.
