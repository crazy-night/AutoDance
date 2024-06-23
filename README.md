<div align='center'>
	<font size=70> AutoDance </font>
</div>

This project is an simple implement of MHformer and aims at transferring dancing video to vmd files.
����Ŀ�Ƕ�MHFormer��ʵ��Ӧ�ã�����ʶ�����赸��Ƶ������vmd�ļ��������뵥��ȫ����Ƶ�Ա�֤Ч������Ƶ��һ֡�������������ɵ����ﶯ������h36m��ʽ������17���ؼ��㣬û�б��顢�ֲ����㲿������



## Installation
Please check your environment before installation. Nvidia GPU is highly recommanded.
���ڰ�װǰ������Ӳ�����Ƽ�Ӣΰ���Կ����鿴����ͺ��Ƿ�֧��cuda11.1
Install anaconda,cuda,cudnn first.
���Ȱ�װAnaconda������NVIDIA�����Լ���װcuda��cudnn��

- Create a conda environment: 
- �������⻷����
```conda create -n mhformer python=3.9```
- Activate the environment:
- �������⻷����
```conda activate mhformer```
- Install PyTorch 1.9.1, Torchvision 0.10.1 and cuda11.1 following the [official instructions](https://pytorch.org/):
- ����[�ٷ��ĵ�](https://pytorch.org/)��װcuda11.1��pytorch 1.9.1��torchvision 0.10.1������汾�����н������
```pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html```
- Install packages: 
- ��װ��Ӧ�İ���
```pip3 install -r requirements.txt```
- Install mmpose following the [official instructions](https://mmpose.readthedocs.io/zh-cn/dev-1.x/installation.html). Start from Step.3:
- ����[�ٷ��ĵ�](https://mmpose.readthedocs.io/zh-cn/dev-1.x/installation.html)�ӵ�������ʼ��װmmpose��
```pip install -U openmim```
```mim install mmengine```
```mim install "mmcv>=2.0.1"```
```mim install "mmdet>=3.1.0"```
```mim install "mmpose>=1.1.0"```
```mim install 'mmpretrain>=1.0.0'```
  

## Download pretrained model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing) (from MHFormer), please download it and put it in the './checkpoint/pretrained' directory.
MHFormer�ṩ��Ԥѵ��ģ�ͣ�����[�˴�](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing)���أ����ŵ�'./checkpoint/pretrained'Ŀ¼�¡�


## Demo
MMPose will automately download the pretrained models. Please leave enough place.
mmpose���Զ����������ģ�ͣ�����c�������㹻�Ŀռ䡣��Ĭ�ϵ�vitposeĿǰ׼ȷ����ߣ��ٶȽ�����Լռ2G��
Then, you need to put your in-the-wild videos in the './demo/video' directory. 
Ȼ������Ҫ���Լ�����Ƶ����'./demo/video'Ŀ¼�¡�
Run the command like below:
����������ִ���������ʾ�����£�
```bash
python demo/vis.py --video sample_video.mp4 --model vitpose-h
```

- P.S.
- ע��
Please ignore the warning messages from mmengine.
���������mmengine�ľ��档
The output will be saved in the './demo/output' directory.
����ļ����ᱣ����'./demo/output'�ļ����¡�
You can change the model according to the [official instructions](https://mmpose.readthedocs.io/zh-cn/dev-1.x/model_zoo/body_2d_keypoint.html)
����Ը���mmpose��[�ٷ��ĵ�](https://mmpose.readthedocs.io/zh-cn/dev-1.x/model_zoo/body_2d_keypoint.html)��������ʹ�õ�ģ�͡�


##

## Acknowledgement

My code is extended from the following repositories. I thank the authors for releasing the codes. 

- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [OpenMMD](https://github.com/peterljq/OpenMMD)
- [MMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x)
## Licence

This project is licensed under the terms of the MIT license.
