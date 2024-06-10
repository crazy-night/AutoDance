# AutoDance
This is just a uncompleted introduction. This project is an implement of MHformer and aims at transfer video to vmd files, which can be used in games like HS2.



## Installation
Please check your environment before installation.
install anaconda,cuda,cudnn before

- Create a conda environment: ```conda create -n mhformer python=3.9```
- Activate the environment: ```conda activate mhformer```
- Install PyTorch 1.7.1 and Torchvision 0.8.2 following the [official instructions](https://pytorch.org/)
- ```pip3 install -r requirements.txt```
  

## Download pretrained model

The pretrained model can be found in [here](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing), please download it and put it in the './checkpoint/pretrained' directory. 


## Demo
The YOLOv8 pretrained models will be automatedly downloaded to your computer.
Then, you need to put your in-the-wild videos in the './demo/video' directory. 

Run the command below:
```bash
python demo/vis.py --video sample_video.mp4
```


## Acknowledgement

My code is extended from the following repositories. We thank the authors for releasing the codes. 

- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
## Licence

This project is licensed under the terms of the MIT license.
