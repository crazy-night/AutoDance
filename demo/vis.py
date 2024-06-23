import sys
import argparse
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from lib.preprocess import h36m_coco_format, revise_kpts
from lib.mmlab.gen_kpts import gen_video_kpts as mm_pose
from pos2vmd import pos2vmd

import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed



sys.path.append(os.getcwd())
from model.mhformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend("agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def get_pose2D(video_path, output_dir,model):
    print("\nGenerating 2D pose...")

    output_dir += "input_2D/"
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + "keypoints.npz"
    
    if os.path.exists(output_npz):       
        print("2D pose exists!")
        return
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores = mm_pose(video_path, model,num_peroson=1)
        #keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    np.savez_compressed(output_npz, reconstruction=re_kpts)
    print("Generating 2D pose successfully!")


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.pad = (args.frames - 1) // 2
    args.previous_dir = "checkpoint/pretrained/351"
    args.n_joints, args.out_joints = 17, 17

    ## 3D
    print("\nGenerating 3D pose...")
    output_3d_all = []

    os.makedirs(output_dir + "output_3D/", exist_ok=True)
    output_npz = output_dir + "output_3D/" + "output_keypoints_3d.npz"

    if os.path.exists(output_npz):
        print("3D pose exists!")
        return
    
    ## Reload
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, "*.pth")))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    ## input
    keypoints = np.load(output_dir + "input_2D/keypoints.npz", allow_pickle=True)[
        "reconstruction"
    ]

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    

    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        img_size = img.shape

        ## input frames
        start = max(0, i - args.pad)
        end = min(i + args.pad, len(keypoints[0]) - 1)

        input_2D_no = keypoints[0][start : end + 1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(
                input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), "edge"
            )

        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(
            input_2D_no, w=img_size[1], h=img_size[0]
        )

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[
            :, joints_right + joints_left
        ]
        input_2D = np.concatenate(
            (np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0
        )

        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype("float32")).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[
            :, :, joints_right + joints_left, :
        ]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, args.pad].unsqueeze(1)
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        output_3d_all.append(post_out)

    ## save 3D keypoints
    output_3d_all = np.stack(output_3d_all, axis=0)
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print("Generating 3D pose successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="sample_video.mp4", help="input video"
    )
    parser.add_argument("--gpu", type=str, default="0", help="input video")
    parser.add_argument("--model", type=str, default="vitpose-h", help="input video")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = "./demo/video/" + args.video
    video_name = video_path.split("/")[-1].split(".")[0]
    output_dir = "./demo/output/" + video_name + "/"

    get_pose2D(video_path, output_dir, args.model)
    get_pose3D(video_path, output_dir)
    vmd_path = output_dir
    #default: 1 person
    pos2vmd(output_dir + "output_3D/" + "output_keypoints_3d.npz", path=vmd_path)
