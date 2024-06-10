from tabnanny import verbose
from ultralytics import YOLO
import numpy as np
import cv2
import copy
from tqdm import tqdm

model_path = "checkpoint/yolov8x-pose.pt"


def gen_video_kpts(video, num_peroson=1):

    # Loading detector and pose model
    model = YOLO(model_path, task="pose", verbose=False)

    results = model(video, stream=True, verbose=False)

    kpts_result = []
    scores_result = []
    for index, result in enumerate(tqdm(results)):
        temp = result.keypoints.cpu()
        kpts = temp.xy.numpy()
        score = temp.conf.numpy()
        kpts_result.append(kpts)
        scores_result.append(score)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(
        1, 0, 2, 3
    )  # (Time, Person, Num, 2) --> (Person, Time, Num, 2)
    scores = scores.transpose(1, 0, 2)  # (Time, Person, Num) --> (Person, Time, Num)
    return keypoints, scores
