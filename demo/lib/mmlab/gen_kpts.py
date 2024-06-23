# -*- coding: utf-8 -*-
from mmpose.apis import MMPoseInferencer
from mmpose.utils import register_all_modules
import numpy as np
import cv2
from tqdm import tqdm


def gen_video_kpts(video, model, num_peroson=1):

    register_all_modules()

    # 使用模型别名创建推理器
    inferencer = MMPoseInferencer(model)

    # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer(video, show=False)

    kpts_result = []
    scores_result = []

    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for result in tqdm(result_generator, total=frame_count):
        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for person in range(num_peroson):
            preds = result["predictions"][0][person]
            kpts[person] = preds["keypoints"]
            scores[person] = preds["keypoint_scores"]
        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)
    keypoints = keypoints.transpose(
        1, 0, 2, 3
    )  # (Time, Person, Num, 2) --> (Person, Time, Num, 2)
    scores = scores.transpose(1, 0, 2)  # (Time, Person, Num) --> (Person, Time, Num)

    return keypoints, scores
