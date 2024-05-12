# coding=utf-8
import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


#资源
src_path='./Source/'
out_path='./Result/'
src='bus.jpg'
video='2.mp4'
model_path = 'Model/pose_landmarker_heavy.task'

if __name__=='__main__':
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    cap = cv2.VideoCapture(src_path+video)
    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            #获取帧，帧率，时间戳
            success, frame = cap.read()
            frame_timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            if not success:
              print("Ignoring empty camera frame.")
              # If loading a video, use 'break' instead of 'continue'.
              break

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
	        #画图
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            
            annotated_image = draw_landmarks_on_image(frame, results)


 
            cv2.imshow('MediaPipe Holistic', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
              break
               
    cap.release()
    cv2.destroyAllWindows()

    






