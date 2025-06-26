from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model
model = YOLO('yolov8n-pose.pt')

# Load video file
video_path = 'exercise_video.mp4'  # replace with your actual file
cap = cv2.VideoCapture(video_path)

# Setup video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
out = cv2.VideoWriter('output_annotated.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    kps = results[0].keypoints

    if kps is not None:
        keypoints = kps.xy[0]  # person 0
        try:
            hip = keypoints[11]
            knee = keypoints[13]
            ankle = keypoints[15]

            ang = angle(hip, knee, ankle)
            cv2.putText(annotated, f'Knee angle: {int(ang)} deg', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass  # keypoints may be missing or not detected

    out.write(annotated)
    cv2.imshow('Video Pose Tracking', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
