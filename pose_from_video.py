from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model
model = YOLO('yolov8n-pose.pt')

# Set exercise type manually for now: 'squat' or 'pushup'
EXERCISE = 'pushup'

# Toggle for live mode or video analysis
USE_WEBCAM = True
VIDEO_PATH = 'squat_test.mp4'  # Replace with video file

# Capture source
if USE_WEBCAM:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

# Save output only for video files
SAVE_OUTPUT = not USE_WEBCAM
out = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter('output_annotated.mp4', fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

# Angle calculation helper
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# Rep tracking
rep_count = 0
squat_down = False
feedback = ""
color = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    kps = results[0].keypoints

    if kps is not None and kps.xy.shape[0] > 0:
        try:
            keypoints = kps.xy[0]

            if EXERCISE == 'squat':
                hip = keypoints[11]
                knee = keypoints[13]
                ankle = keypoints[15]
                ang = angle(hip, knee, ankle)

                cv2.putText(annotated, f'Knee angle: {int(ang)}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if ang > 130:
                    feedback = "Too Shallow"
                    color = (0, 0, 255)
                elif ang < 60:
                    feedback = "Too Deep"
                    color = (0, 165, 255)
                else:
                    feedback = "Good Depth"
                    color = (0, 255, 0)

                if ang < 90 and not squat_down:
                    squat_down = True
                if ang > 150 and squat_down:
                    rep_count += 1
                    squat_down = False

            elif EXERCISE == 'pushup':
                shoulder = keypoints[6]
                elbow = keypoints[8]
                wrist = keypoints[10]
                ang = angle(shoulder, elbow, wrist)

                cv2.putText(annotated, f'Elbow angle: {int(ang)}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if ang > 150:
                    feedback = "Arms Too Straight"
                    color = (0, 0, 255)
                elif ang < 60:
                    feedback = "Too Low"
                    color = (0, 165, 255)
                else:
                    feedback = "Good Form"
                    color = (0, 255, 0)

                if ang < 90 and not squat_down:
                    squat_down = True
                if ang > 140 and squat_down:
                    rep_count += 1
                    squat_down = False

            # Shared overlay
            cv2.putText(annotated, feedback, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated, f'Reps: {rep_count}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No person detected in this frame.")
        cv2.putText(annotated, 'No person detected', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if SAVE_OUTPUT and out is not None:
        out.write(annotated)

    cv2.imshow('Video Pose Tracking', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_OUTPUT and out is not None:
    out.release()
cv2.destroyAllWindows()
