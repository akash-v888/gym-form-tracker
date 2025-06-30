import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from train_model_knn import ImprovedKNNClassifier

# Load KNN model
with open('gesture_knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# Load label encoder
with open('gesture_knn_label_encoder.pkl', 'rb') as f:
    label_encoder: LabelEncoder = pickle.load(f)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting KNN gesture recognition. Press ESC to exit.")

prev_label = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        keypoints = np.array(keypoints).reshape(1, -1)

        # Predict top-3
        top_preds = knn.predict(keypoints, return_top_k=True)[0]  # get top-k from first (and only) sample
        labels = label_encoder.inverse_transform([pred[0] for pred in top_preds])
        confidences = [pred[1] for pred in top_preds]

        top_predictions = [f"{lbl} ({conf:.2f})" for lbl, conf in zip(labels, confidences)]
        prediction_text = " | ".join(top_predictions)
        prev_label = prediction_text

    else:
        if prev_label:
            prediction_text = f"Last seen: {prev_label}"

    # Display result
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("KNN Gesture Recognizer", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
