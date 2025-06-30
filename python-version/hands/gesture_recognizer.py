import cv2
import mediapipe as mp
import numpy as np
import pickle
from neural_net_algo import SimpleNeuralNet
from sklearn.preprocessing import LabelEncoder

# Load label encoder
with open('gesture_label_encoder.pkl', 'rb') as f:
    label_encoder: LabelEncoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

# Load trained model
model = SimpleNeuralNet(input_size=42, hidden_size=64, output_size=num_classes)
model.load_model('gesture_model.pkl')

# Setup MediaPipe Hands
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
print("Starting gesture recognition. Press ESC to exit.")

prev_label = None
conf_thresh = 0.5

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

        # Predict with softmax
        probs = model.forward(keypoints)
        pred_class = np.argmax(probs)
        confidence = np.max(probs)
        label = label_encoder.inverse_transform([pred_class])[0]

        # Show top 3 predictions
        top3_idx = np.argsort(probs[0])[::-1][:3]
        top3_labels = label_encoder.inverse_transform(top3_idx)
        top3_conf = probs[0][top3_idx]

        print("Top-3 Predictions:")
        for lbl, conf in zip(top3_labels, top3_conf):
            print(f"  {lbl}: {conf:.2f}")

        if confidence > conf_thresh:
            prediction_text = f"{label} ({confidence:.2f})"
            prev_label = prediction_text
        else:
            prediction_text = "Uncertain"

    else:
        if prev_label:
            prediction_text = f"Last seen: {prev_label}"

    # Display result
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Gesture Recognizer", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
