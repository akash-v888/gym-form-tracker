import cv2
import mediapipe as mp
import pickle
import os

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Dataset and label storage
data = []
labels = []

# Label map
label_map = {
    'f': 'fist',
    'o': 'ok',
    't': 'thumbs_up',
    'd': 'thumbs_down',
    'p': 'peace',
    'h': 'open_hand'
}

print("==== Hand Gesture Data Collection ====")
print("Press one of the following keys to label a gesture:")
for k, v in label_map.items():
    print(f"  '{k}' --> {v}")
print("Press Backspace to undo the last sample.")
print("Press ESC to stop and save data.\n")

save_path = "hand_gesture_dataset.pkl"
append_mode = False

if os.path.exists(save_path):
    choice = input(f"\n'{save_path}' already exists. Append to it? (y/n): ").strip().lower()
    if choice == 'y':
        append_mode = True
        with open(save_path, 'rb') as f:
            existing = pickle.load(f)
            data.extend(existing.get('data', []))
            labels.extend(existing.get('labels', []))
        print(f"Loaded {len(existing['labels'])} existing samples.")
    else:
        print("Starting fresh. Existing file will be overwritten.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    gesture_label = None
    landmarks = []
    hand_label = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        for lm in hand.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        # Get handedness label
        if results.multi_handedness:
            handedness_dict = results.multi_handedness[0].classification[0]
            hand_label = handedness_dict.label  # 'Left' or 'Right'

        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

        handedness_msg = f"Detected: {hand_label} hand" if hand_label else "Hand detected"
        cv2.putText(frame, handedness_msg,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    # Screen commands
    y_offset = 60
    for k, v in label_map.items():
        msg = f"'{k}' --> {v}"
        cv2.putText(frame, msg, (10, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 255, 200), 2)
        y_offset += 40

    # Add undo instruction
    cv2.putText(frame, "'Backspace' --> Undo last sample", (10, y_offset + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

    # Show frame
    cv2.imshow('Collect Hand Data', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 8:  # Backspace
        if data and labels:
            removed_label = labels.pop()
            data.pop()
            print(f"Undo: Removed last sample ({removed_label}), total: {len(labels)})")
        else:
            print("Nothing to undo.")
    elif chr(key) in label_map and landmarks and hand_label:
        gesture = label_map[chr(key)]
        gesture_label = f"{gesture}_{hand_label[0]}"  # e.g., thumbs_up_L
        data.append(landmarks)
        labels.append(gesture_label)
        print(f"Captured: {gesture_label} (total: {len(labels)})")

with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\nSaved {len(labels)} samples to '{save_path}'")
cap.release()
cv2.destroyAllWindows()
