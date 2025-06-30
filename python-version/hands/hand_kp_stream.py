import cv2
import mediapipe as mp
import socket
import struct
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# TCP connection to localhost:9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 9999))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        # Flatten 21 (x,y) into a list
        landmarks = []
        for lm in hand.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        # Send as packed floats
        data = struct.pack('f' * len(landmarks), *landmarks)
        try:
            sock.sendall(data)
        except BrokenPipeError:
            print("Connection to C++ receiver lost.")
            break

    time.sleep(0.05)  # avoid spamming too fast

cap.release()
sock.close()
