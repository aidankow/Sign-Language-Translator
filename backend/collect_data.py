import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

current_label = "A"  # Default starting letter
dataset_file = "dataset.csv"

print("="*50)
print("ASL DATA COLLECTOR STARTED")
print("1. Hold your hand sign in front of the camera.")
print("2. Tap SPACEBAR repeatedly to capture samples.")
print("3. Type letter keys (A-Z) to switch labels.")
print("4. Press 'q' to quit and save.")
print("="*50)

data_buffer = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalise hand landmarks by subtracting location of the wrist
            wrist = hand_landmarks.landmark[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    # Display UI box
    cv2.rectangle(frame, (20, 20), (450, 110), (40, 40, 40), -1)
    cv2.putText(frame, f"Target Label: {current_label}", (35, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Captured Samples: {len(data_buffer)}", (35, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    # waitKey(1) - returns 32-bit integer representing the pressed key
    # 0xFF - bitwise mask; last 8-bits represents ASCII value of the presed key
    
    if key == ord('q'):
        break
    elif key == ord(' '):
        if hand_detected:
            row = [current_label] + landmarks
            data_buffer.append(row)
            print(f"Captured sample for '{current_label}'! Total: {len(data_buffer)}")
        else:
            print("⚠️ No hand detected!")
    elif 97 <= key <= 122:
        current_label = chr(key).upper()
        print(f"Switched label to: {current_label}")
    elif 65 <= key <= 90:
        current_label = chr(key)
        print(f"Switched label to: {current_label}")

cap.release()
cv2.destroyAllWindows()

if len(data_buffer) > 0:
    with open(dataset_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_buffer)
    print(f"Successfully saved {len(data_buffer)} samples to {dataset_file}!")