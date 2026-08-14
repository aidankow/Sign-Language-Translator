import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("model.joblib")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_norm_loc_zoom(hand_landmarks):
    # Norm_Loc
    wrist = hand_landmarks.landmark[0]
    centered = []
    for lm in hand_landmarks.landmark:
        centered.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    # Norm_Zoom
    max_val = max(max(abs(pt[0]), abs(pt[1])) for pt in centered)
    max_val = max(max_val, 1e-6)  # Prevent division by zero

    features = []
    for pt in centered:
        features.extend([pt[0] / max_val, pt[1] / max_val, pt[2] / max_val])

    return np.array(features, dtype=np.float32)

def process_frame(frame):
    """Takes an OpenCV frame, processes MediaPipe landmarks, runs prediction, and draws overlays."""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_norm_loc_zoom(hand_landmarks)

            input_data = np.array([features])
            probabilities = model.predict_proba(input_data)[0]
            max_index = np.argmax(probabilities)
            
            prediction = model.classes_[max_index]
            confidence = probabilities[max_index] * 100
            prediction_text = f"{prediction} ({confidence:.1f}%)"

            cv2.rectangle(frame, (20, 20), (320, 130), (40, 40, 40), -1)
            cv2.putText(frame, "Prediction & Confidence:", (35, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{prediction}  ({confidence:.1f}%)", (35, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, prediction_text

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("="*50)
    print("STANDALONE TRANSLATOR STARTED")
    print("Press 'q' to quit.")
    print("="*50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame, _ = process_frame(frame)
        cv2.imshow("Sign Language Translator", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()