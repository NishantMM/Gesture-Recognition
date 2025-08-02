import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load('models/gesture_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time gesture recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    label = "No Hand"
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords += [lm.x, lm.y, lm.z]

        if len(coords) == 63:
            pred = model.predict([coords])
            label = pred[0]

        mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, f'Gesture: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
