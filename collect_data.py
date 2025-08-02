import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

gesture_label = input("Enter label for gesture (e.g., thumbs_up): ")

data = []
cap = cv2.VideoCapture(0)

print("[INFO] Showing webcam feed. Press 'q' to stop recording.")

while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Could not access webcam.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_landmarks = result.multi_hand_landmarks[0]
        hand_type = result.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

        coords = []
        for lm in hand_landmarks.landmark:
            x = 1 - lm.x if hand_type == 'Left' else lm.x  # ✅ Flip if left hand
            coords.extend([x, lm.y, lm.z])

        if len(coords) == 63:
            data.append(coords)

        # Draw landmarks for user feedback
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display info
    cv2.putText(img, f"Gesture: {gesture_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Gesture Data", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(f"data/{gesture_label}.csv", index=False)
print(f"[INFO] Data saved for gesture: {gesture_label} ({len(data)} samples)")
