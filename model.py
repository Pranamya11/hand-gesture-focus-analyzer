import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        h, w, _ = img.shape
        x = int(lm[8].x * w)   # Index finger tip
        y = int(lm[8].y * h)

        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        # Draw only when index finger is up
        if lm[8].y < lm[6].y:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

    img = cv2.add(img, canvas)
    cv2.imshow("Air Drawing", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
