# This file is OK

import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!

# cap_width = 1280
# cap_height = 720
cap_width = 1920
cap_height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, value=cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value=cap_height)

# r, frame = cap.read()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # Drawing utilities

pTime = 0
cTime = 0

while cap.isOpened():
    success, img = cap.read()

    # flip the image by horizontally:
    # 0-vertically (flip around the x-axis)
    # 1-horizontally (flip around the y-axis)
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    yellow_color = (0, 255, 255) #BGR
    lightBlue_color = (255, 255, 0) #BGR

    hand_type = ""
    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[idx].classification[0].label
            hand_color = yellow_color if hand_type=="Left" else lightBlue_color

            for id, lm in enumerate(handLms.landmark):
                # print("----- ", id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 7, hand_color, cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        hand_len = len(results.multi_handedness)
        if hand_type == "":
            hand_type = "Hand not found"
        elif hand_len == 2:
            hand_type = "Both Hands"
        else:
            hand_type = hand_type + " Hand"

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(img, hand_type, (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()