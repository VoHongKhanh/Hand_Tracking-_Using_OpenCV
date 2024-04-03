import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# For webcam input:
mp_hands = mp.solutions.hands

def isThumbOpen(hand_landmarks):
    pseudoFixKeyPoint = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
    thumbState = 0
    if pseudoFixKeyPoint > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
        thumbState = 1
    return thumbState


def isIndexOpen(hand_landmarks):
    pseudoFixKeyPoint = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    indexState = 0
    if pseudoFixKeyPoint > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
        indexState = 1
    return indexState


def isMiddleOpen(hand_landmarks):
    pseudoFixKeyPoint = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    middleState = 0
    if pseudoFixKeyPoint > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
        middleState = 1
    return middleState


def isRingOpen(hand_landmarks):
    pseudoFixKeyPoint = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    ringState = 0
    if pseudoFixKeyPoint > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
        ringState = 1
    return ringState


def isPinkyOpen(hand_landmarks):
    pseudoFixKeyPoint = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    pinkyState = 0
    if pseudoFixKeyPoint > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y:
        pinkyState = 1
    return pinkyState


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = []
                thumbState = isThumbOpen(hand_landmarks)
                if thumbState == 1:
                    fingers += ['Thumb']

                indexState = isIndexOpen(hand_landmarks)
                if indexState == 1:
                    fingers += ['Index']

                middleState = isMiddleOpen(hand_landmarks)
                if middleState == 1:
                    fingers += ['Middle']

                ringState = isRingOpen(hand_landmarks)
                if ringState == 1:
                    fingers += ['Ring']

                pinkyState = isPinkyOpen(hand_landmarks)
                if pinkyState == 1:
                    fingers += ['Pinky']

                fingersSum = thumbState + indexState + middleState + ringState + pinkyState
                # print(fingersSum)
                fps= str(fingersSum) + ' fingers'
                cv2.putText(image, (fps), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                cv2.putText(image, ", ".join(fingers), (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            cv2.putText(image, '0 finger', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(image, "Empty finger", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
