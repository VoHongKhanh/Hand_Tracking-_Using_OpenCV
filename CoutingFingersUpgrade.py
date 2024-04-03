import cv2
import mediapipe as mp
import tkinter as tk

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# For webcam input:
mp_hands = mp.solutions.hands


class VhkHand:
    def __init__(self, mp_hands, mp_drawing, image, cv2, hand_landmarks, hand_type):
        self.handType = hand_type
        self.hand_landmarks = hand_landmarks
        self.mp_hands = mp_hands
        self.image = image
        self.mp_drawing = mp_drawing
        self.cv2 = cv2
        self.fingersCount = 0
        self.fingers = []

    def isLeftHand(self):
        return self.handType == "Left"

    def isRightHand(self):
        return self.handType == "Right"

    def isThumbOpen(self):
        pseudoFixKeyPoint = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x
        thumbState = 0
        condition = pseudoFixKeyPoint > self.hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x > \
                self.hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x
        if (condition and self.isRightHand()) or (not condition and self.isLeftHand()):
            thumbState = 1
        return thumbState

    def isIndexOpen(self):
        pseudoFixKeyPoint = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        indexState = 0
        if pseudoFixKeyPoint > self.hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y > \
                self.hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
            indexState = 1
        return indexState

    def isMiddleOpen(self):
        pseudoFixKeyPoint = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        middleState = 0
        if pseudoFixKeyPoint > self.hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y > \
                self.hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
            middleState = 1
        return middleState

    def isRingOpen(self):
        pseudoFixKeyPoint = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        ringState = 0
        if pseudoFixKeyPoint > self.hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y > \
                self.hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y:
            ringState = 1
        return ringState

    def isPinkyOpen(self):
        pseudoFixKeyPoint = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        pinkyState = 0
        if pseudoFixKeyPoint > self.hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y > \
                self.hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.PINKY_TIP].y:
            pinkyState = 1
        return pinkyState

    def draw(self):
        self.mp_drawing.draw_landmarks(self.image, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def drawMainAxis(self):
        image_height, image_width, _ = image.shape
        sx = int(self.hand_landmarks.landmark[0].x * image_width)
        sy = int(self.hand_landmarks.landmark[0].y * image_height)
        start_point = (sx, sy)

        ex = int((self.hand_landmarks.landmark[9].x + self.hand_landmarks.landmark[13].x) * image_width / 2)
        ey = int((self.hand_landmarks.landmark[9].y + self.hand_landmarks.landmark[13].y) * image_height / 2)
        end_point = (ex, ey)

        # print(start_point, end_point)
        # White color in BGR
        color = (255, 255, 255)
        # Line thickness of 5 px
        thickness = 5
        # self.image = self.cv2.line(image, start_point, end_point, color, thickness)
        self.cv2.line(image, start_point, end_point, color, thickness)

    def checkHandStatus(self):
        thumb = self.isThumbOpen()
        index = self.isIndexOpen()
        middle = self.isMiddleOpen()
        ring = self.isRingOpen()
        pinky = self.isPinkyOpen()

        self.fingersCount = thumb + index + middle + ring + pinky
        if self.fingersCount == 0:
            self.fingers = ["None"]
        else:
            self.fingers = []
            if thumb == 1:
                self.fingers += ["Thumb"]
            if index == 1:
                self.fingers += ["Index"]
            if middle == 1:
                self.fingers += ["Middle"]
            if ring == 1:
                self.fingers += ["Ring"]
            if pinky == 1:
                self.fingers += ["Pinky"]


class VhkHands:
    def __init__(self, mp_hands, mp_drawing, results, image, cv2):
        self.leftHand = None
        self.rightHand = None
        self.image = image
        self.cv2 = cv2
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.results = results
        self.hands = ["None"]
        if results.multi_hand_landmarks:
            self.hands = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[idx].classification[0].label
                if hand_type == "Left":
                    self.hands += [hand_type]
                    self.leftHand = VhkHand(self.mp_hands, self.mp_drawing, self.image, self.cv2, hand_landmarks, hand_type)
                elif hand_type == "Right":
                    self.rightHand = VhkHand(self.mp_hands, self.mp_drawing, self.image, self.cv2, hand_landmarks, hand_type)
                    self.hands += [hand_type]

    def haveLeftHand(self):
        return "Left" in self.hands

    def haveRightHand(self):
        return "Right" in self.hands

    def printHands(self):
        y = 40
        c = 0
        h = self.hands
        if not h:
            h += "None"
        self.cv2.putText(self.image, "Detect hands: " + ", ".join(h), (40, y), self.cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        if self.leftHand:
            self.leftHand.checkHandStatus()
            y = y + 40
            fc = self.leftHand.fingersCount
            c = c + fc
            self.cv2.putText(self.image,
                             "Left hand (" + str(fc) + " finger" + ("s" if fc > 1 else "") + "): " +
                             ", ".join(self.leftHand.fingers),
                             (40, y), self.cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        if self.rightHand:
            self.rightHand.checkHandStatus()
            y = y + 40
            fc = self.rightHand.fingersCount
            c = c + fc
            self.cv2.putText(self.image,
                             "Right hand (" + str(fc) + " finger" + ("s" if fc > 1 else "") + "): " +
                             ", ".join(self.rightHand.fingers),
                             (40, y), self.cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        y = y + 40
        self.cv2.putText(self.image,
                         "Finger(s) count: " + str(c) + " finger" + ("s" if c > 1 else ""),
                         (40, y), self.cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


    def drawHands(self):
        if self.leftHand:
            self.leftHand.draw()
        if self.rightHand:
            self.rightHand.draw()

    def drawMainAxis(self):
        if self.leftHand:
            self.leftHand.drawMainAxis()
        if self.rightHand:
            self.rightHand.drawMainAxis()



def showFrameCenterScreen(cv2, winname, image, sw, sh):
    h, w, _ = image.shape
    cv2.namedWindow(winname)  # Create a named window
    x = int((sw - w) / 2)
    y = int((sh - h) / 2)
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, image)


# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!

cap_width = 1280
cap_height = 720
# cap_width = 1920
# cap_height = 1080

cap.set(cv2.CAP_PROP_FRAME_WIDTH, value=cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value=cap_height)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

        data = VhkHands(mp_hands, mp_drawing, results, image, cv2)
        data.printHands()
        data.drawHands()
        data.drawMainAxis()

        showFrameCenterScreen(cv2, 'MediaPipe Hands', image, screen_width, screen_height)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
