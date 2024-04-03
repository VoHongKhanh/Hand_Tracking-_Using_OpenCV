import math
import mediapipe as mp
import cv2


class SimpleGestureDetector:
    # region: Member variables
    # mediaPipe configuration hands object
    __mpHands = mp.solutions.hands
    # mediaPipe detector objet
    __mpHandDetector = None

    def __init__(self):
        self.__setDefaultHandConfiguration()

    def __setDefaultHandConfiguration(self):
        self.__mpHandDetector = self.__mpHands.Hands(
            # default = 2
            max_num_hands=2,
            # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully (default= 0.5)
            min_detection_confidence=0.5,
            # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. (default = 0.5)
            min_tracking_confidence=0.5
        )

    def __getEuclideanDistance(self, posA, posB):
        return math.sqrt((posA.x - posB.x) ** 2 + (posA.y - posB.y) ** 2)

    def __isThumbNearIndexFinger(self, thumbPos, indexPos):
        return self.__getEuclideanDistance(thumbPos, indexPos) < 0.1

    def detectHands(self, capture):
        if self.__mpHandDetector is None:
            return

        image = capture.color
        # Input image must contain three channel rgb data.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # lock image for hand detection
        image.flags.writeable = False
        # start handDetector on current image
        detectorResults = self.__mpHandDetector.process(image)
        # unlock image
        image.flags.writeable = True

        if detectorResults.multi_hand_landmarks:
            for handLandmarks in detectorResults.multi_hand_landmarks:
                self.simpleGesture(handLandmarks.landmark)

    def simpleGesture(self, handLandmarks):

        thumbIsOpen = False
        indexIsOpen = False
        middelIsOpen = False
        ringIsOpen = False
        pinkyIsOpen = False

        pseudoFixKeyPoint = handLandmarks[2].x
        if handLandmarks[3].x < pseudoFixKeyPoint and handLandmarks[4].x < pseudoFixKeyPoint:
            thumbIsOpen = True

        pseudoFixKeyPoint = handLandmarks[6].y
        if handLandmarks[7].y < pseudoFixKeyPoint and handLandmarks[8].y < pseudoFixKeyPoint:
            indexIsOpen = True

        pseudoFixKeyPoint = handLandmarks[10].y
        if handLandmarks[11].y < pseudoFixKeyPoint and handLandmarks[12].y < pseudoFixKeyPoint:
            middelIsOpen = True

        pseudoFixKeyPoint = handLandmarks[14].y
        if handLandmarks[15].y < pseudoFixKeyPoint and handLandmarks[16].y < pseudoFixKeyPoint:
            ringIsOpen = True

        pseudoFixKeyPoint = handLandmarks[18].y
        if handLandmarks[19].y < pseudoFixKeyPoint and handLandmarks[20].y < pseudoFixKeyPoint:
            pinkyIsOpen = True

        if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FIVE!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FOUR!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
            print("THREE!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("TWO!")

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("ONE!")

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("ROCK!")

        elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("SPIDERMAN!")

        elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("FIST!")

        elif not indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen and self.__isThumbNearIndexFinger(
                handLandmarks[4], handLandmarks[8]):
            print("OK!")

        print("FingerState: thumbIsOpen? " + str(thumbIsOpen) + " - indexIsOpen? " + str(
            indexIsOpen) + " - middelIsOpen? " +
              str(middelIsOpen) + " - ringIsOpen? " + str(ringIsOpen) + " - pinkyIsOpen? " + str(pinkyIsOpen))