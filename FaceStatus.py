import math
import time
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)


####################################################################################################

def ConvertToPoint(landmark):
    return [landmark.x, landmark.y, landmark.z]


####################################################################################################

def CalcDistance(point1, point2):
    x1, y1, z1 = ConvertToPoint(point1)
    x2, y2, z2 = ConvertToPoint(point2)
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


####################################################################################################

def DetectLandmarks(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("no face")
        return None

    landmarks = results.multi_face_landmarks
    return landmarks


####################################################################################################

def DetectDirection(landmark):
    left = CalcDistance(landmark[5], landmark[234])
    right = CalcDistance(landmark[5], landmark[454])

    threshold = 2.5
    result = "straight"

    if (left < right):
        ratio = right / left
        if (ratio > threshold):
            result = "left"
    elif (right < left):
        ratio = left / right
        if (ratio > threshold):
            result = "right"

    return result


####################################################################################################

def DrawLandmark(frame, landmarks):
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())

    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_iris_connections_style())


####################################################################################################

# For static images:
IMAGE_FILES = []

startTime = time.time()
if (len(IMAGE_FILES) > 0):
    for idx, file in enumerate(IMAGE_FILES):
        frame = cv2.imread(file)
        landmarks = DetectLandmarks(frame)

        if (len(landmarks) == 0):
            continue

        landmark = landmarks[0].landmark

        direction = DetectDirection(landmark)
        DrawLandmark(frame, landmarks[0])

        cv2.putText(frame, direction, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("result {}".format(idx), frame)

    print("Elapsed: " + str(time.time() - startTime))
    cv2.waitKey()

else:  # không có ảnh truyền vào thì đọc webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        landmarks = DetectLandmarks(frame)

        if (len(landmarks) == 0):
            continue

        landmark = landmarks[0].landmark

        direction = DetectDirection(landmark)
        DrawLandmark(frame, landmarks[0])

        cv2.putText(frame, direction, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("result", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break