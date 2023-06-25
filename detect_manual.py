import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def pushup_detector(frame):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            wrist_distance = abs(left_wrist.x - right_wrist.x)
            pushup_threshold = 0.15

            if wrist_distance < pushup_threshold:
                return "PUSHUP"

    return "NOT A PUSH UP"


capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    result = pushup_detector(frame)
    cv.putText(frame, result, (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Pushup Detector", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows
