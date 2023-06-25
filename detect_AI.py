import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

with open('detector.pkl', 'rb') as file:
    model = pickle.load(file)


def pushup_detector(frame):
    data = []
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        pose_info = []
        for landmark in landmarks:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            pose_info.extend([x, y, z])

        data.append(pose_info)
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    if prediction[0] == 'pushup':
        return "PUSHUP"
    elif prediction[0] == 'pullup':
        return "PULLUP"
    else:
        return "NONE"


def upload_image():
    path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')])
    if path:
        image = cv.imread(path)
        result = pushup_detector(image)
        display_result(result)


def live_detection():
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
    cv.destroyAllWindows()


def display_result(result):
    result_label.config(text=result)


window = tk.Tk()
window.title("Pushup Detector - Anshal Kapil")


upload_image_button = tk.Button(
    window, text="Upload Image", command=upload_image)
upload_image_button.pack(pady=10)

live_detection_button = tk.Button(
    window, text="Live Detection", command=live_detection)
live_detection_button.pack(pady=10)


result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack(pady=20)

window.mainloop()
