import time
import cv2
import mediapipe as mp

from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

with mp_hands.Hands(
    static_image_mode = True,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        _____start_read = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        _____end_read = time.time()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        #image = image[:360, 640:]

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        _____start_process = time.time()
        results = hands.process(image)
        _____end_process = time.time()

        ______start_vis = time.time()
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bbox_width, bbox_height = 0, 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        ______end_vis = time.time()
        """
        print(
            bbox_width, bbox_height,
            "{0:.5f} {1:.5f} {2:.5f} {3:.5f}".format(
                _____end_read - _____start_read,
                _____end_process - _____start_process,
                ______end_vis - ______start_vis,
                ______end_vis - _____start_read
            )
        )
        """

    cap.release()