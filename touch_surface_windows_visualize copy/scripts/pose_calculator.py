from mimetypes import read_mime_types
import os
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, Pipe, Value
from threading import Thread
from PyQt5 import QtWidgets, QtCore, QtGui
import mediapipe as mp

from scripts.calibration_window import CalibrationWindow
from scripts.utils import calc_landmark_list

from scripts.utils import *


import pyautogui
from pynput.mouse import Button, Controller

mouse = Controller()

class Posecalculator() :
    def __init__(
        self,
        to_visualizer_queue,
        frame_width = 1920,
        frame_height = 1080,
        fps         = 30,
        filter_recent_coeff = 0.4
    ) :

        self.to_visualizer_queue = to_visualizer_queue
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.screen_height = pyautogui.size().height
        self.screen_width = pyautogui.size().width
        self.fps = fps        
        
        self.filter_recent_coeff = filter_recent_coeff


        self.configs_n_vals = {
            "homography" : None,
            "checker_corner_shape" : (8, 5),

            "click_threshold" : 3900,
        

            "index_finger_idx" : 1,
            "midle_finger_idx" : 2,

            "index_finger_curr" : 0,
            "index_finger_prev" : 0,
            'midle_finger_curr' : 0,
            'midle_finger_prev' : 0
        }
        self.run()

    def run(self) :    
        self.video_cap = cv2.VideoCapture(0)
        self.video_cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, 
            self.frame_width
        )
        self.video_cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height
        )
        self.video_cap.set(
            cv2.CAP_PROP_FPS, self.fps
        )
        self.video_cap.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.readImage()

        self.calibrate()

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence = 0.3,
            min_tracking_confidence  = 0.3,
        )
        
        prev_index_finger_coord = None
        while True :
            ret = self.readImage()
            if not ret :
                print("IMAGE READ FAILED")

            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            hand_keypoints_result = hands.process(image)
            image.flags.writeable = True
            
            if hand_keypoints_result.multi_hand_landmarks is not None :
                hand_landmarks = hand_keypoints_result.multi_hand_landmarks[0]
                landmark_list = calc_landmark_list(image, hand_landmarks)

                index_finger_coord_cam = landmark_list[8]
                middle_finger_coord_cam = landmark_list[12]

                index_finger_coord = self.transformCoord(
                    index_finger_coord_cam
                )
                middle_finger_coord = self.transformCoord(
                    middle_finger_coord_cam
                )

                if index_finger_coord[0] < 0 :
                    #print("trimming low x")
                    index_finger_coord[0] = 1
                if index_finger_coord[0] >= self.screen_width :
                    #print("trimming hight x")
                    index_finger_coord[0] = self.screen_width - 1
                if index_finger_coord[1] < 0 :
                    #print("trimming low y")
                    index_finger_coord[1] = 1
                    #print("trimming")
                if index_finger_coord[1] >= self.screen_width :
                    index_finger_coord[1] = self.screen_height - 1
                
                index_finger_coord = np.array(index_finger_coord)
                if prev_index_finger_coord is not None :
                    index_finger_coord = index_finger_coord * self.filter_recent_coeff + prev_index_finger_coord * (1 - self.filter_recent_coeff)
                mouse.position = index_finger_coord

                print(index_finger_coord)

                for hand_landmarks, handedness in zip(
                    hand_keypoints_result.multi_hand_landmarks,
                    hand_keypoints_result.multi_handedness
                ) :
                    brect = calc_bounding_rect(self.image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(self.image, hand_landmarks)

                    # Drawing part
                    debug_image = draw_bounding_rect(
                        True,
                        self.image,
                        brect
                    )
                    debug_image = draw_landmarks(
                        debug_image,
                        landmark_list,
                        line_thickness = 4
                    )
                self.to_visualizer_queue.put(
                    debug_image
                )

            else : continue

    def readImage(self) :
        image = 10
        ret, image = self.video_cap.read()
        self.image = image
        return ret

    def calibrate(self) :
        app = QtWidgets.QApplication(sys.argv)

        self.calibrate_window = CalibrationWindow(
            self.configs_n_vals["checker_corner_shape"],
            self
        )
        self.calibrate_window.show()

        image_read_timer = QtCore.QTimer()
        image_read_timer.setInterval(100)
        image_read_timer.timeout.connect(self.readImage)

        app.exec_() 
        print("calibrated")

    def verifyCalibrating(self) :
        #print("checking homography exists")
        if self.configs_n_vals["homography"] is not None :
            self.calibrate_window = None

    def transformCoord(self, coord) :
        """
        coord : [x_coord, y_coord]
        """
        coord = np.array([coord[0], coord[1], 1]).reshape(3, 1)

        transformed = np.matmul(
            self.configs_n_vals["homography"],
            coord
        )
        
        #print(transformed)
        
        transformed = transformed.reshape(3,)
        transformed /= transformed[2]

        #print(transformed)
        #print()

        return [transformed[0], transformed[1]]

