from cProfile import label
from ctypes import sizeof
import os
import sys 
import time
import cv2
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


from scripts.image_plotter import ImagePlotter

class CalibrationWindow(QtWidgets.QMainWindow) :
    def __init__(
        self,
        checker_corner_shape,
        parent,
        margin_rate = 0.1
    ) :
        super(CalibrationWindow, self).__init__()

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.showMaximized()

        self.parent = parent
        self.checker_corner_shape = checker_corner_shape

        self.num_checker_hor = self.checker_corner_shape[0]
        self.num_checker_ver = self.checker_corner_shape[1]


        size_object = QtWidgets.QDesktopWidget().screenGeometry(-1)

        checker_image = self.generateCheckerBoard(
            screen_width  = size_object.width(),
            screen_height = size_object.height(),
            num_checker_ver = self.num_checker_ver,
            num_checker_hor = self.num_checker_hor,
            margin_rate = margin_rate
        )

        print(type(checker_image))
        print(checker_image.shape)

        self.label = QtWidgets.QLabel()
        self.label = ImagePlotter(size_object.width(), size_object.height())

        self.setCentralWidget(self.label)

        self.checker_image = checker_image
        
        self.label.update(checker_image)

        print(self.size())
        print(self.label.size())

        self.image_timer = QtCore.QTimer()
        self.image_timer.setInterval(100)

        def tmp() :
            self.parent.readImage()
            image = self.parent.image
            #self.label.update(image)

        #self.image_timer.timeout.connect(self.parent.readImage)
        self.image_timer.timeout.connect(tmp)
        self.image_timer.start()

        self.verify_timer = QtCore.QTimer()
        self.verify_timer.setInterval(100)
        self.verify_timer.timeout.connect(self.parent.verifyCalibrating)
        self.verify_timer.start()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.calibrateBase)
        self.timer.start()
        
    def generateCheckerBoard(
        self,
        screen_width,
        screen_height,
        num_checker_ver,
        num_checker_hor,
        margin_rate
    ) :
        checker_size_pix = math.floor(min(
            screen_height * (1 - margin_rate) / num_checker_ver,
            screen_width  * (1 - margin_rate) / num_checker_hor
        ))

        margin_top   = int((screen_height - checker_size_pix * num_checker_ver) / 2)
        margin_botom = screen_height - margin_top - checker_size_pix * num_checker_ver
        margin_left  = int((screen_width  - checker_size_pix * num_checker_hor) / 2)
        margin_right = screen_width - margin_left - checker_size_pix * num_checker_hor

        checker_seed = np.zeros((num_checker_ver, num_checker_hor), dtype=np.uint8)
        checker_seed[1::2,  ::2] = 255
        checker_seed[ ::2, 1::2] = 255

        checker = np.dstack((
            np.kron(checker_seed, np.ones((checker_size_pix, checker_size_pix))),
            np.kron(checker_seed, np.ones((checker_size_pix, checker_size_pix))),
            np.kron(checker_seed, np.ones((checker_size_pix, checker_size_pix)))
        ))

        checker_image = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
        checker_image[
            margin_top:screen_height - margin_botom,
            margin_left:screen_width -  margin_right,
            :    
        ] = checker

        return checker_image

    def calibrateBase(self) :
        print("calibrating")
        try :
            image_1 = self.parent.image
            image_2 = self.checker_image

            ret, corners_1 = cv2.findChessboardCorners(
                image_1,
                (self.num_checker_hor - 1, self.num_checker_ver - 1), 
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH +
                    cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            ret, corners_2 = cv2.findChessboardCorners(
                image_2,
                (self.num_checker_hor - 1, self.num_checker_ver - 1), 
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH +
                    cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if corners_1 is not None and corners_2 is not None :
                homography, status = cv2.findHomography(corners_1, corners_2)

                self.parent.configs_n_vals["homography"] = homography
        except Exception as e :
            print(e)
        if self.parent.configs_n_vals["homography"] is not None :
            print(self.parent.configs_n_vals["homography"])
            self.close()

if __name__ == "__main__" :

    resource_root_path = "/".join(
        os.path.abspath(
            os.path.dirname(sys.argv[0])
        ).split("/")[:-1]
    ) + "/"

    class FakeClass :
        def __init__(self) -> None:
            self.configs_n_vals = {
                "image_data" : {
                    "color_1" : np.zeros((100, 100, 3), dtype=np.uint8),
                    "color_2" : np.zeros((100, 100, 3), dtype=np.uint8)
                },
                "homography" : None
            }

    fake_objs = FakeClass()

    app = QtWidgets.QApplication(sys.argv)
    w = CalibrationWindow(
        (9, 5),
        fake_objs
    )


    w.show()
    sys.exit(app.exec_())
