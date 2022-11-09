import cv2
import numpy as np
import sys 
import random
from PyQt5 import QtWidgets, QtCore, QtGui

class ImagePlotter(QtWidgets.QLabel) :
    
    def __init__(
        self,
        width = 640,
        height = 480
    ) :
        super(ImagePlotter, self).__init__()
        self.width = width
        self.height = height
        self.resize(width, height)

    def update(self, image:np.ndarray) :
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            image.data, w, h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888
        ).scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(QtGui.QPixmap.fromImage(q_image))