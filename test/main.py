import time
import numpy as np
import cv2


from multiprocessing import Process, Queue, Pipe, Value, shared_memory


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
IMAGE_FPS = 30

if __name__ == "__main__" :

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH,)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, IMAGE_FPS)


    # init multi processes


    while True :
        pass