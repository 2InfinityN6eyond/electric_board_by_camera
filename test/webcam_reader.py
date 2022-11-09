import time
import numpy as np
import cv2


from multiprocessing import Process, Queue, Pipe, Value, shared_memory

class WebcamReader(Process) :
    def __init__(
        self,
        image_width = 1280,
        image_height = 720,
        fps = 30,
        num_shms = 3,
    ) :
        super(WebcamReader, self).__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        self.num_shms = num_shms

        self.shms

    def run(self) :
        cap = cv2.VideoCapture(0)
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, 
            self.image_width
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, self.image_height
        )
        cap.set(
            cv2.CAP_PROP_FPS, self.fps
        )

        index_iterator = self.indexIterater()
        while True :
            __start = time.time()
            ret, image = cap.read()
            if not ret :
                continue
            __end = time.time()

            

            print(next(index_iterator), __end - __start)

    def indexIterater(self) :
        idx = 0
        while True :
            yield idx
            idx += 1
            if idx == self.num_shms :
                idx = 0


if __name__ == "__main__" :
    webcam_reader = WebcamReader()
    
    webcam_reader.start()

    webcam_reader.join()