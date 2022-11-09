import cv2
from multiprocessing import Process, Queue
import os
import time

save_root_path = "C:/Users/hjp1n/OneDrive/바탕 화면"
save_image = True

class HandVisualizer(Process) :
    def __init__(
        self,
        data_queue
    ) :
        super(HandVisualizer, self).__init__()
        self.data_queue = data_queue

    def run(self) :
        cv2.namedWindow("hand_pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("hand_pose", 1920, 1080)
        
        last_saved = time.time()

        while True :
            if not self.data_queue.empty() :
                data = self.data_queue.get()

                cv2.imshow("hand_pose", data)
                if cv2.waitKey(1) == ord('q') :
                    break

                if save_image and time.time() - last_saved > 1 :
                    print("saving..")
                    cv2.imwrite(
                        os.path.join(
                            save_root_path,
                            str(int(time.time()))[-4:] + ".png"
                        ),
                        data
                    )
                    last_saved = time.time()