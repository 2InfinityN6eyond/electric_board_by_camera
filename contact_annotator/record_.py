import os
import sys
import time
import cv2
import pickle
import mediapipe as mp

from utils import *

def indexIterator(length) :
    idx = 0
    while True :
        yield idx
        idx += 1
        if idx == length :
            idx = 0
index_iterator = indexIterator(5)

label_types = [
    "100000", # thumb
    "010000", # index
    "001000", # middle
    "000100", # ring,
    "000010", # little
    "000001", # palm
]

contact_parts = {
    "thumb" : 100000, 
    "index" : 10000, 
    "middle" : 1000, 
    "ring" : 100, 
    "little" : 10, 
    "palm" : 1, 
    "none" : 0,
} 

if len(sys.argv) < 2 :
    print("usage : ")
    exit(-1)


data_label = 0
for contact_part in sys.argv[1:] :
    if contact_part not in contact_parts :
        print("invalid contact part")
        exit(-1)
    data_label += contact_parts[contact_part]

data_label = str(data_label)
data_label = "0" * (6 - len(data_label)) + data_label
print(data_label)

os.makedirs(data_label, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

record_flag = -1

camera_direction = 0
with mp_hands.Hands(
    static_image_mode = True,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as hands:
    while cap.isOpened():
    
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if camera_direction == 0 :
            image = image[:360, :640] # 1 0
                                      # 0 0
        if camera_direction == 1 :
            image = image[:360, 640:] # 0 1
                                      # 0 0
        if camera_direction == 2 :
            image = image[360:, :640] # 0 0
                                      # 1 0
        if camera_direction == 3 :
            image = image[360:, 640:] # 0 0 
                                      # 0 1
        
       
       
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bbox_width, bbox_height = 0, 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                bbox = calc_bounding_rect(image=image, landmarks=hand_landmarks)
                landmark_list = calc_landmark_list(image=image, landmarks=hand_landmarks)

                bbox_l, bbox_t, bbox_r, bbox_b = bbox
                
                bbox_height = int((bbox_b - bbox_t) * 0.7)
                bbox_width  = int((bbox_r - bbox_l) * 0.7)

                bbox_size = max(bbox_height, bbox_width)
                bbox_width = bbox_size
                bbox_height = bbox_size

                bbox_middle_y = (bbox_t + bbox_b) // 2
                bbox_middle_x = (bbox_l + bbox_r) // 2

                bbox_t = max(bbox_middle_y - bbox_height, 0)
                bbox_l = max(bbox_middle_x - bbox_width, 0)
                bbox_b = min(bbox_middle_y + bbox_height, image.shape[0]-1)
                bbox_r = min(bbox_middle_x + bbox_width, image.shape[1]-1)

                bbox_width *= 2
                bbox_height *= 2

           
                if record_flag > 0 :
                    image_name = os.path.join(
                        ".",
                        data_label,
                        f"{time.time()}_{data_label}.png",
                    )
                    print(
                        next(index_iterator),
                        "{0:3d} {1:3d}".format(
                            bbox_width, bbox_height
                        ),
                        image_name
                    )

                    cv2.imwrite(
                        image_name,
                        noram
                    )
        

        if results.multi_hand_landmarks:
            #cv2.imshow("cropped", image[bbox_t:bbox_b , bbox_l:bbox_r])
            cropped = image[bbox_t:bbox_b , bbox_l:bbox_r].copy()
            draw_landmarks(image, landmark_list, 1)
            image = np.hstack(
                (
                    image,
                    cv2.resize(cropped, (image.shape[0], image.shape[0]))
                )
            )
        cv2.imshow('MediaPipe Hands', image)
        key = cv2.waitKey(5)
        if key & 0xFF == 27 :
            break
        if key == ord(" ") :
            record_flag *= -1
        
        if key == ord("0") :
            camera_direction = 0
        if key == ord("1") :
            camera_direction = 1
        if key == ord("2") :
            camera_direction = 2
        if key == ord("3") :
            camera_direction = 3


    cap.release()