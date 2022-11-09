import os
import sys
import time
import cv2
import pickle
import mediapipe as mp
import torch
import pickle 

from utils import *
import utils, visualization

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


class point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
   		return f"({self.x}, {self.y})"

class Config:
    img_height = 720
    img_width = 1280
    img_channel = 3
    img_shape = (img_height, img_width, img_channel)

    box_margin = 100
    box_alpha = 0.7
    model_config = {
    'model_complexity': 1,
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3
    }

def get_landmark_array(landmarks, image_shape, normal_x, normal_y):

    image_width, image_height = image_shape[1], image_shape[0]
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = int(landmark.x * image_width) + normal_x
        landmark_y = int(landmark.y * image_height) + normal_y
        landmark_list.append((landmark_x, landmark_y))
    return np.array(landmark_list)

def get_bounding_rect(landmark_array, p1, p2):

    x, y, w, h = cv2.boundingRect(landmark_array)
    p1.x = int(Config.box_alpha * p1.x + (1 - Config.box_alpha) * x)
    p1.y = int(Config.box_alpha * p1.y + (1 - Config.box_alpha) * y)
    p2.x = int(Config.box_alpha * p2.x + (1 - Config.box_alpha) * (x + w))
    p2.y = int(Config.box_alpha * p2.y + (1 - Config.box_alpha) * (y + h))

    return p1, p2

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

last_frame_failed = False

camera_direction = 0
with mp_hands.Hands(
    static_image_mode = True,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as hands:


    p1, p2 = point(0, 0), point(Config.img_width - 1, Config.img_height - 1)

    while cap.isOpened():
        success, image = cap.read()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        normal_x, normal_y = max(p1.x - Config.box_margin, 0), max(p1.y - Config.box_margin, 0)
        cropped_image = image[normal_y:p2.y + Config.box_margin, normal_x:p2.x + Config.box_margin, ::]

        results = hands.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmark_array = get_landmark_array(hand_landmarks, cropped_image.shape, normal_x, normal_y)
                p1, p2 = get_bounding_rect(landmark_array, p1, p2)

                if not last_frame_failed :
                    rotated_normalized_img, rotated_normalized_landmark = mediapipe_out_to_blazehand_in2(
                        image,
                        landmark_array,
                        (p1.x, p1.y, p2.x, p2.y)
                    )
                    
            
                    if record_flag > 0 :
                        image_name = os.path.join(
                            ".",
                            data_label,
                            f"{time.time()}_{data_label}.png",
                        )
                        print(
                            next(index_iterator),
                            "{0:3d} {0:3d}".format(
                                rotated_normalized_img.shape[0]
                            ),
                            image_name
                        )
                        cv2.imwrite(
                            image_name,
                            rotated_normalized_img,
                        )
                        with open(image_name.replace("png", "pkl"), "wb") as fp :
                            pickle.dump(
                                torch.Tensor(rotated_normalized_landmark),
                                fp
                            )


                    visualization.draw_normalized_hand_landmarks_on_cropped(
                        rotated_normalized_img,
                        torch.Tensor(rotated_normalized_landmark)
                    )

            last_frame_failed = False

        else:
            p1, p2 = point(0, 0), point(Config.img_width - 1, Config.img_height - 1)
            last_frame_failed = True

        if results.multi_hand_landmarks:
            try :
                draw_landmarks(image, landmark_array, 1)
                image = np.hstack(
                (
                        image,
                        cv2.resize(rotated_normalized_img, (image.shape[0], image.shape[0]))
                    )
                )        
            except : 
                pass


        cv2.imshow(
            'MediaPipe Hands',
            cv2.resize(
                image,
                (image.shape[1] // 2, image.shape[0] // 2)
            )
        )
        key = cv2.waitKey(5)
        if key & 0xFF == 27 :
            break
        if key == ord(" ") :
            record_flag *= -1
        

    cap.release()