import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

WINDOW='test'
cv2.namedWindow(WINDOW)

frame = cv2.imread("/Users/hjp/Desktop/hand.png")

img1, img2, scale, pad = resize_pad(frame)

normalized_palm_detections = palm_detector.predict_on_image(img1)
palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)

for i in range(len(flags2)):
    landmark, flag = landmarks2[i], flags2[i]
    if flag>.5:
        draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
        pass
#draw_roi(frame, box2)
#draw_detections(frame, palm_detections)

cv2.imshow(
    WINDOW,
    cv2.cvtColor(frame[:,:,::-1], cv2.COLOR_BGR2RGB)
)

key = cv2.waitKey(-1)
if key == 27:
    pass
cv2.destroyAllWindows()