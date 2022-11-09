
import numpy as np
import cv2 as cv
import cv2


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def calc_theta(landmarks) :
    normalized_landmark_list = calc_normalized_landmark_list(landmarks)
    palm_pos = normalized_landmark_list[0][:2]
    fing_pos = normalized_landmark_list[[8, 12, 16, 20], :2].mean(axis = 0)
    return np.arctan2(*(palm_pos - fing_pos))

def calc_normalized_landmark_list(landmarks):
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(landmark.x, 0.99999)
        landmark_y = min(landmark.y, 0.9999)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return np.array(landmark_point)

def mediapipe_out_to_blazehand_in(
    image,
    landmark, # direct from mediapipe
) :
    """
    return landmark : normalized.
    """
    bbox_l, bbox_t, bbox_r, bbox_b = calc_bounding_rect(image, landmark)
    landmark_arr = np.array(
        calc_landmark_list(image, landmark)
    )

    bbox_height = int((bbox_b - bbox_t) * 1.2)
    bbox_width  = int((bbox_r - bbox_l) * 1.2)
    
    bbox_height = max(bbox_height, bbox_width)
    bbox_width  = max(bbox_height, bbox_width)

    bbox_middle_y = (bbox_t + bbox_b) // 2
    bbox_middle_x = (bbox_l + bbox_r) // 2

    bbox_t = max(bbox_middle_y - bbox_height, 0)
    bbox_l = max(bbox_middle_x - bbox_width, 0)
    bbox_b = min(bbox_middle_y + bbox_height, image.shape[0]-1)
    bbox_r = min(bbox_middle_x + bbox_width, image.shape[1]-1)

    bbox_width *= 2
    bbox_height *= 2

    rotated_image, rotated_landmark = rotate_image_and_landmark(
        image[bbox_t:bbox_b , bbox_l:bbox_r],
        landmark_arr - np.array([[bbox_l, bbox_t]]),
        -calc_theta(landmark),
        bbox_width // 2,
        bbox_height // 2
    )

    return cv2.resize(rotated_image, (256, 256)), rotated_landmark / np.array([rotated_image.shape[1], rotated_image.shape[0]])


def rotate_image_and_landmark(
    image,
    landmark,   # denormalized
    theta,      # radian
    cx,         # center x, denormalized
    cy          # center y, denormalized
) :
    rot_matrix_img = cv2.getRotationMatrix2D(
        (cx, cy), np.rad2deg(theta), 1.0
    )
    rotated_image = cv2.warpAffine(
        image, rot_matrix_img, (image.shape[1], image.shape[0])
    )
    
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta),  np.cos(-theta)]])
    o = np.atleast_2d((cx, cy))
    return rotated_image, np.squeeze((R @ (landmark.T-o.T) + o.T).T)


def draw_landmarks(
    image,
    landmark_point,
    line_thickness = 1,
    vertex_radius = 2,
    fingertip_point_radius = 3,
    
):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), line_thickness)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), line_thickness)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), line_thickness)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), line_thickness)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), line_thickness)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), line_thickness)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), line_thickness * 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), line_thickness)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), vertex_radius, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), fingertip_point_radius, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


# https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
