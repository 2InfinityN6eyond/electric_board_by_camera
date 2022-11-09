import os
import cv2

DATA_ROOT_PATH = "./data"
image_paths = os.listdir(DATA_ROOT_PATH)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
for image_path in image_paths :
    image = cv2.imread(os.path.join(DATA_ROOT_PATH, image_path))
    image = cv2.resize(image, (224,224))
    image = cv2.putText(
        image, image_path.split("_")[-1].split(".")[0],
        (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 1
    )
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q") :
        break