import numpy as np
import cv2

key = ''
while key != 113:
    check_img = np.full((640, 480, 4), (0, 0, 0, 0), np.uint8)
    check_img[200, 200, 0] = 255
    check_img[200, 200, 1] = 255
    check_img[200, 200, 2] = 255
    cv2.imshow("check", check_img)
    key = cv2.waitKey(5)