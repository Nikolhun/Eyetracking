import numpy as np
import cv2 as cv

coordinate_x = 80
coordinate_y = 100
interpolation_size = (400, 100)
mask_for_eyetracking = np.zeros((interpolation_size[1], interpolation_size[0]), np.uint8) + 255
mask_for_eyetracking_bgr = cv.cvtColor(mask_for_eyetracking, cv.COLOR_GRAY2BGR)


mask_for_eyetracking_bgr[np.abs(coordinate_x-(interpolation_size[1]-1))][coordinate_y][0] = 0
mask_for_eyetracking_bgr[np.abs(coordinate_x-(interpolation_size[1]-1))][coordinate_y][1] = 0
#mask_for_eyetracking_bgr[50][100][1] = 0
#mask_for_eyetracking_bgr[50][100][0] = 0
#mask_for_eyetracking_bgr[50][100][2] = 0

cv.imshow("mask add pixels", mask_for_eyetracking_bgr)

k = cv.waitKey(0)

