import cv2
import numpy as np

while True:
    mask = np.zeros((20, 20), np.uint8) + 255
    #mask = mask[::4, ::4]
    cv2.imshow("test", mask)
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        print(k)
    # press 'q' to exit
    if k == ord('q'):
        break
    elif k == ord('e'):
        print("e")
    elif k == ord('d'):
        print("d")
    elif k == ord('1'):
        print("1")
    elif k == ord('v'):
        print("v")
    elif k == ord('9'):
        print("9")
