import cv2

#rozdělit na detect a odstranit?
def detect_corneal_reflection(img, threshold_from_trackbar):
    '''
    It removes conrneal reflection.
    :param img: image
    :param threshold: threshold from trackbar in image window
    :return: image withou corneal reflection
    '''
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, threshold = cv2.threshold(l, 210, 255, cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold, None, iterations=1)  # dilate picture
    row = threshold.shape[0]
    coll = threshold.shape[1]
    #contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("threshold", threshold)
    #cv2.imshow("corneal reflection", gray_frame)
    #cv2.drawContours(gray_frame, contours, -1, (0, 0, 255), 1)
    #mask = np.zeros(3, 3)
    for i in range(0, row):
        for y in range(0, coll):
            if threshold[i, y] == 0:
                l[i, y] = l[i, y]
            elif threshold[i, y] == 255:
                if i == 0 and y == 0:
                    l[i, y] = threshold_from_trackbar/2
                elif i == 0 and y != 0:
                    l[i, y] = l[i, y - 1]
                elif y == 0 and i != 0:
                    l[i, y] = l[i - 1, y]
                else:
                    #l[i, y] = l[i - 1, y]
                    l[i, y] = 0
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray_frame