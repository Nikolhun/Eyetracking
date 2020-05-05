import cv2
import numpy as np
import imutils

def converting_gray_to_hsv(img):
    '''
    Convert gray frame into hsv frame.
    :param img: image in gray frame you want to convert
    :return: hsv frame
    '''
    color_frame_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # converting from rgb to hsv
    hsv_img = cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2HSV)  # converting from rgb to hsv
    return hsv_img


def filtration(img):
    '''
    Bilateral filtration, median filtration, gaussian filtration.
    :param img: image in gray frame
    :return: filtrated image in gray frame
    '''
    img_median = cv2.medianBlur(img, 5)  # median filtration
    filtrated_img = cv2.bilateralFilter(img_median, 9, 75, 75)  # bilateral filtration
    #filtrated_img = cv2.GaussianBlur(img_median, (5, 5), 0)  # Gaussian filtration
    return filtrated_img


def gama_correction(img, gamma):
    '''
    Function makes gamma correction acording to picture 'img' according to the parameter 'gamma'.
    :param img: picture in GRAY
    :param gamma: gamma parameter
    :return: gamma corrected picture in GRAY
    '''
    # Apply gamma correction.
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return gamma_corrected


def preprocessing(img, threshold):
    '''
    Preprocessing picture: erode, dilate, morphology opening, morphology closing.
    :param img: image in gray frame or hsv frame
    :param threshold: threshold for thresholding image into binary image
    :return: preprocessed_image, threshold
    '''
    _, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # gray to binary image with threshold
    img_erode = cv2.erode(img_binary, None, iterations=1)  # erode picture
    img_dilate = cv2.dilate(img_erode, None, iterations=1)  # dilate picture
    img_open = cv2.morphologyEx(img_dilate, cv2.MORPH_OPEN, None)  # morphology opening
    preprocessed_img = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, None)  # morphology closing
    return preprocessed_img


def contours_of_shape(img, threshold):
    '''
    Finds contours of shape.
    :param img: HSV image
    :param threshold: threshold from trackbar
    :return: file of contours of shape
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    return cnts

def blob_process(img, detector_blob):
    '''
    Function for blob process
    :param img: image in gray
    :param detector_blob: detector iniciated in iniciation part
    :return: kepoints of blob
    '''
    keypoints = detector_blob.detect(img)  # getting keypoints of eye
    return keypoints