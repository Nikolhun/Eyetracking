import cv2
import numpy as np

#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################

# ----------------------------- Haar features ----------------------------------------------------------------------- #
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # haar cascade classifiers for face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # haar cascade classifiers for eye

# ----------------------- Parametres for Blob Detection ------------------------------------------------------------- #
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True  # activates filtering by area
detector_params.filterByCircularity = 1  # activates filtering by circularity
detector_params.minCircularity = 0.75  # min circularity
detector_params.maxCircularity = 1  # max circularity
detector_params.maxArea = 1800  # max area
detector = cv2.SimpleBlobDetector_create(detector_params)  # saving parametres into detector

#######################################################################################################################
# ------------------------------- Functions ------------------------------------------------------------------------- #
#######################################################################################################################
def detect_faces(img, cascade):
    # detects face from frame
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting from rgb to gray
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)  # getting coordinated of face
    if len(coords) > 1:  # getting the biggest face
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]  # crop the image
    # cv2.imshow('detect_faces', img)  # show face detection
    return frame


def detect_eyes(img, cascade):
    # detects eyes from face
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting from rgb to gray
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # getting coordinates of eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        hh = h / 2  # getting half of the picture
        if y + hh < height / 2:  # eyes are in the upper part
            eyecenter = x + w / 2  # get the eye center
            if eyecenter < width * 0.5:  # right eye is on the left side of the face
                right_eye = img[y:y + h, x:x + w]
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # left eye is on the right side of the face
                left_eye = img[y:y + h, x:x + w]
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imshow('detect_eyes', img)  # show eye detection
    return left_eye, right_eye


def cut_eyebrows(img):
    # cuts eyebrows from eyes
    height, width = img.shape[:2]  # get height and width of the picture
    eyebrow_h = int(height / 4) # get height of eyebrows
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows
    return img


def blob_process(img, threshold, detector):
    # detects pupil from eyes 
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting from rgb to gray
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)  # gray to binary image with threshold
    # cv2.imshow('blob_before', img)  # show situation before blob process
    img = cv2.erode(img, None, iterations=2)  # erode picture
    img = cv2.dilate(img, None, iterations=4)  # dilate picture
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)  # morphology opening
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, None)  # morphology closing
    img = cv2.medianBlur(img, 5)  # median filtration
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian filtration
    # cv2.imshow('blob_after', img)  # show after blob process
    keypoints = detector.detect(img)  # getting keypoints of eye
    return keypoints


def nothing(x):
    # for creating Trackbar
    pass


#######################################################################################################################
# ---------------------------------- Main --------------------------------------------------------------------------- #
#######################################################################################################################

def main():
    cap = cv2.VideoCapture(0)  # reaching the port 0 for video capture
    cv2.namedWindow('image')  # creating frame for results
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)  # threshold track bar
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        face_frame = detect_faces(frame, face_cascade)  # function for face detection
        if face_frame is not None:  # if the face is detected
            eyes = detect_eyes(face_frame, eye_cascade)  # function for eye detection
            for eye in eyes:  # for every eye in eyes
                if eye is not None:  # if eye is detected
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')  # getting position of the trackbar
                    eye = cut_eyebrows(eye)  # cutting eyebrows
                    keypoints = blob_process(eye, threshold, detector)  # detecting pupil
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # drawig red circle into image
                    print(eye)
        cv2.imshow('image', frame)  # visualization of detection
        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" means close the detection
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
