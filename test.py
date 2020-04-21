import cv2
import dlib
from dlib_landmarks import draw_point, eye_center_dlib, landmarks_array, fill_frame, crop_eyes
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import detect_corneal_reflection
from vector import find_vector
from calibration import upper_left, upper_right, lower_left, lower_right
import keyboard

print("Set threshold for left and right eye.")
print("Start and end vector mode by pressing v.")
print('For starting calibration mode. Look into upper right corner and press 1.')

#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("Dlib_landmarks/shape_predictor_68_face_landmarks.dat")

# ----------------------- Parametres for Blob Detection ------------------------------------------------------------- #
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True  # activates filtering by area
detector_params.filterByCircularity = 1  # activates filtering by circularity
detector_params.minCircularity = 0.5  # min circularity (0.75)
detector_params.maxCircularity = 1  # max circularity
detector_params.maxArea = 5000  # max area (1800)
detector_blob = cv2.SimpleBlobDetector_create(detector_params)  # saving parametres into detector


#######################################################################################################################
# ------------------------------- Creating trackbar ----------------------------------------------------------------- #
#######################################################################################################################
def nothing(x):
    '''
    creating Trackbar
    '''
    pass


#######################################################################################################################
# ---------------------------------- Main --------------------------------------------------------------------------- #
#######################################################################################################################
def main():
    cap = cv2.VideoCapture(0)  # reaching the port 0 for video capture
    cv2.namedWindow('Dlib Landmarks')  # Dlib landmark left eye
    cv2.createTrackbar('Right', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    cv2.createTrackbar('Left', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    left_center_pupil_in_eye_frame = [0, 0]
    right_center_pupil_in_eye_frame = [0, 0]
    output_vector_in_eye_frame = [0, 0, 0]
    left_eye_crop = [0, 0]
    right_eye_crop = [0, 0]
    min_left = [0, 0]
    min_right = [0, 0]
    vector_mode = False
    send_calibration_data_state = True
    upper_left_corner = upper_right_corner = lower_left_corner = lower_right_corner = [0, 0, 0]
    press_1 = False
    press_2 = False
    press_3 = False
    press_4 = False
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ---------------------------------- Dlib Landmark face detection --------------------------------------------------- #
        faces = detector_dlib(gray)
        for face in faces:
            #view_face_frame(face, frame)  # view face frame
            landmarks = predictor_dlib(gray, face)  # detect face structures using landmarks

            # crop eyes from the video
            left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41, landmarks, gray, lines=0)
            right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47, landmarks, gray, lines=0)

            eye_fill = fill_frame(gray, left_landmarks_array, right_landmarks_array)  # black mask with just eyes
            #cv2.imshow("eye_fill", eye_fill)

            # crop eyes from black rectangle
            left_eye_crop, min_left = crop_eyes(eye_fill, left_landmarks_array)
            right_eye_crop, min_right = crop_eyes(eye_fill, right_landmarks_array)

            # draw points into eyes
            for i in range(36, 48):
                draw_point(i, landmarks, frame)

            # draw points into face
            # for i in range(0, 67):
            #   draw_point(i, landmarks, frame)

# ---------------------------------- Left eye ---------------------------------------------------------------------- #
            threshold_left = cv2.getTrackbarPos('Right', 'Dlib Landmarks')  # getting position of the trackbar
            no_reflex_left = detect_corneal_reflection(left_eye_crop, threshold_left)
            hsv_img_left = converting_gray_to_hsv(no_reflex_left)
            filtrated_img_left = filtration(hsv_img_left)
            gama_corrected_left = gama_correction(filtrated_img_left, 1.2)
            eye_preprocessed_left = preprocessing(gama_corrected_left, threshold_left)
            cv2.imshow("eye_processed_left", eye_preprocessed_left)
            contours_left = contours_of_shape(eye_preprocessed_left, threshold_left)
            if contours_left is not None:
                for c_left in contours_left:
                    if c_left is not None:
                        m_left = cv2.moments(c_left)
                        if m_left["m00"] > 1:
                            cx_left = int(m_left["m10"] / m_left["m00"])
                            cy_left = int(m_left["m01"] / m_left["m00"])
                            # cv2.drawContours(eye_no_eyebrows_left, [c], -1, (0, 255, 0), 2)
                            #cv2.circle(left_eye_crop, (cx_left, cy_left), 1, (0, 0, 255), 2)
                            left_center_pupil = [cx_left + min_left[0], cy_left + min_left[1]]
                            left_center_pupil_in_eye_frame = [cx_left, cy_left]
                            cv2.circle(frame, (left_center_pupil[0], left_center_pupil[1]), 1, (255, 0, 0), 2)

# ---------------------------------- Left eye ---------------------------------------------------------------------- #
            threshold_right = cv2.getTrackbarPos('Left', 'Dlib Landmarks')  # getting position of the trackbar
            no_reflex_right = detect_corneal_reflection(right_eye_crop, threshold_right)
            hsv_img_right = converting_gray_to_hsv(no_reflex_right)
            filtrated_img_right = filtration(hsv_img_right)
            gama_corrected_right = gama_correction(filtrated_img_right, 1.2)
            eye_preprocessed_right = preprocessing(gama_corrected_right, threshold_right)
            cv2.imshow("preprocessing_right", eye_preprocessed_right)
            contours_right = contours_of_shape(eye_preprocessed_right, threshold_right)
            if contours_right is not None:
                for c_right in contours_right:
                    if c_right is not None:
                        m_right = cv2.moments(c_right)
                        if m_right["m00"] > 1:
                            cx_right = int(m_right["m10"] / m_right["m00"])
                            cy_right = int(m_right["m01"] / m_right["m00"])

                            #cv2.drawContours(eye_no_eyebrows_right, [c], -1, (0, 255, 0), 2)
                            #cv2.circle(right_eye_crop, (cx_right, cy_right), 1, (0, 0, 255), 2)
                            #cv2.imshow("right", right_eye_crop)

                            right_center_pupil = [cx_right + min_right[0], cy_right + min_right[1]]
                            right_center_pupil_in_eye_frame = [cx_right, cy_right]
                            cv2.circle(frame, (right_center_pupil[0], right_center_pupil[1]), 1,
                                       (255, 0, 0), 2)

# ---------------------------------- Quit program after pressing q -------------------------------------------------- #

        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" means close the detection
            break

# ---------------------------------- Show vector after pressing v --------------------------------------------------- #

        if keyboard.is_pressed("v"):  # "q" means close the detection
            if vector_mode == True:
                vector_mode = False
                print("Vector mode deactivated.")
            else:
                vector_mode = True
                print("Vector mode activated.")
        if vector_mode:
            # finding eye center
            left_center_eye, left_center_eye_in_eye_frame = eye_center_dlib(left_eye_crop, [min_left[0], min_left[1]])
            right_center_eye, right_center_eye_in_eye_frame = eye_center_dlib(right_eye_crop, [min_right[0],
                                                                                               min_right[1]])

            # finding vector
            output_vector_in_eye_frame = find_vector(left_center_pupil_in_eye_frame, left_center_eye_in_eye_frame,
                                                     right_center_pupil_in_eye_frame, right_center_eye_in_eye_frame)

            # start of vector
            start_left = (left_center_eye[0], left_center_eye[1])
            start_right = (right_center_eye[0], right_center_eye[1])

            # end of vector
            end_left = (output_vector_in_eye_frame[0] + left_center_eye[0],
                    output_vector_in_eye_frame[1] + left_center_eye[1])
            end_right = (output_vector_in_eye_frame[0] + right_center_eye[0],
                     output_vector_in_eye_frame[1] + right_center_eye[1])

            if end_left == [0, 0] or end_right == [0, 0]:
                print("Pupil not detected. Try to adjust threshold better and press v again..")
            if end_left is not None and end_right is not None:
                if output_vector_in_eye_frame[2] > 1:
                    cv2.arrowedLine(frame, start_left, end_left, color=(0, 255, 0), thickness=1)
                    cv2.arrowedLine(frame, start_right, end_right, color=(0, 255, 0), thickness=1)

# ---------------------------------- Start calibration after pressing c --------------------------------------------- #

        #if keyboard.is_pressed("c"):  # "c" means start the calibration
            #if calibration_mode == True:
                #calibration_mode = False
                #print("Calibration mode deactivated.")
            #else:
                #calibration_mode = True
                #print("Calibration mode activated.")
            #upper_left_corner = upper_right_corner = lower_left_corner = lower_right_corner = [0, 0]

        if keyboard.is_pressed("1") and press_1 == False:
            press_1 = True
            upper_left_corner = upper_left(output_vector_in_eye_frame)
            print("Upper left corner saved.")
            print('Look into upper right corner and press 2.')
        if keyboard.is_pressed("2") and press_2 == False:
            press_2 = True
            upper_right_corner = upper_right(output_vector_in_eye_frame)
            print("Upper right corner saved.")
            print('Look into lower right corner and press 3.')
        if keyboard.is_pressed("3") and press_3 == False:
            press_3 = True
            lower_right_corner = lower_right(output_vector_in_eye_frame)
            print("Lower right corner saved.")
            print('Look into lower left corner and press 4.')
        if keyboard.is_pressed("4") and press_4 == False:
            press_4 = True
            lower_left_corner = lower_left(output_vector_in_eye_frame)
            print("Lower left corner saved.")

        if keyboard.is_pressed("d"):
            print("Measured data from corners deleted.")
            press_1 = False
            upper_left_corner = [0, 0, 0]
            press_2 = False
            upper_right_corner = [0, 0, 0]
            press_3 = False
            lower_left_corner = [0, 0, 0]
            press_4 = False
            lower_right_corner = [0, 0, 0]

        if upper_left_corner != [0, 0, 0] and upper_right_corner != [0, 0, 0] and \
            lower_left_corner != [0, 0, 0] and lower_right_corner != [0, 0, 0] and send_calibration_data_state == True\
                and keyboard.is_pressed("enter"):
            send_calibration_data_state = False
            print("Data for calibration were taken.")
            print("Calibration starts...")
            print("upper left corner", upper_left_corner)
            print("upper right corner", upper_right_corner)
            print("lower left corner", lower_left_corner)
            print("lower right corner", lower_right_corner)
            # tady interpolace
            print("Calibration done successfully.")

        cv2.imshow('Dlib Landmarks', frame)  # visualization of detection
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
