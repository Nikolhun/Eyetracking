from PIL import Image, ImageTk
import numpy as np
import dlib
import cv2
from tkinter import *
from dlib_landmarks import draw_point, eye_center_dlib, landmarks_array, fill_frame_gui, crop_eyes
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import detect_corneal_reflection
from vector import find_vector
from calibration import upper_left, upper_right, lower_left, lower_right

global cam
cam = cv2.VideoCapture(0)

detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("Dlib_landmarks/shape_predictor_68_face_landmarks.dat")
#output_vector_in_eye_frame = [0, 0, 0]

main_window = Tk()
main_window.title("Eyetracking")
main_window.geometry("1280x768")
#screen_width, screen_height = (main_window.winfo_screenheight(), main_window.winfo_screenwidth())
screen_width, screen_height = (1280, 768)

# Create a frame
app = Frame(main_window, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()


def stream_video():
    while cam.isOpened():
        _, frame_to_flip = cam.read()  # convert cap to matrix for future work
        frame = cv2.flip(frame_to_flip, 1)  # flip video to not be mirrored
        cv2.imshow("test", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector_dlib(gray)
        for face in faces:
            # view_face_frame(face, frame)  # view face frame
            landmarks = predictor_dlib(gray, face)  # detect face structures using landmarks

            # draw points into eyes
            for i in range(36, 48):
                draw_point(i, landmarks, frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        print(cv2image)
        print(cv2image.shape)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, stream_video)

            # draw points into face
            # for i in range(0, 67):
            #   draw_point(i, landmarks, frame)

            # crop eyes from the video
            #left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41, landmarks, gray, lines=0)
            #right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47, landmarks, gray, lines=0)

            #eye_fill = fill_frame(gray, left_landmarks_array, right_landmarks_array)  # black mask with just eyes
            # cv2.imshow("eye_fill", eye_fill)

            # crop eyes from black rectangle
            #left_eye_crop, min_left = crop_eyes(eye_fill, left_landmarks_array)
            #right_eye_crop, min_right = crop_eyes(eye_fill, right_landmarks_array)



            # left eye
            #threshold_left = cv2.getTrackbarPos('Right', 'Dlib Landmarks')  # getting position of the trackbar

            # right eye
            #threshold_right = cv2.getTrackbarPos('Left', 'Dlib Landmarks')  # getting position of the trackbar

            # input('Set right and left eye threshold and press ENTER.')
            # finding eye center
            #left_center_eye, left_center_eye_in_eye_frame = eye_center_dlib(left_eye_crop, [min_left[0], min_left[1]])
            #right_center_eye, right_center_eye_in_eye_frame = eye_center_dlib(right_eye_crop, [min_right[0],
                                                                                            #   min_right[1]])

            # finding vector
            #output_vector_in_eye_frame = find_vector(left_center_pupil_in_eye_frame, left_center_eye_in_eye_frame,
             #                                        right_center_pupil_in_eye_frame, right_center_eye_in_eye_frame)

            # start of vector
            #start_left = (left_center_eye[0], left_center_eye[1])
            #start_right = (right_center_eye[0], right_center_eye[1])

            # end of vector
            #end_left = (output_vector_in_eye_frame[0] + left_center_eye[0],
            #            output_vector_in_eye_frame[1] + left_center_eye[1])
            #end_right = (output_vector_in_eye_frame[0] + right_center_eye[0],
            #             output_vector_in_eye_frame[1] + right_center_eye[1])

            #if end_left == [0, 0] or end_right == [0, 0]:
             #   print("Pupil not detected, set threshold in left/right trackbar.")
            #else:
             #   if end_left is not None and end_right is not None:
             #       if output_vector_in_eye_frame[2] > 1:
             #           cv2.arrowedLine(frame, start_left, end_left, color=(0, 255, 0), thickness=1)
              #          cv2.arrowedLine(frame, start_right, end_right, color=(0, 255, 0), thickness=1)
        cv2.imshow('Dlib Landmarks', frame)  # visualization of detection

        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" means close the detection
            break
    cam.release()
    cv2.destroyAllWindows()

def get_threshold_left(event):
    print(threshold_left_scale.get())


def get_threshold_right(event):
    print(threshold_right_scale.get())
vector = Button(main_window, text="Testing", command=stream_video, width=20, height=2)
vector.grid(row=0, column=3)

threshold_left_scale = Scale(main_window, label="Right eye", from_=0, to=255, length=(screen_width/2)-10, orient=HORIZONTAL, command=get_threshold_left)
threshold_left_scale.grid(row=0, column=0)

threshold_right_scale = Scale(main_window, label="Left eye", from_=0, to=255, length=(screen_width/2)-10, orient=HORIZONTAL, command=get_threshold_right)
threshold_right_scale.grid(row=0, column=1)




stream_video()
main_window.mainloop()

#if output_vector_in_eye_frame == [0, 0, 0]:
#    print("Pupil not detected, set threshold in left/right trackbar.")
#else:
#    upper_left_corner = upper_left(output_vector_in_eye_frame)
#    upper_right_corner = upper_right(output_vector_in_eye_frame)
#    lower_left_corner = lower_right(output_vector_in_eye_frame)
#    lower_right_corner = lower_left(output_vector_in_eye_frame)
#    print(lower_right_corner)
