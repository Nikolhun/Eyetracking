# Eyetracking
Eye movement tracking using the Raspberry Pi platform.


These ".py" scripts works with webcamera so far. 
 
File "detect_pupil_haar_cascade.py" detect pupil using Haar Cascade - you need file haarcascade_eye.xml and haarcascade_frontalface_default.xml from Haar_cascade

File "eyetracker.py" detects eye pupil and shows eye vector. There are two ways of detection eye pupil:
 
1) Haar cascade - You need file haarcascade_eye.xml and haarcascade_frontalface_default.xml from Haar_cascade
 
2) Dlib landmarks - You need file shape_predictor_68_face_landmarks.dat in Dlib_landarks





