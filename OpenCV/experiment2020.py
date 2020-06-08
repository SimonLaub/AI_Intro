import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# https://github.com/opencv/opencv/tree/master/data/haarcascades

#To download them, right click “Raw” => “Save link as”. Make sure they are in your working directory.

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')

while True:
    _, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        gray_face = gray_frame[y:y + h, x:x + w]  # cut the gray face frame out
        face = frame[y:y + h, x:x + w]  # cut the face frame out
        eyes = eye_cascade.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            # ToDo make code that keeps just two eyes with similar y values
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 2)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
