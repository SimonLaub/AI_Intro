import numpy as np
import cv2

# For more about detecting red see e.g.
# https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/

cap = cv2.VideoCapture(0)

lower_blue= np.array([78,158,124])
upper_blue = np.array([138,255,255])

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

lower_bluish = np.array([110,50,50])
upper_bluish = np.array([130,255,255])

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

while(True):
  # Capture frame-by-frame
   ret, frame = cap.read()

   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   mask = cv2.inRange(hsv, lower_red, upper_red)
   # Removing noise. Isolation of individual elements and joining disparate elements in an image.
   mask = cv2.erode(mask, None, iterations=2)
   mask = cv2.dilate(mask, None, iterations=2)

   contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

   if len(contours) != 0:
       #c = max(contours, key=cv2.contourArea)
       for contour in contours:
           ((x, y), radius) = cv2.minEnclosingCircle(contour)
           #((x, y), radius) = cv2.minEnclosingCircle(c)

           if radius > 10:
              cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

   # Display the resulting frame
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
