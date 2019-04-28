from collections import deque
import numpy as np
import cv2
import imutils
import time

bufferSize = 16

ObjectIndicate, TargetIndicate = False, False

colorRanges = [
        ((94, 81, 87), (141, 255, 255), "Target"),
        ((18, 156, 100), (76, 255, 255), "Object")]

pts = deque([], maxlen=bufferSize)
pts1 = deque([], maxlen=bufferSize)

x_axis, xB_axis = 0, 0

counter, counter1 = 0, 0

(dX, dY), (dXB, dYB) = (0, 0), (0, 0)

direction = ""
directionB = ""

vs = cv2.VideoCapture(0)

time.sleep(2.0)

Scene = True
while True:
    # grab the current frame
    (grabbed, frame) = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for (lower, upper, colorName) in colorRanges:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        centerTarget, centerObject = None, None
        # only proceed if at least one contour was found

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            MB = cv2.moments(c)
            centerTarget = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centerObject = (int(MB["m10"] / MB["m00"]), int(MB["m01"] / MB["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 15 and colorName == "Target":
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.putText(frame, colorName, centerTarget, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.circle(frame, centerTarget, 5, (0, 0, 255), -1)
                x_axis = int(x)
                pts.appendleft(centerTarget)
                if not TargetIndicate:
                    TargetIndicate = True
                    # print("Target Detect = ", TargetIndicate)
            elif radius < 10:
                TargetIndicate = False
                # print("Target Detect = ", TargetIndicate)

            if radius > 30 and colorName == "Object":
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                cv2.putText(frame, colorName, centerObject, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.circle(frame, centerObject, 5, (0, 0, 255), -1)
                xB_axis = int(x)
                pts1.appendleft(centerObject)
                if not ObjectIndicate:
                    ObjectIndicate = True
                    # print("Object Detect = ", ObjectIndicate)
            elif radius < 30:
                ObjectIndicate = False
                # print("Object Detect = ", ObjectIndicate)
        if TargetIndicate and ObjectIndicate and Scene:
            Scene = False
            if x_axis > xB_axis:
                print(x_axis, ">", xB_axis, "Target is on the right side of the Object")
            if x_axis < xB_axis:
                print(x_axis, "<", xB_axis, "Target is on the left side of the object")
        if not TargetIndicate and ObjectIndicate and not Scene:
            Scene = True
            print("Totally Occluded")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
