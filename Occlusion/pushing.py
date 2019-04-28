from collections import deque
import numpy as np
import cv2
import imutils
import time

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

defaultSpeed = 100
updateSpeed = 90
windowCenter = 400
centerBuffer = 10
pwmBound = float(100)
cameraBound = float(400)
kp = pwmBound / cameraBound
leftBound = int(windowCenter - centerBuffer)
rightBound = int(windowCenter + centerBuffer)
error = 0
# Motor
leftA = 26  # L298N in2
leftB = 24  # L298N in1
rightA = 19   # L298N in3
rightB = 21   # L298N in4
frontA = 33   # second l298n in4
frontB = 31  # second l298n in3

GPIO.setup(rightA, GPIO.OUT)
GPIO.setup(rightB, GPIO.OUT)
GPIO.setup(leftA, GPIO.OUT)
GPIO.setup(leftB, GPIO.OUT)
GPIO.setup(frontA, GPIO.OUT)
GPIO.setup(frontB, GPIO.OUT)

# Disable movements on startup
GPIO.output(leftA, GPIO.LOW)
GPIO.output(leftB, GPIO.LOW)
GPIO.output(rightA, GPIO.LOW)
GPIO.output(rightB, GPIO.LOW)
GPIO.output(frontA, GPIO.LOW)
GPIO.output(frontB, GPIO.LOW)

# PWM Initialization

rightMotorFwd = GPIO.PWM(rightA, 100)
leftMotorFwd = GPIO.PWM(leftA, 100)
frontMotorFwd = GPIO.PWM(frontA, 100)
rightMotorRev = GPIO.PWM(rightB, 100)
leftMotorRev = GPIO.PWM(leftB, 100)
frontMotorRev = GPIO.PWM(frontB, 100)

rightMotorFwd.start(100)
leftMotorFwd.start(100)
frontMotorFwd.start(100)
rightMotorRev.start(100)
leftMotorRev.start(100)
frontMotorRev.start(100)

def stop():
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(0)
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(0)

def forward():
    rightMotorFwd.ChangeDutyCycle(80)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(100)
    leftMotorRev.ChangeDutyCycle(0)
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(0)
    
def reverse():
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(75)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(100)
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(0)
    
def top_right():
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(0)
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(50)
    rightMotorFwd.ChangeDutyCycle(37)
    rightMotorRev.ChangeDutyCycle(0)
    
def top_left():
    frontMotorFwd.ChangeDutyCycle(50)
    frontMotorRev.ChangeDutyCycle(0)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(37)
    leftMotorRev.ChangeDutyCycle(0)

def spin_left():
    frontMotorFwd.ChangeDutyCycle(100)
    frontMotorRev.ChangeDutyCycle(0)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(0)
    
def spin_right():
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(100)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(0)
    
def bottom_left():
    frontMotorFwd.ChangeDutyCycle(100)
    frontMotorRev.ChangeDutyCycle(0)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(100)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(0)
    
def bottom_right():
    frontMotorFwd.ChangeDutyCycle(0)
    frontMotorRev.ChangeDutyCycle(100)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(0)
    leftMotorRev.ChangeDutyCycle(100)

bufferSize = 5

ObjectIndicate, TargetIndicate = False, False

colorRanges = [
        ((22, 60, 200), (60, 255, 255), "Target"),
        ((136, 87, 111), (180, 255, 255), "Object")]

pts = deque([], maxlen=bufferSize)
pts1 = deque([], maxlen=bufferSize)

x_axis, xB_axis = 0, 0

(dX, dY), (dXB, dYB) = (0, 0), (0, 0)

direction = ""
directionB = ""

vs = cv2.VideoCapture(0)

time.sleep(2.0)
Scene = True
while True:
    # grab the current frame
    (grabbed, frame) = vs.read()
    frame = imutils.resize(frame, width=800)
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
            if radius > 20 and colorName == "Target":
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.putText(frame, colorName, centerTarget, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.circle(frame, centerTarget, 5, (0, 0, 255), -1)
                x_axis = int(x)
                pts.appendleft(centerTarget)
                if not TargetIndicate:
                    TargetIndicate = True
                    # print("Target Detect = ", TargetIndicate)
            elif radius < 20:
                TargetIndicate = False
                # print("Target Detect = ", TargetIndicate)

            if radius > 20 and colorName == "Object":
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                cv2.putText(frame, colorName, centerObject, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.circle(frame, centerObject, 5, (0, 0, 255), -1)
                xB_axis = int(x)
                radiusB = int(radius)
                pts1.appendleft(centerObject)
                if not ObjectIndicate:
                    ObjectIndicate = True
                    # print("Object Detect = ", ObjectIndicate)
            elif radius < 20:
                ObjectIndicate = False
                # print("Object Detect = ", ObjectIndicate)
        if TargetIndicate and ObjectIndicate and Scene:
            Scene = False
            if x_axis > xB_axis:
                print(x_axis, ">", xB_axis, "Target is on the right side of object")
                top_left()
                                            
            if x_axis < xB_axis:
                print(x_axis, "<", xB_axis, "Target is on left side of the object")
                top_right()
                """ if (xB_axis < leftBound) or (xB_axis > rightBound):
                    error = (windowCenter - xB_axis)/2
                    pwmOut = abs(error * kp * 0.1)
                    turnPwm = (pwmOut + defaultSpeed)
                    print("turn_pwm = ", turnPwm)
                    if turnPwm > 100:
                        turnPwm = 100
                if xB_axis < (leftBound):
                    if xB_axis < 100:
                        # print(xB_axis)
                        top_right(defaultSpeed, updateSpeed)
                    else:
                        top_right(turnPwm, defaultSpeed)
                elif xB_axis > rightBound:
                    if xB_axis > 700:
                        top_right(updateSpeed, defaultSpeed)
                    else:
                        top_right(defaultSpeed, turnPwm)
            else:
                if radiusB < 100:
                    UpdatePwm(defaultSpeed, defaultSpeed) """
                
        if not TargetIndicate and ObjectIndicate and not Scene:
            Scene = True
            print("Totally Occluded")
            forward()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    
GPIO.cleanup()
vs.release()
cv2.destroyAllWindows()

