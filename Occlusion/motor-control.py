import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

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
    rightMotorFwd.ChangeDutyCycle(75)
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
    frontMotorRev.ChangeDutyCycle(100)
    rightMotorFwd.ChangeDutyCycle(75)
    rightMotorRev.ChangeDutyCycle(0)
    
def top_left():
    frontMotorFwd.ChangeDutyCycle(100)
    frontMotorRev.ChangeDutyCycle(0)
    rightMotorFwd.ChangeDutyCycle(0)
    rightMotorRev.ChangeDutyCycle(0)
    leftMotorFwd.ChangeDutyCycle(75)
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
    
def main():
    top_left()
    time.sleep(3)
    top_right()
    time.sleep(3)
    forward()
    time.sleep(3)
    reverse()
    time.sleep(3)
    stop()
    # print("Stop")
    GPIO.cleanup()

if __name__ == "__main__":   
    main()

