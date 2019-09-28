import cv2
import numpy as np

cv2.namedWindow('Camera Output')
cv2.namedWindow('Camera Output2')

videoFrame = cv2.VideoCapture(0)
check=0
saveimg=[]

while(True):

    readSucsess, sourceImage = videoFrame.read()
    sourceImage = cv2.flip(sourceImage,1)

    if check<10:
        check+=1

    elif check==10:
        saveimg = sourceImage
        check+=1

    else:
        gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

        image_hsv = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2HSV)

        yellowscale = cv2.inRange(image_hsv, (30, 80, 120), (70, 120, 200)) #HSV range of COLOR(YELLOW)

        contours, hierarchy = cv2.findContours(yellowscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            sourceImage[y:y+h, x:x+w] = saveimg[y:y+h, x:x+w]

        cv2.imshow('Camera Output', yellowscale)
        cv2.imshow('Camera Output2', sourceImage)

        cv2.moveWindow('Camera Output', 40, 30)
        k = cv2.waitKey(1)
        if k == 27:
            break

cv2.destroyWindow('Camera Output')
videoFrame.release()
