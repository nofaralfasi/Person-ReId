from tkinter import *
import cv2
import numpy as np

objectToDetect = int(input("Which object?\n0: lowerbody\n1: upperbody\n2: cat face\n3: full body\n"))
objectsList = ['haarcascade_lowerbody.xml', 'haarcascade_upperbody.xml', 'haarcascade_frontalcatface.xml', 'haarcascade_fullbody.xml']

object_cascade = cv2.CascadeClassifier('haarcascades/' + objectsList[objectToDetect])

cap = cv2.VideoCapture('walking.avi')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    faces = object_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) #draw rect around face (img, starting point, ending point, color, line width)
        roi_gray = gray[y:y+h, x:x+w]  # [starting point:ending point, starting point: ending point]
        roi_color = gray[y:y+h, x:x+w]  # reimpose color

    cv2.imshow('img',img)  # show frame
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()