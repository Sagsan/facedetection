from random import randrange as r
import cv2

#dataset load
traineddata=cv2.CascadeClassifier('face.xml')


#start the webcam
webcam=cv2.VideoCapture(0)

while True:
    sucess,frame=webcam.read()

    #conversion to blact and white(grayscale)
    grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #DETECT FACES
    faceCorrdinates=traineddata.detectMultiScale(grayimg)

    for x,y,w,h in faceCorrdinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)


    cv2.imshow('window',frame)
    key=cv2.waitKey(1)
    if(key==65 or key==97):
        break

webcam.release()

