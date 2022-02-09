import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#To capture video from webcam 
#webcam = cv2.VideoCapture(0) # 0 to capture video from webcam. File name to capture video from file. 

#Capture from video file 
webcam = cv2.VideoCapture('funnybaby.mp4')

#Iterate forever over frames
while True:

    # Read the current frame 
    successful_frame_read , frame = webcam.read()

    #make image grayscale for algorithm 
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around face/s 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)) , 3)

    #show the image 
    cv2.imshow('Face/s Detected', frame)
    key = cv2.waitKey(1)

    ####Stop if Q key is pressed####
    if key == 81 or key == 113:
        break

##Release webcam 
webcam.release()

print ("Code Complete")