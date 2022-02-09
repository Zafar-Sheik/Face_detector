import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in 
#img = cv2.imread('rdj.jpg')
img = cv2.imread('spidermen.png')

#make image grayscale for algorithm 
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces 
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#show coordinates of face 
print (face_coordinates)

#draw rectangles around face/s 

for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)) , 3)


#show the image 
cv2.imshow('Face/s Detected', img)

cv2.waitKey() #need wait key or image will close instantly

print ("Code Complete")



