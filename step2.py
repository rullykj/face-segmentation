# ### Import Required Modules

#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

#there is no label 0 in our training data so subject name for index/label 0 is empty
#subjects = ["", "Ramiz Raja", "Elvis Presley"]
subjects = ["", "Anto S", "Gembong", "Made Gunawan", "Agung S", "Gunarso", "Asril", "Elvira", "Rachmawan", "Rully", "Teduh", "Eka Wibowo", "Budhy", "Hari Y", "Rizki", "Lyla", "Pebi", "Gita"]

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    imgTemp = cv2.resize(gray[y:y+w, x:x+h], (200, 200))
    
    #return only the face part of the image
    return imgTemp, faces[0]
    #return gray[y:y+w, x:x+h], faces[0]
    #return gray[y:200, x:200], faces[0]


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainner/trainnerLBPHFace.yml')

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer.read('trainner/trainnerEigenFace.yml')

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer.read('trainner/trainnerFisherFace.yml')


#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img, label_text



print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

#perform a prediction
predicted_img1, label_text1 = predict(test_img1)
predicted_img2, label_text2 = predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow(label_text1, cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(label_text2, cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





