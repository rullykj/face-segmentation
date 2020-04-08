from PIL import Image
from imutils import paths
import imutils
import os
import cv2


imagePaths = list(paths.list_images("DataSet"))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(cascade, test_image, scaleFactor = 1.1):
	#create a copy of the image to prevent any changes to the original one.
	image_copy = test_image.copy()
	
	#convert the test image to gray scale as opencv face detector expects gray images
	gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
	
	#Applying the haar classifier to detect faces
	faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
	
	#for (x, y, w, h) in faces_rect:
	#cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
	for (x,y,w,h) in faces_rect:
		#image_copy = cv2.rectangle(image_copy,(x,y),(x+w,y+h),(255,0,0),2)
		#roi_gray = gray_image[y:y+h, x:x+w]
		#roi_color = image_copy[y:y+h, x:x+w]
		#eyes = eye_cascade.detectMultiScale(roi_gray)
		#for (ex,ey,ew,eh) in eyes:
		#    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		test_image = test_image[y-25:y+h+25, x-25:x+w+25]

	return test_image

for (i, imagePath) in enumerate(imagePaths):
	#extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	#image = face_recognition.load_image_file(imagePath)
	
	#load the image, resize it to have a width of 600 pixels (while
	#maintaining the aspect ratio), and then grab the image
	#dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	face_image = detect_faces(face_cascade, image);
	#myPath = "DataSetCropped\\"+name+"\\"
	myPath = "DataSetCropped\\"
	num_files = len([f for f in os.listdir(myPath)if os.path.isfile(os.path.join(myPath, f))])
	try:
		pil_image = Image.fromarray(face_image)
		pil_image.save(myPath+str(num_files+1)+".jpg")
	except:
		cv2.imwrite(myPath+str(num_files+1)+".jpg",gray_image)
		print(myPath+str(num_files+1)+" error") 
	
	#pil_image.show()
	
	
	
	# construct a blob from the image
	#imageBlob = cv2.dnn.blobFromImage(
	#	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	#	(104.0, 177.0, 123.0), swapRB=False, crop=False)
	
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	#detector.setInput(imageBlob)
	#detections = detector.forward()
	
	#face_locations = face_recognition.face_locations(image)
	# Applying the haar classifier to detect faces
	#faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
	
	#for face_location in face_locations:
	#
	#	# Print the location of each face in this image
	#	top, right, bottom, left = face_location
	#	print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
	#
	#	# You can access the actual face itself like this:
	#	face_image = image[top-25:bottom+25, left-25:right+25]
	#	pil_image = Image.fromarray(face_image)
	#	#pil_image.show()
	#	myPath = "DataSetCropped\\"+name+"\\"
	#	num_files = len([f for f in os.listdir(myPath)if os.path.isfile(os.path.join(myPath, f))])
	#	pil_image.save(myPath+str(num_files+1)+".jpg")