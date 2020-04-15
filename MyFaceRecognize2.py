""" 
Usage: 
MyFaceRecognize2.py -e <train_file_encodings> -n <train_file_names> -i <test_image> 

Options: 
-h, --help					 Show this help 
-e, --train_file_encodings =<train_file_encodings> Training file 
-n, --train_file_names =<train_file_names> Training file 
-i, --test_image =<test_image> Test image 
"""

# importing libraries 
import face_recognition 
import docopt 
from PIL import Image, ImageDraw
import numpy as np
from joblib import dump, load
import os 
import datetime

def face_recognize(file_encodings, file_names, test_image): 
	known_face_encodings = load(file_encodings) 
	known_face_names = load(file_names) 

	# Load the test image with unknown faces into a numpy array 
	unknown_image = face_recognition.load_image_file(test_image) 
	
	# Find all the faces and face encodings in the unknown image
	face_locations = face_recognition.face_locations(unknown_image)
	face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

	# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
	# See http://pillow.readthedocs.io/ for more about PIL/Pillow
	pil_image = Image.fromarray(unknown_image)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)

	# Find all the faces in the test image using the default HOG-based model 
	no = len(face_locations) 
	print("Number of faces detected: ", no) 
	
	# Predict all the faces in the test image using the trained classifier 
	a = datetime.datetime.now() 
	# Loop through each face found in the unknown image
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)
		name = "Unknown"
		
		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]
			print("Found:")
			print(name)
		else:
			print("Not Found")
	
		# Draw a box around the face using the Pillow module
		#draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))

		# Draw a label with a name below the face
		#text_width, text_height = draw.textsize(name)
		#draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(0, 0, 255))
		#draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
	# Remove the drawing library from memory as per the Pillow docs
	#del draw
	b = datetime.datetime.now()
	delta = b - a
	print(delta.microseconds)
	
	# Display the resulting image
	#pil_image.show()

	# You can also save a copy of the new image to disk if you want by uncommenting this line
	# pil_image.save("image_with_boxes.jpg")
	
	#for i in range(1): 
	#	test_image_enc = face_recognition.face_encodings(test_image)[i] 
	#	matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
	#	name = clf.predict([test_image_enc]) 
	#	print(*name) 

def main(): 
	args = docopt.docopt(__doc__) 
	train_file_encodings = args["--train_file_encodings"] 
	train_file_names = args["--train_file_names"] 
	test_image = args["--test_image"] 
	face_recognize(train_file_encodings, train_file_names, test_image) 

if __name__=="__main__": 
	main() 
