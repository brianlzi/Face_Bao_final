'''
This project Face Recognition is used for Bao workshop.
A completed App includes 4 Threads:
    1. Labeling: Draw box in a/many faces
    2. Training: Calculate embeded vectors
    3. Testing: Show box in another image
    4. Clear : Clear path, file in Labeling
'''

import cv2
import face_recognition
import os

from PIL import Image, ImageDraw
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import threading
import tkinter
import tkinter.messagebox

DIR_PATH = 'TRAIN'
DIR_TEST = 'TEST'

def labelImg():
    os.system(" cd .. && cd labelImg && python labelImg.py")

def Training():
	'''
	solve xml to face
	'''
	known_face_encodings = []
	known_face_names = []

	if not os.path.isdir(DIR_PATH):
		os.mkdir(DIR_PATH)

	for root, dirs, files in os.walk(DIR_PATH):
		for xml_file in files:
			if xml_file[-3:] == "xml":
				tree = ET.parse(os.path.join(root,xml_file))
				root = tree.getroot()
				f_img = root.find('path').text
				img = cv2.imread(f_img)

				for obj in root.iter('object'):
					cls = obj.find('name').text
					xmlbox = obj.find('bndbox')
					b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
					face_box = img[b[2]:b[3],b[0]:b[1],:]
					known_face_encodings.append(face_recognition.face_encodings(face_box)[0])
					known_face_names.append(cls)

	return known_face_encodings, known_face_names
	
		
def Testing():
	known_face_encodings, known_face_names = Training()
	# STEP 2: Using the trained classifier, make predictions for unknown images
	for image_file in os.listdir(DIR_TEST):
		full_file_path = os.path.join(DIR_TEST, image_file)
		# unknown_image = face_recognition.load_image_file(full_file_path)
		unknown_image = face_recognition.load_image_file(full_file_path)

		face_locations = face_recognition.face_locations(unknown_image)
		face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
		
		# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
		# See http://pillow.readthedocs.io/ for more about PIL/Pillow
		pil_image = Image.fromarray(unknown_image).convert("RGB")
		# Create a Pillow ImageDraw Draw instance to draw with
		draw = ImageDraw.Draw(pil_image)

		matches = face_recognition.compare_faces(face_encodings, known_face_encodings[0])
		face_distances = face_recognition.face_distance(face_encodings, known_face_encodings[0])
		best_match_index = np.argmin(face_distances)
		num_face = len(face_encodings)
		for i, (top, right, bottom, left) in zip(range(num_face), face_locations):
			# See if the face is a match for the known face(s)
			

			name = "Unknown"

			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			
			
			if i == best_match_index and matches[best_match_index]:
				name = known_face_names[0]

			# Draw a box around the face using the Pillow module
			draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width = 3)

			# Draw a label with a name below the face
			text_width, text_height = draw.textsize(name)

			draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


			# Remove the drawing library from memory as per the Pillow docs
		del draw

			# Display the resulting image
		pil_image.show()
		
def train_prt():
	# tkinter.Label(window, text = "Train Done !").pack()
	tkinter.messagebox.showinfo(title = 'info', message= 'Train Done !')


def clear():
	for root, dirs, files in os.walk(DIR_PATH):
		for f in files:
			if f[-3:] == 'xml':
				os.unlink(os.path.join(root, f))
		for d in dirs:
			shutil.rmtree(os.path.join(root, d))

	# tkinter.Label(window, text = "Clear Done !").pack()
	tkinter.messagebox.showinfo(title = 'info', message= 'Clear Done !')

if __name__=='__main__':
	window = tkinter.Tk()
	window.title("Face Bao")
	window.geometry("500x90")    

	btn_lb = tkinter.Button(window, text = "Label", command = labelImg)
	btn_lb.pack(side="left", fill="both", expand="yes", padx="20", pady="20")

	btn_train = tkinter.Button(window, text = "Train", command = train_prt)
	btn_train.pack(side="left", fill="both", expand="yes", padx="20", pady="20")

	btn_test = tkinter.Button(window, text = "Test", command = Testing)
	btn_test.pack(side="left", fill="both", expand="yes", padx="20", pady="20")

	btn_cl = tkinter.Button(window, text = "Clear", command = clear)
	btn_cl.pack(side="left", fill="both", expand="yes", padx="20", pady="20")

	
	window.mainloop()
	