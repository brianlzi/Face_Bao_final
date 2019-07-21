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

def labelImg():
    os.system(" cd .. && cd labelImg && python labelImg.py")

def Training():
	'''
	solve xml to face
	'''
	known_face_encodings = []
	known_face_names = []

	if not os.path.isdir("cache face"):
		os.mkdir("cache face")

	for root, dirs, files in os.walk("cache face"):
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
	for image_file in os.listdir("Img Test"):
		full_file_path = os.path.join("Img Test", image_file)
		# unknown_image = face_recognition.load_image_file(full_file_path)
		unknown_image = face_recognition.load_image_file(full_file_path)

		face_locations = face_recognition.face_locations(unknown_image)
		face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

		# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
		# See http://pillow.readthedocs.io/ for more about PIL/Pillow
		pil_image = Image.fromarray(unknown_image).convert("RGB")
		# Create a Pillow ImageDraw Draw instance to draw with
		draw = ImageDraw.Draw(pil_image)


		for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

			name = "Unknown"

			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]

			# Draw a box around the face using the Pillow module
			draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width = 5)

			# Draw a label with a name below the face
			text_width, text_height = draw.textsize(name)
			draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
			draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


			# Remove the drawing library from memory as per the Pillow docs
			del draw

			# Display the resulting image
			pil_image.show()

def train_prt():
	tkinter.Label(window, text = "Train Done !").pack()


def clear():
	for root, dirs, files in os.walk("cache face"):
		for f in files:
			if f[-3:] == 'xml':
				os.unlink(os.path.join(root, f))
		for d in dirs:
			shutil.rmtree(os.path.join(root, d))


if __name__=='__main__':
	window = tkinter.Tk()
	window.title("Face Recognition")
	window.geometry("300x200")    

	tkinter.Button(window, text = "LabelImg", command = labelImg).pack() 
	tkinter.Button(window, text = "Train!", command = train_prt).pack()
	tkinter.Button(window, text = "Test", command = Testing).pack()
	tkinter.Button(window, text = "Clear", command = clear).pack()
	window.mainloop()
	