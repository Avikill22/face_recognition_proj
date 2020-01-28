import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR,"img")

label_ids={}
current_id=0
x_train = []
y_labels = []

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label= os.path.basename(os.path.dirname(path)).lower().replace(" ","-")
            #print(label ,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            pil_Img = Image.open(path).convert("L")
            size = (550,550)
            final_Img = pil_Img.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_Img, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.1, 9)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("label.pickle",'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("train.yml")