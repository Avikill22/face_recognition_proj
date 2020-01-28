import cv2
import pickle
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")
labels ={}
with open("label.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while cap.isOpened() :
    ref , frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,9)

    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (x,y), font, 1, (255,135,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+w),(205,102,0),1)

    cv2.imshow('img',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()