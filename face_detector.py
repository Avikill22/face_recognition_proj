import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img  = cv2.imread("shruti2.png");
cap = cv2.VideoCapture('y2mate.com - selena_gomez_rare_official_music_video_ia1iuXbEaYQ_360p.mp4')
while cap.isOpened() :
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0),3)

    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()