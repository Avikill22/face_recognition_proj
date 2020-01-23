import numpy as nm
import cv2
cam = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc,20.0,(640,480))
print(cam.isOpened())
while(cam.isOpened()):
    ret ,frame = cam.read()
    print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows() 