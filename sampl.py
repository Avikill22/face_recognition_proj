import cv2
img = cv2.imread("shruti.jpg",-1)
cv2.imshow("image",img)
i = cv2.waitKey(0)
if i==27:
    cv2.destroyAllWindows()
elif i==ord('s'):
    cv2.imwrite("shruti_crush.png",img)
    cv2.destroyAllWindows()


