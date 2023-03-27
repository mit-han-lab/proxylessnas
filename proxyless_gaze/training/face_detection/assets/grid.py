import cv2
import numpy as np

def mouseListener(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x},{y})")

img = cv2.imread("./normal_face.jpg")
print(img.shape)
img = cv2.resize(img, None, fx=0.25, fy=0.25)
zero = np.zeros((128, 160, 3), dtype=np.uint8)
zero[:120, :160] = img[:,:]
img = zero[:,:,:]
cv2.imshow("img", img)
cv2.setMouseCallback('img', mouseListener)
cv2.waitKey()
cv2.destroyAllWindows()
