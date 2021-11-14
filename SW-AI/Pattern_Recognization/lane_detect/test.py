import cv2
import matplotlib.pyplot as plt

path = './data'

# img = cv2.imread(path+'/test_images/challenge.jpg')
#
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cmap='gray')
#
def pipeline(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray,(9,9),0.0)

    tmp = cv2.Canny(blurred_img,70,140)
    return tmp
# plt.imshow(tmp,cmap='gray')

cap = cv2.VideoCapture(path+'/test_videos/solidWhiteRight.mp4')

while(True):
    ok, frame = cap.read()
    if not ok:
        break
    frame = pipeline(frame)
    cv2.imshow('frame',frame)
    key = cv2.waitKey((10))
    if key == ord('x'):
        break

cap.release()