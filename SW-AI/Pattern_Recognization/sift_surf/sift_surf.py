import cv2
import numpy as np
import time

sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성
surf = cv2.xfeatures2d.SURF_create()

filepath = 'test.jpg'
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_resize = [0.5,1,2,4,8]
for x in img_resize:
    tmp = cv2.resize(gray,dsize=(0,0) ,fx=x,fy=x)
    print('img_shape {}'.format(tmp.shape))
    start = time.time()
    sift.detect(image=tmp, mask=None)
    print("SIFT time :{:.5f} sec".format(time.time() - start))

    start = time.time()
    surf.detect(image=tmp, mask=None)
    print("SURF time :{:.5f} sec".format(time.time() - start))