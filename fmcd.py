import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('CardsPng/diamonds.jpg',0)
img2 = cv2.imread('CardsPng/test2.jpg',0)

orb  = cv2.ORB_create()
print(orb)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:5],None,flags=2)

plt.imshow(img3)
plt.show()
print(matches)
#cv2.imshow('detected',img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
