import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

right = cv.imread("images/phai1.jpg")
right = cv.resize(right, (0, 0), fx=0.5, fy=0.5)
img1 = cv.cvtColor(right, cv.COLOR_RGB2GRAY)
left = cv.imread("images/trai1.jpg")
left = cv.resize(left, (0, 0), fx=0.5, fy=0.5)
img2 = cv.cvtColor(left, cv.COLOR_RGB2GRAY)

sift = cv.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

match = cv.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

good = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good.append(m)
matches = np.asarray(good)

if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

    H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)

else:
    print("Can't find enough Key points !!!")

dst = cv.warpPerspective(right, H, (left.shape[1] + right.shape[1], left.shape[0]))



plt.figure()
dst[0:left.shape[0], 0:left.shape[1]] = left
cv.imwrite('stitched.jpg', dst)
cv.imshow('stitched', dst)

cv.waitKey(0)
cv.destroyAllWindows()
