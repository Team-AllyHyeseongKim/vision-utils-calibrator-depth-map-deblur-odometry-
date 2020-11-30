'''
made by heebin Yoo in cvml in cau, seoul, RoK
for stereo calibration
image size should < 7 for remap output
ret val should < 0.7
'''


import cv2
import numpy as np
import glob
from tqdm import tqdm

main = 'C:\\cailb'

left = main+'\\left'
right = main+'\\right'

criteria = (cv2.TERM_CRITERIA_EPS +
                 cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)

worldPoints = np.zeros((9 * 6, 3), np.float32)  # 모두 [0] 체스판에서 찾을 점의 3D좌표 셋 바탕
worldPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objectPoints = []  # 3d point in real world space
imagePointsL = []  # 2d points in image plane.
imagePointsR = []  # 2d points in image plane.

leftimagefile = glob.glob(left + '\\*.jpg')
rightimagefile = glob.glob(right + '\\*.jpg')

print("start to load files")
for i in tqdm(range(len(leftimagefile))):

    imgmatL = cv2.imread(leftimagefile[i])
    imgmatR = cv2.imread(rightimagefile[i])

    imggrayL = cv2.cvtColor(imgmatL, cv2.COLOR_BGR2GRAY)
    imggrayR = cv2.cvtColor(imgmatR, cv2.COLOR_BGR2GRAY)

    left_found, left_corners = cv2.findChessboardCorners(imggrayL, (9, 6), None)
    right_found, right_corners = cv2.findChessboardCorners(imggrayR, (9, 6), None)

    if left_found and right_found:
        cv2.cornerSubPix(imggrayL, left_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.cornerSubPix(imggrayR, right_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        imagePointsL.append(left_corners)
        imagePointsR.append(right_corners)
        objectPoints.append(worldPoints)

cam1 = cv2.initCameraMatrix2D(objectPoints, imagePointsL, imggrayL.shape,0)
cam2 = cv2.initCameraMatrix2D(objectPoints, imagePointsR, imggrayR.shape,0)

ret, cam1, dist1, cam2, dist2, R, T, E, F \
    = cv2.stereoCalibrate(
    objectPoints=objectPoints,
    imagePoints1=imagePointsL,
    imagePoints2=imagePointsR,
    imageSize=imggrayL.shape,
    cameraMatrix1=cam1,
    distCoeffs1=None,
    cameraMatrix2=cam2,
    distCoeffs2=None
    )

intrinsic = np.hstack((cam1, np.array([0,0,0]).reshape(-1,1)))
intrinsic = np.vstack((intrinsic, np.array([0,0,0,1])))

np.savetxt('C:\\scanData\\result\\intrinsic_color.txt', intrinsic, "%.6f")
np.savetxt('C:\\scanData\\result\\intrinsic_depth.txt', intrinsic, "%.6f")

print(ret)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cam1,dist1,cam2,dist2,imggrayL.shape,R,T,alpha=-1)

left_maps = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, imggrayL.shape,
                                             cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cam2, dist2, R2, P2, imggrayR.shape,
                                              cv2.CV_16SC2)



import glob
leftimagefile = glob.glob(left + '/*.jpg')
imgmatL = cv2.imread(leftimagefile[5])
cv2.imshow("orgL", imgmatL)

imgmatL = cv2.remap(imgmatL, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
cv2.imshow("newL", imgmatL)


rightimagefile = glob.glob(right + '/*.jpg')
imgmatR = cv2.imread(rightimagefile[5])
cv2.imshow("orgR", imgmatR)

imgmatR = cv2.remap(imgmatR, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
cv2.imshow("newR", imgmatR)


cv2.waitKey(0)

np.save(main+'\\left_maps0.npy', left_maps[0])
np.save(main+'\\left_maps1.npy', left_maps[1])
np.save(main+'\\right_maps0.npy', right_maps[0])
np.save(main+'\\right_maps1.npy', right_maps[1])
