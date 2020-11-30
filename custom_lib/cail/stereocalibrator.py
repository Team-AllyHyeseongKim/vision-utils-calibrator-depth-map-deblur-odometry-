import glob

import cv2
import numpy as np
from tqdm import tqdm


class StereoCalibration(object):
    def __init__(self, leftpath, rightpath, caml, distl, camr, distr, img_shape):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.worldPoints = np.zeros((9 * 6, 3), np.float32)  # 모두 [0] 체스판에서 찾을 점의 3D좌표 셋 바탕
        self.worldPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # 월드 좌표계의 베이스를 잡아줌, 왼쪽 맨 위를 원점으로 함(0,0,0)
        # 체스판이므로 Z-좌표값은 0으로 함

        # Arrays to store object points and image points from all the images.
        self.objectPoints = []  # 3d point in real world space
        self.imagePointsL = []  # 2d points in image plane.
        self.imagePointsR = []  # 2d points in image plane.

        self.cameraMatrixL = caml
        self.distortionL = distl

        self.cameraMatrixR = camr
        self.distortionR = distr
        self.img_shape = img_shape

        self.R = None
        self.T = None
        self.E = None
        self.F = None

        self.readfile(leftpath, rightpath)

    def readfile(self, leftpath, rightpath):

        leftimagefile = glob.glob(leftpath + '/*.jpg')
        rightimagefile = glob.glob(rightpath + '/*.jpg')
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
                self.imagePointsL.append(left_corners)
                self.imagePointsR.append(right_corners)
                self.objectPoints.append(self.worldPoints)

        self.calibrate()

    def calibrate(self):
        print("enter stereo calibration")
        ret, self.cameraMatrix1, self.distortion1, self.cameraMatrix2, self.distortion2, self.R, self.T, self.E, self.F \
            = cv2.stereoCalibrate(
            objectPoints=self.objectPoints,
            imagePoints1=self.imagePointsL,
            imagePoints2=self.imagePointsR,
            imageSize=self.img_shape,
            cameraMatrix1=self.cameraMatrixL,
            distCoeffs1=self.distortionL,
            cameraMatrix2=self.cameraMatrixR,
            distCoeffs2=self.distortionR,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )

        # 한 카메라에 대한 켈리브레이션:
        # 성공여부, camera matrix, distortion coefficients, rotation and translation vector"s"
        # R|t는 한 뷰에 대한 월드 원점 - 2D 뷰의 영상중점 (cx,cy)간 좌표변환들
        print("exit stereo calibration")
