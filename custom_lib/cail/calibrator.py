import glob

import cv2
import numpy as np
from tqdm import tqdm


class Calibration(object):
    def __init__(self, targetfilepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.worldPoints = np.zeros((9 * 6, 3), np.float32)  # 모두 [0] 체스판에서 찾을 점의 3D좌표 셋 바탕
        self.worldPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # 월드 좌표계의 베이스를 잡아줌, 왼쪽 맨 위를 원점으로 함(0,0,0)
        # 체스판이므로 Z-좌표값은 0으로 함

        # Arrays to store object points and image points from all the images.
        self.objectPoints = []  # 3d point in real world space, 켈리브레이션 넣을때 셋 지정하려고
        self.imagePoints = []  # 2d points in image plane.

        self.cameraMatrix = None
        self.distortion = None
        self.img_shape = None
        self.rvecs = None
        self.tvecs = None

# 덤프를 제거한 원하는 본질만 담은 것 & 회전벡터는 3x3으로 처리해놓기도 함
        self.targetRvecs = []
        self.targetTvecs = []

        self.readfile(targetfilepath)

    def readfile(self, targetfilepath):

        targetimagefile = glob.glob(targetfilepath + '\\*.jpg')
        #targetimagefile.sort()

        print("start loading files")
        for i in tqdm(range(len(targetimagefile))):

            # print(targetimagefile[i])

            imgmat = cv2.imread(targetimagefile[i])

            # 그레이 스케일로 변경
            imggray = cv2.cvtColor(imgmat, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # 성공 여부, 코너 포인트 리스트
            # 이미지, 모서리 수, 플래그
            ret, corners = cv2.findChessboardCorners(imggray, (9, 6), None)

            if ret is True:
                # If found, add object points, image points (after refining them)
                cv2.cornerSubPix(imggray, corners, (11, 11), (-1, -1), self.criteria)  # 찾은 코너에 대한 보정
                self.imagePoints.append(corners)
                self.objectPoints.append(self.worldPoints)

                # ret = cv2.drawChessboardCorners(imgmat, (9, 6), corners, ret)
                # cv2.imshow("test", imgmat)
                # cv2.waitKey(1)

        self.img_shape = cv2.cvtColor(cv2.imread(targetimagefile[0]), cv2.COLOR_BGR2GRAY).shape[::-1]
        self.calibrate(len(targetimagefile))

    def calibrate(self, target_lenght):
        print("enter 1d calibration")
        ret, self.cameraMatrix, self.distortion, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objectPoints=self.objectPoints,
            imagePoints=self.imagePoints,
            imageSize=self.img_shape,
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distortion,
            rvecs=self.rvecs,
            tvecs=self.tvecs)
        print(ret)
        # 한 카메라에 대한 켈리브레이션:
        # 성공여부, camera matrix, distortion coefficients, rotation and translation vector"s"
        # R|t는 한 뷰에 대한 월드 원점 - 2D 뷰의 영상중점 (cx,cy)간 좌표변환들
        print("exit 1d calibration")

        # for i in tqdm(range(target_lenght)):
        #     dst, _ = cv2.Rodrigues(self.rvecs[i])
        #     self.targetRvecs.append(dst)
        #     self.targetTvecs.append(self.tvecs[i])
        # print("Rodrigues eqs is solved")
        # for i, (r, t) in enumerate(zip(self.rvecs, self.tvecs)):
        #        dst, _ = cv2.Rodrigues(r)  # 회전벡터의 경우 표현방식이 달라서 변환 필요. 로드리게스 변환 참조.
        #        print(i, "번째 회전이동 : \n", dst)
        #        print(i, "번째 평행이동 : \n", t)
