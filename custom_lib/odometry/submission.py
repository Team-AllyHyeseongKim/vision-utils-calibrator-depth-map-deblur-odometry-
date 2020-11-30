import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.io
from odometry.helperFunctions import *
from scipy.optimize import least_squares


class odometry():
    #dm = submission.depth_map_calculator('/content/drive/My Drive/custom_lib/final-768px.tar')
    def __init__(self, dm):
        self.dm = dm


    # each image is numpy, color
    # Proj 은 world to camera matrix 이다.
    # 월드 원점부터 카메라 사이의 변환이다 : kitti의 경우 월드 원점이 두 카메라 사이이다.
    # Proj(3x4, ndarray) = 내부왜곡(3x3) * 외부왜곡(3x4) from kitti database
    # 왼쪽이 주시 카메라다.
    def run(self, ImT1_L, ImT1_R, ImT2_L, ImT2_R, Proj1, Proj2, isGray=False):
        if isGray:
            ImT1_Lo = cv2.merge([ImT1_L,ImT1_L,ImT1_L])
            ImT1_Ro = cv2.merge([ImT1_R,ImT1_R,ImT1_R])
            ImT2_Lo = cv2.merge([ImT2_L,ImT2_L,ImT2_L])
            ImT2_Ro = cv2.merge([ImT2_R,ImT2_R,ImT2_R])
        else:
            ImT1_Lo = ImT1_L
            ImT1_Ro = ImT1_R
            ImT2_Lo = ImT2_L
            ImT2_Ro = ImT2_R

        ImT1_disparity, _ = self.dm.run(ImT1_Lo, ImT1_Ro)
        ImT2_disparity, _ = self.dm.run(ImT2_Lo, ImT2_Ro)

        ImT1_disparityA = np.divide(ImT1_disparity, 16.0)
        ImT2_disparityA = np.divide(ImT2_disparity, 16.0)

        if not isGray:
            ImT1_L = cv2.cvtColor(ImT1_L, cv2.COLOR_BGR2GRAY)
            ImT1_R = cv2.cvtColor(ImT1_R, cv2.COLOR_BGR2GRAY)
            ImT2_L = cv2.cvtColor(ImT2_L, cv2.COLOR_BGR2GRAY)
            ImT2_R = cv2.cvtColor(ImT2_R, cv2.COLOR_BGR2GRAY)

        # 특징 추출 부분
        TILE_H = 10
        TILE_W = 20
        fastFeatureEngine = cv2.FastFeatureDetector_create()

        #20x10 (wxh) tiles for extracting less features from images 
        H,W = ImT1_L.shape
        kp = []
        idx = 0
        for y in range(0, H, TILE_H):
            for x in range(0, W, TILE_W):
                imPatch = ImT1_L[y:y+TILE_H, x:x+TILE_W]
                keypoints = fastFeatureEngine.detect(imPatch)
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
                
                if (len(keypoints) > 10):
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    for kpt in keypoints[0:10]:
                        kp.append(kpt)
                else:
                    for kpt in keypoints:
                        kp.append(kpt)
        if len(kp) <=0:
            print("sorry, it's hard to find key points in this image")
            return None

        #루이스 카나데 부분
        # pack keypoint 2-d coords into numpy array
        # kp는 특징점들의 리스트
        # trackPoints1 *,1,2 모양으로, 맨 마지막 차원이 x, y를 나타냄
        trackPoints1 = np.zeros((len(kp),1,2), dtype=np.float32)
        for i,kpt in enumerate(kp):
            trackPoints1[i,:,0] = kpt.pt[0]
            trackPoints1[i,:,1] = kpt.pt[1]

        # Parameters for lucas kanade optical flow
        # 딕셔너리로 만들고, ** 붙여서 풀어버리면 이름있는 파라미터로서 전달됨
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        # 원 이미지, 미래의 이미지, 원 이미지의 특징점을 전달하면, 원 이미지의 특징점들을 미래의 이미지에서 찾아서 반환해준다.
        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(ImT1_L, ImT2_L, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)

        
        # separate points that were tracked successfully
        ptTrackable = np.where(st == 1, 1,0).astype(bool)
        trackPoints1_KLT = trackPoints1[ptTrackable, ...]

        trackPoints2_KLT_t = trackPoints2[ptTrackable, ...] # 라운드하기 위한 임시변수
        trackPoints2_KLT = np.around(trackPoints2_KLT_t)
        


        # among tracked points take points within error measue
        error = 4
        errTrackablePoints = err[ptTrackable, ...]
        errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
        trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
        trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]        

        # 삼차원 정보 사용하는 부분
        #compute right image disparity displaced points
        trackPoints1_KLT_L = trackPoints1_KLT
        trackPoints2_KLT_L = trackPoints2_KLT

        trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
        trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
        selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

        disparityMinThres = 0.0
        disparityMaxThres = 100.0
        for i in range(trackPoints1_KLT_L.shape[0]):
            if trackPoints2_KLT_L[i,1] >= ImT1_disparityA.shape[0] or trackPoints2_KLT_L[i,0] >= ImT1_disparityA.shape[1]:
                continue
            # print(int(trackPoints2_KLT_L[i,1]), ", ", int(trackPoints2_KLT_L[i,0]))

            T1Disparity = ImT1_disparityA[int(trackPoints1_KLT_L[i,1]), int(trackPoints1_KLT_L[i,0])]
            T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]
            
            if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres 
                and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
                trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
                trackPoints2_KLT_R[i, 0] = trackPoints2_KLT_L[i, 0] - T2Disparity
                selectedPointMap[i] = 1
                
        selectedPointMap = selectedPointMap.astype(bool)
        trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
        trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]
        trackPoints2_KLT_L_3d = trackPoints2_KLT_L[selectedPointMap, ...]
        trackPoints2_KLT_R_3d = trackPoints2_KLT_R[selectedPointMap, ...]

        # 3d movement point cloud generation
        # 3d point cloud triagulation
        # 각 시점마다 3d pcloud 점 집합을 구성한다.
        numPoints = trackPoints1_KLT_L_3d.shape[0]
        d3dPointsT1 = generate3DPoints(trackPoints1_KLT_L_3d, trackPoints1_KLT_R_3d, Proj1, Proj2)
        d3dPointsT2 = generate3DPoints(trackPoints2_KLT_L_3d, trackPoints2_KLT_R_3d, Proj1, Proj2)

        if 0 >= numPoints:
            print("sorry, it's hard to find track points in this image")
            return None

        #ransac으로 에러 정정 및 이동 정보 확보
        ransacError = float('inf')
        dOut = None
        # RANSAC
        ransacSize = 6

        for ransacItr in range(250):
            sampledPoints = np.random.randint(0, numPoints, ransacSize)
            rD2dPoints1_L = trackPoints1_KLT_L_3d[sampledPoints]
            rD2dPoints2_L = trackPoints2_KLT_L_3d[sampledPoints]
            rD3dPointsT1 = d3dPointsT1[sampledPoints]
            rD3dPointsT2 = d3dPointsT2[sampledPoints]

            dSeed = np.zeros(6)
            #TODO
            #minimizeReprojection(d, trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1)
            optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=200,
                                args=(rD2dPoints1_L, rD2dPoints2_L, rD3dPointsT1, rD3dPointsT2, Proj1))
            #TODO
            #error = optRes.fun
            error = minimizeReprojection(optRes.x, trackPoints1_KLT_L_3d, trackPoints2_KLT_L_3d,
                                            d3dPointsT1, d3dPointsT2, Proj1)

            eCoords = error.reshape((d3dPointsT1.shape[0]*2, 3))
            totalError = np.sum(np.linalg.norm(eCoords, axis=1))

            if (totalError < ransacError):
                ransacError = totalError
                dOut = optRes.x

        # r, t generation
        # translation[0],translation[1],translation[2] X,Y,Z
        rotation = genEulerZXZMatrix(dOut[0], dOut[1], dOut[2])
        translation = np.array([[dOut[3]], [dOut[4]], [dOut[5]]])
        outMat = np.hstack((rotation, translation))

        return outMat