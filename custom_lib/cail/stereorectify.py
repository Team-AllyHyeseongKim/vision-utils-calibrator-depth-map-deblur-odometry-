import glob
import os

import cv2
from tqdm import tqdm


class Stereorectify(object):
    def __init__(self, cameraMatrixL, distortionL, cameraMatrixR, distortionR, img_shape, R, T):
        print("ch ver.")
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi1 = None
        self.roi2 = None

        self.cameraMatrixL = cameraMatrixL
        self.distortionL = distortionL

        self.cameraMatrixR = cameraMatrixR
        self.distortionR = distortionR

        self.img_shape = img_shape

        self.R = R
        self.T = T

        self.run()

    def run(self):

        # def get_key(fp):
        #     filename = os.path.splitext(os.path.basename(fp))[0]
        #     int_part = filename.split()[0]
        #     return int(int_part)
        #
        # leftimagefile = sorted(glob.glob(self.leftpath + '/*.jpg'), key=get_key)
        # rightimagefile = sorted(glob.glob(self.rightpath + '/*.jpg'), key=get_key)

        print("calculate stereoRectify")
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(self.cameraMatrixL,
                                                                                             self.distortionL,
                                                                                             self.cameraMatrixR,
                                                                                             self.distortionR,
                                                                                             self.img_shape, self.R,
                                                                                             self.T,
                                                                                             alpha=0)
        print("calculate unsidisortion")
        self.left_maps = cv2.initUndistortRectifyMap(self.cameraMatrixL, self.distortionL, self.R1, self.P1, self.img_shape,
                                                cv2.CV_16SC2)
        self.right_maps = cv2.initUndistortRectifyMap(self.cameraMatrixR, self.distortionR, self.R2, self.P2, self.img_shape,
                                                 cv2.CV_16SC2)
        print("done")
        # print("start remapping")
        # for i in tqdm(range(len(leftimagefile))):
        #     imgmatL = cv2.imread(leftimagefile[i])
        #     imgmatR = cv2.imread(rightimagefile[i])
        #
        #     self.left_img_remaps.append(cv2.remap(imgmatL, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4))
        #     self.right_img_remaps.append(cv2.remap(imgmatR, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4))
        # print("end remapping")

