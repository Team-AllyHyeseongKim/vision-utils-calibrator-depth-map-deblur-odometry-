import cv2

from matplotlib import pyplot as plt


class Deepth(object):
    def __init__(self):
        self.SWS = 5
        self.PFS = 5
        self.PFC = 29
        self.MDS = 4
        self.NOD = 128
        self.TTH = 80
        self.UR = 5
        self.SR = 16
        self.SPWS = 45

        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)

        self.stereo.setPreFilterType(1)
        self.stereo.setPreFilterSize(self.PFS)
        self.stereo.setPreFilterCap(self.PFC)

        self.stereo.setTextureThreshold(self.TTH)
        self.stereo.setUniquenessRatio(self.UR)

        self.stereo.setMinDisparity(self.MDS)

        self.stereo.setSpeckleRange(self.SR)
        self.stereo.setSpeckleWindowSize(self.SPWS)

    def compute(self, left, right):
        print("compute deepth")
        disparity = self.stereo.compute(left, right)
        plt.imshow(disparity, 'gray')
        plt.show()
