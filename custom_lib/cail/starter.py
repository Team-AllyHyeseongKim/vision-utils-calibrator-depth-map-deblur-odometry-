import cv2
import numpy as np
from cail.calibrator import Calibration
from cail.deepth import Deepth
from cail.stereocalibrator import StereoCalibration
from cail.stereorectify import Stereorectify
from tqdm import tqdm

# 10번 라인 12번 라인 26번 라인 수정할 것 ( 파일에 맞게 )
# deepth 에서 약간 가중치 조절할 수는 있는데, 혹 겁나 빨라서 되면 몇개 해보고 아니면 말고 파이팅!!
# 켈리브레이션 부분 사실 중복되는 부분인데 파일에 따로 빼놨다가 다시 넣는거 귀찮아서 그냥 냅둠
# 아둔 토리다스
#

# np.savetxt('C:\\calib\\result\\intrinsic.txt', cal.cameraMatrix, delimiter=' ')

# rt = np.hstack((scal.R, scal.T.reshape(-1, 1))).reshape(3, 4)
# np.savetxt('C:\\calib\\result\\extrinsic.txt', rt, delimiter=' ')


main = 'C:\\d\\board\\image'

left = main+'\\left'
right = main+'\\right'

calL = Calibration(targetfilepath=left)

calR = Calibration(targetfilepath=right)

scal = StereoCalibration(left, right,
                         calL.cameraMatrix, calL.distortion, calR.cameraMatrix, calR.distortion, calL.img_shape)

srti = Stereorectify(calL.cameraMatrix, calL.distortion, calR.cameraMatrix, calR.distortion, calL.img_shape, scal.R, scal.T)


# caml = np.load(main+'\\cameraMatrixL.npy')
# disl = np.load(main+'\\distortionL.npy')
# camr = np.load(main+'\\cameraMatrixR.npy')
# disr = np.load(main+'\\distortionR.npy')
# imgs = np.load(main+'\\img_shapeL.npy')
# imgs = (imgs[0], imgs[1])
#
# R = np.load(main+'\\R.npy')
# T = np.load(main+'\\T.npy')
#
# srti = Stereorectify(caml, disl, camr, disr, imgs, R, T)

import glob
leftimagefile = glob.glob(left + '/*.jpg')
imgmatL = cv2.imread(leftimagefile[5])
cv2.imshow("org", imgmatL)

imgmatL = cv2.remap(imgmatL, srti.left_maps[0], srti.left_maps[1], cv2.INTER_LANCZOS4)
cv2.imshow("new", imgmatL)
cv2.waitKey(0)

# np.savetxt(main+'\\intrinsic.txt', calL.cameraMatrix, fmt="%.6f", delimiter=' ')
#
# np.save(main+'\\cameraMatrixL.npy', calL.cameraMatrix)
# np.save(main+'\\distortionL.npy', calL.distortion)
# np.save(main+'\\img_shapeL.npy', calL.img_shape)
# np.save(main+'\\cameraMatrixR.npy', calR.cameraMatrix)
# np.save(main+'\\distortionR.npy', calR.distortion)
# np.save(main+'\\img_shapeR.npy', calR.img_shape)
#
# np.save(main+'\\R.npy', scal.R)
# np.save(main+'\\T.npy', scal.T)
#
# np.save(main+'\\left_maps0.npy', srti.left_maps[0])
# np.save(main+'\\left_maps1.npy', srti.left_maps[1])
# np.save(main+'\\right_maps0.npy', srti.right_maps[0])
# np.save(main+'\\right_maps1.npy', srti.right_maps[1])
# np.save(main+'\\P1.npy', srti.P1)
# np.save(main+'\\P2.npy', srti.P2)


# np.savetxt('C:\\calib\\result\\intrinsic.txt', cal.cameraMatrix, delimiter=' ')

# rt = np.hstack((scal.R, scal.T.reshape(-1, 1))).reshape(3, 4)
# np.savetxt('C:\\calib\\result\\extrinsic.txt', rt, delimiter=' ')







# srti = Stereorectify(cal.cameraMatrix, cal.distortion, cal.img_shape, scal.R, scal.T,
#                      'C:\\scanData\\left', 'C:\\scanData\\right')


#
# i=0
# for l, r in tqdm(zip(srti.left_img_remaps, srti.right_img_remaps)):
#     cv2.imwrite('C:\scanData\\result\\left\\'+str(i)+'.jpg', l)
#     cv2.imwrite('C:\scanData\\result\\right\\'+str(i)+'.jpg', r)
#     i+=1




'''
deep = Deepth()

for l, r in zip(srti.left_img_remaps, srti.right_img_remaps):
    # cv2.imshow('left',l)
    # cv2.waitKey(0)
    # cv2.imshow('right',r)
    # cv2.waitKey(0)

    deep.compute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
'''
print("cleared")

'''
rts = []
for r, t in zip(cal.targetRvecs, cal.targetTvecs):
    rt = np.hstack((r, t.reshape(-1, 1))).reshape(-1, 12)
    rts.append(rt)

rts = np.array(rts).squeeze(1)

np.savetxt('C:\\calib\\result\\extrinsic.npy', rts, delimiter=' ', fmt='%1.6e', newline='\n')
'''



