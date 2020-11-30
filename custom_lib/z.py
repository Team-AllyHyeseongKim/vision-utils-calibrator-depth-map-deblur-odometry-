import cv2
import numpy as np

#remap 파라미터
left_maps0 = np.load("c:\\servertest\\left_maps0.npy")
left_maps1 = np.load("c:\\servertest\\left_maps1.npy")
right_maps0 = np.load("c:\\servertest\\right_maps0.npy")
right_maps1 = np.load("c:\\servertest\\right_maps1.npy")

imgL_o = cv2.imread('C:\\Users\\SW교육지원팀\\Google 드라이브\\custom_lib\\depth\\data\\our\\L.jpg')
imgR_o = cv2.imread('C:\\Users\\SW교육지원팀\\Google 드라이브\\custom_lib\\depth\\data\\our\\R.jpg')

left_image = cv2.remap(imgL_o, left_maps0, left_maps1, cv2.INTER_LANCZOS4)
right_image = cv2.remap(imgR_o, right_maps0, right_maps1, cv2.INTER_LANCZOS4)

cv2.imwrite('C:\\Users\\SW교육지원팀\\Google 드라이브\\custom_lib\\depth\\data\\our\\L.png', left_image)
cv2.imwrite('C:\\Users\\SW교육지원팀\\Google 드라이브\\custom_lib\\depth\\data\\our\\R.png', right_image)

