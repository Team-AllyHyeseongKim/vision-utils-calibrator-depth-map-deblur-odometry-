import depth.submission
import odometry.submission
import numpy as np
import skimage.io
import cv2
from tqdm import tqdm

if __name__ == '__main__':

    dm = depth.submission.depth_map_calculator('/content/drive/My Drive/custom_lib/depth/final-768px.tar')
    #dm = depth.submission.depth_map_calculator('C:\\final-768px.tar')

    # imgL_o = cv2.imread('/content/drive/My Drive/custom_lib/depth/data/kitti/L.png')
    # imgR_o = cv2.imread('/content/drive/My Drive/custom_lib/depth/data/kitti/R.png')
   
   
    # imgL_o = cv2.bilateralFilter(imgL_o,9,75,75)
    # imgR_o = cv2.bilateralFilter(imgR_o,9,75,75)
   
    # d, _ = dm.run(imgL_o, imgR_o)
    # cv2.imwrite('/content/drive/My Drive/custom_lib/depth/data/kitti/result.png', d)

    imgL_o = cv2.imread('/content/drive/My Drive/custom_lib/depth/data/our/L.jpg')
    imgR_o = cv2.imread('/content/drive/My Drive/custom_lib/depth/data/our/R.jpg')

    d, _ = dm.run(imgL_o, imgR_o)
    cv2.imwrite('/content/drive/My Drive/custom_lib/depth/data/our/result.png', d)

    # for i in tqdm(range(799)):
    #
    #     imgL_o = cv2.imread('C:\\03\\image_0\\'+"%06d" % i +'.png')
    #     imgR_o = cv2.imread('C:\\03\\image_1\\'+"%06d" % i +'.png')
    #
    #     #imgL_o = cv2.imread('C:\\scanData\\result\\color\\'+str(i)+'.jpg')
    #     #imgR_o = cv2.imread('C:\\scanData\\result\\right\\'+str(i)+'.jpg')
    #     d, _ = dm.run(imgL_o, imgR_o)
    #     cv2.imshow("dd", d)
    #     cv2.waitKey(50)
    #     # d = d.astype(np.longdouble)
    #     # d = d * 65535
    #     # d = d / 255
    #     # d = d.astype(np.uint16)
    #     # cv2.imwrite('C:\\scanData\\result\\depth\\'+str(i)+'.png',  d)
    #     np.savetxt('C:\\scanData\\result\\pose\\'+str(i)+'.txt', np.identity(4), fmt='%.6f', delimiter=' ')



    # odo = odometry.submission.odometry(dm)
    #
    # Proj1 = np.array([
    #     [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 0.000000000000e+00],
    #     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
    #     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
    # Proj2 = np.array([
    #     [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.875744000000e+02],
    #     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
    #     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
    #
    # Proj1 = np.array([
    #     [1.23867069e+03, 0.00000000e+00, 2.86672811e+02, 0.00000000e+00],
    #     [0.00000000e+00, 1.23279962e+03, 2.14768802e+02, 0.00000000e+00],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
    # Proj2 = np.array([
    #     [1.23975694e+03, 9.24295614e+00, 2.81786580e+02, -4.64460611e+03],
    #     [-2.07914134e+01, 1.22173896e+03, 2.69892894e+02, 1.00995178e+03],
    #     [4.50612838e-03, -4.48184596e-02, 9.98984985e-01, 1.00405885e+00]])

    '''
    X = []
    Y = []
    Z = []
    
    for fm in tqdm(range(800)):
        ImT1_L = cv2.imread('/content/drive/My Drive/03/image_0/' + '%06d' % fm + '.png', 0)  # 0 flag returns a grayscale image
        ImT1_R = cv2.imread('/content/drive/My Drive/03/image_1/' + '%06d' % fm + '.png', 0)
        ImT2_L = cv2.imread('/content/drive/My Drive/03/image_0/' + '%06d' % (fm + 1) + '.png', 0)
        ImT2_R = cv2.imread('/content/drive/My Drive/03/image_1/' + '%06d' % (fm + 1) + '.png', 0)
        # ImT1_L, ImT1_R, ImT2_L, ImT2_R, Proj1, Proj2, isGray=false
        mat = odo.run(ImT1_L, ImT1_R, ImT2_L, ImT2_R, Proj1, Proj2, isGray=False)
    
        
        X.append(mat[0][3])
        Y.append(mat[1][3])
        Z.append(mat[2][3])
        if fm % 10 == 0:
            np.save('/content/drive/My Drive/odooutput/Xcoord' + str(fm) + '.npy', X)
            np.save('/content/drive/My Drive/odooutput/Ycoord' + str(fm) + '.npy', Y)
            np.save('/content/drive/My Drive/odooutput/Zcoord' + str(fm) + '.npy', Z)
    '''

    # for fm in tqdm(range(49)):
    #     ImT1_L = cv2.imread('C:\\scanData\\result\\left\\' + str(fm) + '.jpg')  # 0 flag returns a grayscale image
    #     ImT1_R = cv2.imread('C:\\scanData\\result\\right\\' + str(fm) + '.jpg')
    #     ImT2_L = cv2.imread('C:\\scanData\\result\\left\\' + str(fm+1) + '.jpg')
    #     ImT2_R = cv2.imread('C:\\scanData\\result\\right\\' + str(fm+1) + '.jpg')
    #     # ImT1_L, ImT1_R, ImT2_L, ImT2_R, Proj1, Proj2, isGray=false
    #     mat = odo.run(ImT1_L, ImT1_R, ImT2_L, ImT2_R, Proj1, Proj2, isGray=False)
    #     if mat is not None:
    #         mat = np.vstack((mat, np.array([0, 0, 0, 1]).reshape(1, -1)))
    #         np.savetxt('C:\\scanData\\result\\pose\\' + str(fm+1) + '.txt', mat, fmt='%.6f', delimiter=' ')
    #     else:
    #         np.savetxt('C:\\scanData\\result\\pose\\' + str(fm + 1) + '.txt', np.identity(4), fmt='%.6f', delimiter=' ')
    #     #print(mat)
    #


    '''
    imgL_o = skimage.io.imread('/content/drive/My Drive/03/image_0/000000.png').astype('float32')
    imgR_o = skimage.io.imread('/content/drive/My Drive/03/image_1/000000.png').astype('float32')
    # 흑백이라서 무식한 짓 함
    imgL_o = cv2.merge([imgL_o,imgL_o,imgL_o])
    imgR_o = cv2.merge([imgR_o,imgR_o,imgR_o])
    '''





#np.save('/content/drive/My Drive/temp', d)
'''
imgL_o = (skimage.io.imread('/content/drive/My Drive/custom_lib/depth/q/Adirondack-perfect/im0.png').astype('float32'))[:,:,:3]
imgR_o = (skimage.io.imread('/content/drive/My Drive/custom_lib/depth/q/Adirondack-perfect/im1.png').astype('float32'))[:,:,:3]
'''
