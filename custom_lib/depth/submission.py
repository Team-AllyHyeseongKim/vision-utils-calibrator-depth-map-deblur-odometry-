import argparse
import cv2
from depth.models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from depth.models.submodule import *
from depth.utils.eval import mkdir_p, save_pfm
from depth.utils.preprocess import get_transform

cudnn.benchmark = False


class depth_map_calculator:
    def __init__(self, modelPath):
        # construct model
        '''
        level = output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)
        '''
        self.model = hsm(128, -1, level=1)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.cuda()

        # load model
        pretrained_dict = torch.load(modelPath)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        self.model.load_state_dict(pretrained_dict['state_dict'], strict=False)
        self.model.eval()
        # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        self.processed = get_transform()

    # 이미지는 ndarray-RGB-float32를 요구한다.
    def run(self, imgL_o, imgR_o):
        testres = 0.5

        # print(test_left_img[inx])
        # imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        # imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]
        imgsize = imgL_o.shape[:2]

        # 어캐 계산한건지는 잘 모르나, calib 파일에서 300정도의 값을 가져오길래, 
        max_disp = 300

        ## change max disp
        tmpdisp = int(max_disp * testres // 64 * 64)
        if (max_disp * testres / 64 * 64) > tmpdisp:
            self.model.module.maxdisp = tmpdisp + 64
        else:
            self.model.module.maxdisp = tmpdisp
        if self.model.module.maxdisp == 64:
            self.model.module.maxdisp = 128

        self.model.module.disp_reg8 = disparityregression(self.model.module.maxdisp, 16).cuda()
        self.model.module.disp_reg16 = disparityregression(self.model.module.maxdisp, 16).cuda()
        self.model.module.disp_reg32 = disparityregression(self.model.module.maxdisp, 32).cuda()
        self.model.module.disp_reg64 = disparityregression(self.model.module.maxdisp, 64).cuda()
        # print(self.model.module.maxdisp)

        # resize
        imgL_o = cv2.resize(imgL_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC)
        imgL = self.processed(imgL_o).numpy()
        imgR = self.processed(imgR_o).numpy()

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h - imgL.shape[2]
        left_pad = max_w - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            pred_disp, entropy = self.model(imgL, imgR)

        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - imgL_o.shape[0]
        left_pad = max_w - imgL_o.shape[1]
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # resize to highres
        pred_disp = cv2.resize(pred_disp / testres, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_LINEAR)
        torch.cuda.empty_cache()

        return pred_disp, entropy
