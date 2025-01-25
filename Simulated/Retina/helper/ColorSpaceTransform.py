import pickle
import numpy as np
import torch
from root_config import ROOT_DIR


class ColorSpaceTransform():
    def __init__(self, cone_fundamentals, device):

        self.device = device
        self.cone_fundamentals = cone_fundamentals
        self.Q = self.cone_fundamentals[:,3]
        self.is_tetrachromatic = torch.max(self.Q) != 0

        # XYZ color matching functions (CIE 1931)
        with open(f'{ROOT_DIR}/Simulated/Retina/helper/data/CIEXYZ.cpkl', 'rb') as f:
            CIEXYZ_dict = pickle.load(f)

        CIEXYZ = []
        for i in range(400, 701):
            CIEXYZ.append(CIEXYZ_dict[i])
        CIEXYZ = np.asarray(CIEXYZ)
        self.CIEXYZ = torch.FloatTensor(CIEXYZ).to(self.device)

        self.CIEXYZ_to_LMS_matrix = torch.linalg.inv(self.CIEXYZ.T @ self.CIEXYZ) @ self.CIEXYZ.T @ self.cone_fundamentals
        self.CIEXYZ_to_LMS_matrix = self.CIEXYZ_to_LMS_matrix.to(self.device)
        
        # assumes D65 illuminant
        self.linsRGB_to_CIEXYZ_matrix = torch.FloatTensor([ [0.4124, 0.3576, 0.1805],
                                                            [0.2126, 0.7152, 0.0722],
                                                            [0.0193, 0.1192, 0.9505]]).to(self.device)

        self.linsRGB_to_LMS_matrix = (self.CIEXYZ_to_LMS_matrix.T @ self.linsRGB_to_CIEXYZ_matrix).T
        self.linsRGB_to_LMS_matrix = self.linsRGB_to_LMS_matrix.to(self.device)

        one = torch.ones(1,1,3).to(self.device)
        self.white_point = one @ self.linsRGB_to_LMS_matrix
        self.white_point = self.white_point.to(self.device)

        # if not tetrachromatic
        if not self.is_tetrachromatic:
            self.LMS_to_linsRGB_matrix = torch.linalg.inv(self.linsRGB_to_LMS_matrix[:, :3])
            if self.device == 'mps:0': # MPS incorrectly handles the inverse of a matrix
                self.LMS_to_linsRGB_matrix = self.LMS_to_linsRGB_matrix.T
            # pad with zeros (3x3 -> 4x3 matrix)
            zeros = torch.zeros(1, 3).to(self.device)
            self.LMS_to_linsRGB_matrix = torch.cat([self.LMS_to_linsRGB_matrix, zeros], dim=0)


    # gamma correction
    def sRGB_to_linsRGB(self, sRGB):
        linRGB = sRGB / 12.92
        mask = sRGB > 0.04045
        linRGB[mask] = ((sRGB[mask] + 0.055)/1.055)**2.4
        return linRGB
    
    def linsRGB_to_sRGB(self, linRGB):
        mask = linRGB > 0.0031308
        sRGB = 12.92 * linRGB
        sRGB[mask] = 1.055 * linRGB[mask]**(1/2.4) - 0.055
        return sRGB

    def linsRGB_to_LMS(self, linsRGB):
        lms = linsRGB @ self.linsRGB_to_LMS_matrix
        return lms

    def LMS_to_linsRGB(self, lms):
        if self.is_tetrachromatic:
            return lms
        else:
            linsRGB = lms @ self.LMS_to_linsRGB_matrix
            return linsRGB
        

# 4.8493, -4.6649, 0.8482