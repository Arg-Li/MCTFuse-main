import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
from PIL import Image
#from flowlib import read, read_weights_file
from skimage import io, transform
from PIL import Image
import numpy as np
import re
from utils import *
import matplotlib.pyplot as plt


def toString(num):
    string = str(num)
    while len(string) < 4:
        string = "0" + string
    return string


class FusionDataset(Dataset):

    def __init__(self, dataroot, image_size):
        """

        """
        self.infrared_dataroot = os.path.join(dataroot, 'ir')
        self.visible_dataroot = os.path.join(dataroot, 'vis')
        self.image_size = image_size
        self.total_image = []


        ir_img_dir = self.infrared_dataroot
        ir_image_list = os.listdir(os.path.join(self.infrared_dataroot))
        ir_image_list.sort(key=lambda x: str(re.split('[._]', x)[1]))
        vis_img_dir = self.visible_dataroot
        vis_image_list = os.listdir(os.path.join(self.visible_dataroot))
        vis_image_list.sort(key=lambda x: str(re.split('[._]', x)[1]))
        data_len = len(vis_image_list) - 1

        for i in range(data_len):
            ir_img = os.path.join(ir_img_dir, ir_image_list[i])
            vis_img = os.path.join(vis_img_dir, vis_image_list[i])
            tmp_img = (ir_img, vis_img)
            self.total_image.append(tmp_img)
        self.lens = len(self.total_image)
        self.transform = transforms.Compose([
            transforms.ToTensor()])
           

    def __len__(self):
        return self.lens

    def __getitem__(self, i):
        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        ir_path1 = self.total_image[i][0]
        vis_path1 = self.total_image[i][1]

        ir_img1 = Image.open(ir_path1).convert('L')
        ir_img = self.transform(ir_img1)
        vis_img1 = Image.open(vis_path1).convert('L')
        vis_img = self.transform(vis_img1)
        return ir_img, vis_img







