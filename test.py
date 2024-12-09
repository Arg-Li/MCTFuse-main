import argparse
from email.policy import default

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import time
from PIL import Image

from network.TransNet import TransNet
from network.AutoEncoder import VIEncoder, IREncoder
from utils import *
import ml_collections

dataset = "TNO"

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default=os.path.join('./test_images', dataset))
parser.add_argument('--weights', type=str, default='./runs/ckpt_val.pth',
                    help='initial weights path')
parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--save_path", default=os.path.join('./results', dataset), type=str)
parser.add_argument("--flag", type=bool, default=False, help="rgb-True or gray-False")



if __name__ == "__main__":
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    vi_path = os.path.join(opt.test_data_path, 'vis')
    ir_path = os.path.join(opt.test_data_path, 'ir')
    dirname_ir = os.listdir(vi_path)
    dirname_vi = os.listdir(ir_path)
    data_len = len(dirname_vi)
    transform = transforms.Compose([transforms.ToTensor()])

    config = get_CTranS_config()
    model_auto_ir = IREncoder().to(device)
    model_auto_vi = VIEncoder().to(device)
    model = TransNet(config, train_flag = False).to(device)

    ckpt_path = opt.weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model_auto_ir.load_state_dict(torch.load(ckpt_path, map_location=device)['model_auto_ir'])
    model_auto_vi.load_state_dict(torch.load(ckpt_path, map_location=device)['model_auto_vi'])
    model.eval()
    model_auto_vi.eval()
    model_auto_ir.eval()
    print("load model OK! ")

    with torch.no_grad():
        t = []
        for i in range(data_len):
            if i != 0:
                start = time.time()

            infrared = Image.open(os.path.join(ir_path, dirname_ir[i])).convert('L')
            infrared = transform(infrared).unsqueeze(0).to(device)
            if opt.flag:
                vis = Image.open(os.path.join(vi_path, dirname_vi[i]))
                vis = transform(vis)
                vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis)
                vis_y_image = vis_y_image.unsqueeze(0).to(device)
                visible = vis_y_image
                vis_cb_image = vis_cb_image.to(device)
                vis_cr_image = vis_cr_image.to(device)  # show color
            else:
                visible = Image.open(os.path.join(vi_path, dirname_vi[i])).convert('L')
                visible = transform(visible).unsqueeze(0).to(device)

            vi = model_auto_vi.fuse_foward(visible)
            ir = model_auto_ir.fuse_foward(infrared)
            fused_img = model(vi, ir)

            if i != 0:
                end = time.time()
                print('consume time:', end - start)
                t.append(end - start)

            fused_img = torch.squeeze(fused_img, 1)
            if opt.flag:
                fused_img = YCrCb2RGB(fused_img, vis_cb_image, vis_cr_image)  # show color

            fused_img = transforms.ToPILImage()(fused_img)
            filename, ext = os.path.splitext(dirname_ir[i])
            save_path = os.path.join(opt.save_path, filename + '.png')
            fused_img.save(save_path)

        print("mean:%s, std: %s" % (np.mean(t), np.std(t)))

