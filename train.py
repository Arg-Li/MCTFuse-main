import os
import sys
import argparse
import torch
import datetime
import warnings
from tqdm import tqdm
import random
import numpy as np
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from fusion_data import FusionDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import kornia
from network.TransNet import TransNet
from network.AutoEncoder import VIEncoder, IREncoder
from utils import *
import time


warnings.filterwarnings("ignore", category=UserWarning)



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=140, help='number of epochs')
parser.add_argument('--epoch_gap', type=int, default=60, help='encoder |-> fusion')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)  # 0.001
parser.add_argument('--dataset_path', type=str, default="./dataset/MSRS")
parser.add_argument('--weights', type=str, default='',
                    help='initial weights path')
parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
parser.add_argument("--image_size", type=int, default=[128, 128])


SSIMLoss = kornia.losses.SSIMLoss(11, reduction='mean')
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
criteria_fusion = Fusionloss()

if __name__ == '__main__':
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    if os.path.exists("./runs") is False:
        os.makedirs("./runs")
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = "./runs/train_{}".format(file_name)
    os.makedirs(file_path)
    file_weights_path = os.path.join(file_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(file_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    start_epoch = 0
    batch_size = opt.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    print("Loading fusion_dataset now !")
    train_dataset_path = os.path.join(opt.dataset_path, "train")
    train_dataset = FusionDataset(train_dataset_path, opt.image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)

    config = get_CTranS_config()
    model_auto_ir = IREncoder().to(device)
    model_auto_vi = VIEncoder().to(device)
    model = TransNet(config, train_flag=True).to(device)

    if opt.use_dp:
        model_auto_ir = torch.nn.DataParallel(model_auto_ir)
        model_auto_vi = torch.nn.DataParallel(model_auto_vi)
        model = torch.nn.DataParallel(model).cuda()

    if opt.weights != "":
        model_auto_ir.load_state_dict(torch.load(opt.weights)['model_auto_ir'])
        model_auto_vi.load_state_dict(torch.load(opt.weights)['model_auto_vi'])
        model.load_state_dict(torch.load(opt.weights)['model'])
        print("load weights successfully !!")

    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_auto_ir.parameters()), lr=opt.lr, weight_decay=0.005)
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_auto_vi.parameters()), lr=opt.lr, weight_decay=0.005)
    optimizer3 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.005)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=10, gamma=0.2)

    for epoch in range(start_epoch, opt.epochs):
        model_auto_vi.train()
        model_auto_ir.train()
        model.train()

        loss_vi = torch.zeros(1, device=device)
        loss_ir = torch.zeros(1, device=device)
        loss_ssim = torch.zeros(1, device=device)
        loss_fusion = torch.zeros(1, device=device)
        loss_total = torch.zeros(1, device=device)

        # train_loader = tqdm(train_loader, file=sys.stdout)
        itt = 0
        for step, data in enumerate(train_loader):
            itt = step + 1
            ir_img = data[0].to(device)
            vi_img = data[1].to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            if epoch < opt.epoch_gap: #phase I
                recon_ir = model_auto_ir(ir_img)
                recon_vi = model_auto_vi(vi_img)

                ir_loss = MSELoss(recon_ir, ir_img) + 5 * SSIMLoss(ir_img, recon_ir)
                vi_loss = MSELoss(recon_vi, vi_img) + 5 * SSIMLoss(vi_img, recon_vi)
                w = [1.0, 3.0]
                loss = w[0] * ir_loss + w[1] * vi_loss
                loss.backward()

                loss_ir += ir_loss
                loss_vi += vi_loss
                loss_total += loss

                lr = optimizer1.param_groups[0]["lr"]
                # train_loader.desc = "[train epoch {}/{}] loss:{:.6f} loss vi:{:.6f} loss ir:{:.6f} lr:{:.6f}".format(
                #     epoch, opt.epochs - 1, loss_total.item() / (step + 1), loss_vi.item() / (step + 1), loss_ir.item() / (step + 1), lr)
                optimizer1.step()
                optimizer2.step()
            else:  #phase II
                ir = model_auto_ir.fuse_foward(ir_img)
                vi = model_auto_vi.fuse_foward(vi_img)
                fused_img = model(vi, ir)

                fusion_loss, _, _ = criteria_fusion(vi_img, ir_img, fused_img)
                ssim_loss = 0.5 * SSIMLoss(vi_img, fused_img) + 0.5 * SSIMLoss(ir_img, fused_img)

                w = [1.0, 5.0]
                loss = w[0] * fusion_loss + w[1] * ssim_loss
                loss.backward()

                loss_ssim += ssim_loss
                loss_fusion += fusion_loss
                loss_total += loss
                lr = optimizer3.param_groups[0]["lr"]
                # train_loader.desc = "[train epoch {}/{}] loss:{:.6f} loss_fusion:{:.6f} loss ssim:{:.6f} lr:{:.6f}".format(
                #     epoch, opt.epochs - 1, loss_total.item() / (step + 1), loss_fusion.item() / (step + 1), loss_ssim.item() / (step + 1), lr)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

        if epoch < opt.epoch_gap:
            print("[{}] [train epoch {}/{}] [loss: {:.6f}] [loss vi:{:.6f}] [loss ir:{:.6f}]"
                  .format(time.ctime()[4:-5], epoch, opt.epochs - 1, loss_total.item() / itt, loss_vi.item() / itt, loss_ir.item() / itt))
        else:
            print("[{}] [train epoch {}/{}] [loss: {:.6f}] [loss fusion:{:.6f}] [loss ssim:{:.6f}]"
                  .format(time.ctime()[4:-5], epoch, opt.epochs - 1, loss_total.item() / itt, loss_fusion.item() / itt, loss_ssim.item() / itt))


        if optimizer1.param_groups[0]['lr'] <= 1e-7:
            optimizer1.param_groups[0]['lr'] = 1e-7
        if optimizer2.param_groups[0]['lr'] <= 1e-7:
            optimizer2.param_groups[0]['lr'] = 1e-7
        if optimizer3.param_groups[0]['lr'] <= 1e-7:
            optimizer3.param_groups[0]['lr'] = 1e-7
        if epoch < opt.epoch_gap:
            scheduler1.step()
            scheduler2.step()
            tb_writer.add_scalar("train_I_loss", loss_total.item() / itt, epoch)
            tb_writer.add_scalar("train_I_vi", loss_vi.item() / itt, epoch)
            tb_writer.add_scalar("train_I_ir", loss_ir.item() / itt, epoch)
        else:
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            tb_writer.add_scalar("train_II_loss", loss_total.item() / itt, epoch)
            tb_writer.add_scalar("train_II_fusion", loss_fusion.item() / itt, epoch)
            tb_writer.add_scalar("train_II_ssim", loss_ssim.item() / itt, epoch)



    ckpt = {
        'model_auto_ir': model_auto_ir.state_dict(),
        'model_auto_vi': model_auto_vi.state_dict(),
        'model': model.state_dict()
    }
    torch.save(ckpt, file_weights_path + "/" + "ckpt_last.pth")


