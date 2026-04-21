import argparse
import os

# import pytorch_ssim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch.optim as op
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
# import tqdm
from torch import nn
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from loss import LaplacianLoss
from data_set_py.data_utils_RS_1 import TrainDatasetFromFolder, ValDatasetFromFolder

########### change here ################
from models.CFGFE import CFGFE
from img_index import ref_evaluate
import pandas as pd
from datetime import datetime
from Pansharpening_Toolbox_Assessment_Python.indexes_evaluation import indexes_evaluation
import time
from torch.nn import functional as FC
import cv2
import thop
from helpers import make_patches


############# need to cahnge here !!!!
sate = 'ik'
#sate = 'pl'
#sate = 'wv3_8'
model_name='CFGFE'
num_chanel = 4
dataset_dir = 'D:\\liurixian\\remote_sensing_image_fusion\\Source Images\\'

# patch_size = 32
val_step = 5

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')  ### need to change
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')  # for 8 band 500
parser.add_argument('--batch_size', default=16, type=int, help='train epoch number')  ###       to change
parser.add_argument('--lr', type=float, default=0.0003,  # for 8 band , need to change !!
                    help='Learning Rate. Default=0.01')  # for 8 band 0.0006, for 4 band half
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by "
                                                          "momentum every n epochs, Default: n=500")
parser.add_argument("--log_path", type=str, default="training_results\\")
### model setting
parser.add_argument('--hsi_channel', type=int, default=4)
parser.add_argument('--msi_channel', type=int, default=1)

### device setting
parser.add_argument('--device', type=str, default='cuda:0')

opt = parser.parse_args()
opt.n_bands = 4
opt.image_size = 64
opt.n_bands_rgb = 1

CROP_SIZE = opt.crop_size  
UPSCALE_FACTOR = opt.upscale_factor 
NUM_EPOCHS = opt.num_epochs  
BATCH_SIZE = opt.batch_size

train_set = TrainDatasetFromFolder(dataset_dir, sate, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)  
val_set = ValDatasetFromFolder(dataset_dir, sate, crop_size=CROP_SIZE,upscale_factor=UPSCALE_FACTOR) 
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)  
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)



# ================== Pre-Define =================== #
SEED = 15
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

if torch.cuda.is_available():
    print("-------running on GPU-------------")
    device=torch.device("cuda:0")
else:
    print("-------running on CPU-------------")
    device=torch.device("cpu")

model = CFGFE(16).cuda()

print('# Model parameters:', sum(param.numel() for param in model.parameters()))
num_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: {:.3f}M".format(num_params / 1e6))
input1 = torch.randn(BATCH_SIZE, num_chanel, CROP_SIZE//4, CROP_SIZE//4).cuda()  ## need to change #######
input2 = torch.randn(BATCH_SIZE, num_chanel, CROP_SIZE, CROP_SIZE).cuda()
input3 = torch.randn(BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE).cuda()

flops, params = thop.profile(model, inputs=(input1,input2, input3))
print("FLOPs: {:.2f}G, Params: {:.3f}M".format(flops / 1e9, params / 1e6))

optimizerG = op.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    if lr < 0.00005:
        lr = 0.00005
    return lr

def gain(image):
    image = image.cuda()
    n, c, w, h = image.shape
    # sum_image[1, w, h] = torch.zeros(1, w, h)
    g = torch.zeros(n, c, w, h).cuda()
    for i in range(n):
        sum_image = torch.zeros(1, 1, w, h).cuda()
        for j in range(c):
            sum_image += image[i, j]
        # sum_image = 1 / c * torch.cat([sum_image, sum_image, sum_image, sum_image], 1)
        sum_image_avg = 1 / c * torch.cat([sum_image] * c, dim=1)
        # sum_image = sum_image.squeeze()
        g[i] = image[i] / sum_image_avg

    return g


def get_detail(ms_up, pan):
    kernel_size = 5
    sigma = 1.0
    pan = pan.cuda()
    ms_up = ms_up.cuda()

    ksize = int(2 * np.ceil(2 * sigma) + 1)
    gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
    gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel.transpose())
    gaussian_kernel = torch.from_numpy(gaussian_kernel).unsqueeze(0).unsqueeze(0).float()
    
    gaussian_kernel = gaussian_kernel.to('cuda')

    output = FC.conv2d(pan.to('cuda'), gaussian_kernel, padding=ksize // 2)
    detail1 = pan - output
    g = gain(ms_up)
    out_f = detail1 * g
    # out_f = (out_f + 0.5)/2
    return out_f


results = {'time': [], 'epoch': [], 'lr': [], 'mse_loss': [], 'l1_loss_d': [], 'l1_loss': [], 'total_loss': [],
           'psnr': [],
           'sam': [],
           'ergas': [], 'scc': [], 'q_avg': [], 'q2n': []}

out_path = 'training_results/'  # 输出路径


# val_results = {'mse_loss': [], 'tv_loss': [], 'lap_loss': [], 'total_loss': []}
writer = SummaryWriter()
count = 0

lr = opt.lr
# for epoch in range(1, NUM_EPOCHS + 1):

#########  start from saved models ########################
log_dir = ' '


if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint)
    # optimizerG.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']
    start_epoch = 165
    print('加载 epoch  成功！')
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')

t = time.strftime("%Y%m%d%H%M")

for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    train_bar = tqdm(train_loader)  # 进度条

    # print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    lr = adjust_learning_rate(optimizerG, epoch)  ##### need to chagne here!!!

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr

  
    running_results = {'mse_loss': 0, 'l1_loss_d': 0, 'l1_loss': 0, 'total_loss': 0, 'batch_sizes': 0}

    model.train()
    # model.eval()
    for ms_up_crop, ms_org_crop, pan_crop, gt_crop in train_bar:
        g_update_first = True
        batch_size = ms_up_crop.size(0)
        running_results['batch_sizes'] += batch_size  # pytorch batch_size 

        ms_up = Variable(ms_up_crop)
        ms_org = Variable(ms_org_crop)
        pan = Variable(pan_crop)
        # detail = Variable(detail)
        # pan_d = Variable(pan_d)
        gt = Variable(gt_crop)
        # ms_d_up = Variable(ms_d_up)
        if torch.cuda.is_available():
            ms_up = ms_up.cuda()
            ms_org = ms_org.cuda()
            pan = pan.cuda()
            gt = gt.cuda()


        out = model(ms_org,ms_up,pan)
        out_image = out
        size_n = int(out_image.size()[2] / 4)
        out_d = FC.interpolate(out_image, size=(size_n, size_n), mode='bicubic')
        target = gt

        # target, unfold_shape = make_patches(target, patch_size=patch_size)

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        optimizerG.zero_grad()  # change
        l1 = nn.L1Loss()
        l1_loss = l1(out_image, target)
        ######## use D_S and D_L
        #
        mse = nn.MSELoss()
        mse_loss = mse(out_image, target)

        total_loss = l1_loss
        total_loss.requires_grad_(True)
        total_loss.backward()
        # optimizerD.step()
        optimizerG.step()

        #### mse_loss, tv_loss, lap_loss, total_loss ####
        running_results['mse_loss'] += mse_loss.item() * batch_size  ### need to change
        # running_results['l1_loss_d'] += l1_loss_d.item() * batch_size
        running_results['l1_loss'] += l1_loss.item() * batch_size
        running_results['total_loss'] += total_loss.item() * batch_size

        train_bar.set_description(desc='lr:%f [%d/%d] mse_loss: %.5f l1_loss: %.5f total_loss: %.5f' % (
            lr, epoch, NUM_EPOCHS, running_results['mse_loss'] / running_results['batch_sizes'],
            running_results['l1_loss'] / running_results['batch_sizes'],
            running_results['total_loss'] / running_results['batch_sizes']))
        writer.add_scalar('mse_loss', running_results['mse_loss'] / running_results['batch_sizes'], count)
        # writer.add_scalar('l1_loss_d', running_results['l1_loss_d'] / running_results['batch_sizes'], count)
        writer.add_scalar('l1_loss', running_results['l1_loss'] / running_results['batch_sizes'], count)
        writer.add_scalar('total_loss', running_results['total_loss'] / running_results['batch_sizes'], count)
        count += 1
    model.eval()
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ############ 验证集 #################
    if epoch % val_step == 0:
        val_bar = tqdm(val_loader)  # 验证集的进度条
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        # for data, ms_gray_crop, pan_crop, ms_up_crop, gt_crop in train_bar:
        l_psnr = []
        l_sam = []
        l_ergas = []
        l_scc = []
        l_q = []
        l_q2n = []
        for ms_up_crop, ms_org_crop, pan_crop, gt_crop in val_bar:
            batch_size = ms_up_crop.size(0)
            print(batch_size)
            size_n = pan_crop.shape[2]
            valing_results['batch_sizes'] += batch_size

            with torch.no_grad():  # validation
                ms_up = Variable(ms_up_crop)
                ms_org = Variable(ms_org_crop)
                pan = Variable(pan_crop)
                if torch.cuda.is_available():
                    ms_up = ms_up.cuda()
                    ms_org = ms_org.cuda()
                    pan = pan.cuda()

                out = model(ms_org, ms_up, pan)

            # v_pan = v_pan.cuda()
            # val_out = model(v_data1, v_data2)  # 验证集生成图片
            output = out.cpu()
            count = 0
            for i in range(batch_size):
                val_images = []
                val_out = out.data[i].cpu().squeeze(0)
                val_gt0 = gt_crop.data[i].cpu().squeeze(0)

                val_fused = val_out
                val_gt = val_gt0

                val_rgb = val_fused[0:3]
                val_gt_rgb = val_gt[0:3]

                # val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])
                val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])

                ##############  index evaluation ######################
                val_gt_np = val_gt.numpy().transpose(1, 2, 0)
                val_fused_np = val_fused.numpy().transpose(1, 2, 0)

                val_images = utils.make_grid(val_images, nrow=2, padding=5)
                utils.save_image(val_images, out_path + sate + '/images/' + sate + '_tensor_%d.tif' % i)

                [c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q] = ref_evaluate(val_fused_np, val_gt_np)
                # [Q2n_index, Q_index, ERGAS_index, SAM_index] = [0, 0, 0, 0]
                [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(val_fused_np, val_gt_np, 4, 8, 32, 1,
                                                                                  1,
                                                                                  1)
                l_psnr.append(c_psnr)
                l_sam.append(SAM_index)
                l_ergas.append(ERGAS_index)
                l_scc.append(c_scc)
                l_q.append(Q_index)
                l_q2n.append(Q2n_index)

        ##### finish val_bar ################
        psnr_avg = np.mean(l_psnr)
        sam_avg = np.mean(l_sam)
        ergas_avg = np.mean(l_ergas)
        scc_avg = np.mean(l_scc)
        q_avg = np.mean(l_q)
        q2n_avg = np.mean(l_q2n)

        print(
            'psnr:{:.4f}, sam:{:.4f}, ergas:{:.4f}, scc:{:.4f}, q:{:.4f},q2n:{:.4f}'.format(psnr_avg, sam_avg,
                                                                                            ergas_avg,
                                                                                            scc_avg, q_avg, q2n_avg))

        torch.save(model.state_dict(),
                   'model_trained/' + sate + '/' + sate + '_'+model_name+'_epoch_%03d.pth' % epoch)  # 存储网络参数

        #### save to excel  ####
        time_curr = "%s" % datetime.now()  # 获取当前时间
        results['time'].append(time_curr)
        results['epoch'].append(epoch)
        results['lr'].append(lr)
        results['mse_loss'].append(running_results['mse_loss'] / running_results['batch_sizes'])
        results['l1_loss_d'].append(running_results['l1_loss_d'] / running_results['batch_sizes'])
        results['l1_loss'].append(running_results['l1_loss'] / running_results['batch_sizes'])
        results['total_loss'].append(running_results['total_loss'] / running_results['batch_sizes'])
        results['psnr'].append(psnr_avg)
        results['sam'].append(sam_avg)
        results['ergas'].append(ergas_avg)
        results['scc'].append(scc_avg)
        results['q_avg'].append(q_avg)
        results['q2n'].append(q2n_avg)

        # train_log = open(os.path.join(opt.log_path, sate, "%s_%s_train.log") % (sate, t), "w")
        df = pd.DataFrame(results)  ###############################  need to change!!!
        df.to_excel(out_path + sate + '/' + sate +'_'+model_name+ f'_{t}.xlsx', index=False)  #### need to change here!!!
writer.close()
