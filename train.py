from utils import utils_data
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.cuda import amp
from model import rlfn
import cv2
import numpy as np
from math import log10
import time
from options import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = utils_data.set_dataloader(image_size=args.image_size, upscale_factor=args.upscale_factor, aug_factor=args.aug_factor, batch_size=args.batch_size, datatype='train')
eval_dataloader = utils_data.set_dataloader(image_size=args.image_size, upscale_factor=args.upscale_factor, aug_factor=args.aug_factor, batch_size=args.batch_size, datatype='valid')


print('===> Building model')
model = rlfn.RLFN(upscale=args.upscale_factor).to(device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

def train(epoch):
    epoch_loss = 0
    for iteration, (lr, hr) in enumerate(train_dataloader):
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        prediction = model(lr)
        if iteration == 0 :
            print(lr.shape)
            print(prediction.shape)
            print(hr.shape)
        loss = criterion(prediction, hr)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()


        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_dataloader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def validation():
    avg_psnr = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            lr, hr = batch[0].to(device), batch[1].to(device)
            prediction = model(lr)
            mse = criterion(prediction, hr)
            psnr = 10 *log10(1 / mse.item())
            avg_psnr += psnr

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(eval_dataloader)))

def checkpoint(epoch):

    model_folder = "model_zoo/model_x{}_/".format(args.upscale_factor)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))




    """model_out_path = "model_zoo/model_epoch_{}_x{}.pth".format(epoch, upscale_factor)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))"""

def start_train():
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(epoch)
        validation()
        scheduler.step()
        checkpoint(epoch)

        print("elapsed time :",(time.time() - start_time), "sec")

    end_time = time.time()

    print("Average elapsed time :",(end_time - start_time) // epochs, "sec")

if __name__ == '__main__':
    start_train()


"""이미지 확인
def imshow(img, name):
    img = img/2 + 0.5

    np_img = img.numpy()
    print(f'np_img : {np_img.shape}')
    print((np.transpose(np_img, (1, 2, 0))).shape)
    result_img = np.transpose(np_img, (1, 2, 0))
    #plt.imshow(np.transpose(np_img, (1,2,0)))

    cv2.imshow(name, result_img)
    cv2.waitKey(0)




sr = None
for iteration, batch in enumerate(train_dataloader, 1):
    input = batch[0].to(device)
    input_ = batch[1].to(device)
    print(type(input_), input.shape)
    sr = model(input_)
    break

imshow(input_[1].detach().cpu(), 'lr')
imshow(sr[1].detach().cpu(), 'sr')
"""




