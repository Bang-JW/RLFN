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
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = utils_data.set_dataloader(image_size=args.image_size, upscale_factor=args.upscale_factor, aug_factor=args.aug_factor, batch_size=args.batch_size, datatype='train')
eval_dataloader = utils_data.set_dataloader(image_size=args.image_size, upscale_factor=args.upscale_factor, aug_factor=args.aug_factor, batch_size=args.batch_size, datatype='valid')


print('===> Building model')
model = rlfn.RLFN(upscale=args.upscale_factor).to(device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)


def train(epoch):
    epoch_loss = 0
    for iteration, (lr, hr) in enumerate(train_dataloader):
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        prediction = model(lr)
        loss = criterion(prediction, hr)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f"===> Epoch[{epoch}]({iteration}/{len(train_dataloader)}): Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}")
    return avg_loss


def validation():
    sum_psnr = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            lr, hr = batch[0].to(device), batch[1].to(device)
            prediction = model(lr)
            mse = criterion(prediction, hr)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr

    avg_psnr = sum_psnr / len(eval_dataloader)
    print(f"===> Avg. PSNR: {avg_psnr:.4f} dB")
    return avg_psnr


def checkpoint(epoch, model_folder):
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def make_log_file(model_folder):
    log_path = model_folder + 'log.csv'
    with open(log_path, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(('epoch', 'loss', 'psnr', 'time'))
    return log_path


def logging(file_path, epoch, loss, psnr, time):
    with open(file_path, 'a', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow((epoch, loss, psnr, time))


def start_train():
    model_folder = f"model_zoo/model_x{args.upscale_factor}_/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    log_path = make_log_file(model_folder)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        avg_loss = train(epoch=epoch)
        avg_psnr = validation()
        scheduler.step()
        checkpoint(epoch=epoch, model_folder=model_folder)
        elapsed_time = time.time() - start_time
        logging(log_path, epoch, avg_loss, avg_psnr, elapsed_time)
        print(f"elapsed time : {elapsed_time:.3f}sec")

    end_time = time.time()
    print(f"Average elapsed time : {(end_time - start_time) / args.epochs:.3f}sec")


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




