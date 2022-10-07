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

if args.load_num != -1:
    PATH = f'model_zoo/model_x{args.upscale_factor}_/epoch_{args.load_num}.pth'
    if not os.path.isfile(PATH):
        raise FileNotFoundError(f'given num is {args.load_num}. But {PATH} does not exist')
    
    print('=====> Loading model')
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5, last_epoch=args.load_num)

def train(epoch, loss_log_path):
    epoch_loss = 0
    for iteration, (lr, hr) in enumerate(train_dataloader):
        start_time = time.time()

        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        prediction = model(lr)
        loss = criterion(prediction, hr)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        logging(loss_log_path, type='loss',
                epoch=epoch,
                iteration=iteration,
                loss=loss.item(),
                time=time.time() - start_time)

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


def checkpoint(epoch, model_folder, loss, psnr):
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
        }, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def make_log_file(model_folder, type='loss', load=False):
    if type not in ('loss', 'psnr'):
        print('check your param "type"')
        print(f'expected : loss or psnr, given : {type}')

    log_path = model_folder + f'{type}_log.csv'
    if not load:
        with open(log_path, 'w', newline='') as logfile:
            writer = csv.writer(logfile)
            header = None
            if type == 'loss':
                header = ('epoch', 'iteration', 'loss', 'time')
            elif type == 'psnr':
                header = ('epoch', 'psnr', 'time')
            writer.writerow(header)
    return log_path


def logging(file_path, type='loss', **kwargs):
    with open(file_path, 'a', newline='') as logfile:
        writer = csv.writer(logfile)
        row = None

        epoch = kwargs['epoch']
        time = kwargs['time']
        if type == 'loss':
            iteration = kwargs['iteration']
            loss = kwargs['loss']
            row = (epoch, iteration, loss, time)
        elif type == 'psnr':
            psnr = kwargs['psnr']
            row = (epoch, psnr, time)
        if row is None:
            return
        writer.writerow(row)


def print_args(args):
    print(f"crop size : {args.image_size}")
    print(f"aug factor : {args.aug_factor}")
    print(f"upscale factor : {args.upscale_factor}")
    print(f"batch size : {args.batch_size}")
    print(f"epochs : {args.epochs}")
    print(f"step size : {args.step_size}")


def start_train():
    print_args(args)
    load = args.load_num != 0

    model_folder = f"model_zoo/model_x{args.upscale_factor}_/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    loss_log_path = make_log_file(model_folder, type='loss', load=load)
    psnr_log_path = make_log_file(model_folder, type='psnr', load=load)

    total_train_time = 0
    for epoch in range(args.load_num + 1, args.epochs):
        train_time = 0
        start_time = time.time()

        loss = train(epoch=epoch, loss_log_path=loss_log_path)
        psnr = 'skip'

        train_time += time.time() - start_time
        if epoch % 10 == 0:
            psnr = validation()
        start_time = time.time()

        scheduler.step()
        checkpoint(epoch=epoch, model_folder=model_folder, loss=loss, psnr=psnr)

        train_time += time.time() - start_time
        print(f"train time : {train_time:.3f}sec")
        total_train_time += train_time

        logging(psnr_log_path, type='psnr',
                epoch=epoch,
                psnr=psnr,
                time=train_time)

    print(f"Average train time : {total_train_time / (args.epochs - args.load_num - 1):.3f}sec")


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




