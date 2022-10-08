import csv
import os
import time

import torch
import cv2

from utils import utils_data
from utils.utils_image import tensor2np, calculate_psnr, calculate_ssim
from options import args
from model import rlfn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    data_dir = os.getcwd() + '/data/' if args.on_local else '/home/shh950422/images/test/'

    test_dataloader = utils_data.testset_dataloader(data_dir=data_dir, upscale_factor=args.upscale_factor)

    model_dir = f'/model_test/x{args.upscale_factor}/'
    model_names = [model for model in os.listdir(os.getcwd() + model_dir)
                   if model.startswith('epoch')]
    if len(model_names) == 0:
        raise FileNotFoundError(f"There's no model file in {model_dir}")

    model = rlfn.RLFN(upscale=args.upscale_factor).to(device)
    model_state_dict = torch.load('.' + model_dir + model_names[0], map_location=device)['model_state_dict']
    model.load_state_dict(model_state_dict)

    log_file = f'test_x{args.upscale_factor}'
    with open(log_file, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(('index', 'filename', 'runtime', 'psnr', 'ssim'))

    filenames = list(map(lambda name: name.split('/')[-1], test_dataloader.dataset.filenames))
    for idx, (lr, hr) in enumerate(test_dataloader):
        start_time = time.time()

        lr, hr = lr.to(device), hr.to(device)

        sr = model(lr)

        lr_img = tensor2np(lr[0])
        sr_img = tensor2np(sr[0])
        hr_img = tensor2np(hr[0])
        if idx < args.display_num:
            cv2.imshow('LR / SR / HR', cv2.hconcat([cv2.resize(lr_img, dsize=(0, 0), fx=args.upscale_factor, fy=args.upscale_factor), sr_img, hr_img]))
            cv2.waitKey(0)

        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_RGB2YCrCb)[:, :, 0] * 255
        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YCrCb)[:, :, 0] * 255

        psnr = calculate_psnr(sr_y, hr_y)
        ssim = calculate_ssim(sr_y, hr_y)
        print(f'{idx:>4d}, {psnr:>15f}, {ssim:>15f}')


if __name__ == '__main__':
    main()













