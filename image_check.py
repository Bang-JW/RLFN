import cv2
import numpy as np
import torch
import os
from model import rlfn
from utils import utils_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

testset_L = 'USR-248_LR'
test_folder = os.path.join('data', testset_L)
result_folder = os.path.join('data', testset_L+'_results')
utils_image.mkdir(result_folder)

model_path = os.path.join('model_zoo\model_x4_', 'epoch_36.pth')
model = rlfn.RLFN(in_channels=3, out_channels=3)
model.load_state_dict(torch.load(model_path), strict=True)
#model.eval()
#for k, v in model.named_parameters():
#    v.requires_grad = False
model = model.to(device)

idx = 0

for img in utils_image.get_image_paths(test_folder):

    idx += 1
    img_name, ext = os.path.splitext(os.path.basename(img))


    img_L = utils_image.imread_uint(img, n_channels=3)
    img_L = utils_image.uint2tensor4(img_L)
    img_L = img_L.to(device)

    img_E = model(img_L)
    img_E = utils_image.tensor2uint(img_E)

    utils_image.imsave(img_E, os.path.join(result_folder, img_name[:11] + ext))

