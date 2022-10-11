import os
from utils import utils_image

lr_dir = 'C:\\Users\\bjw97\PycharmProjects\RLFN\data\\USR-248_LR_results'
hr_dir = 'C:\\Users\\bjw97\PycharmProjects\RLFN\data\\USR-248_HR'

lr_result = os.listdir('C:\\Users\\bjw97\PycharmProjects\RLFN\data\\USR-248_LR_results')
hr = os.listdir('C:\\Users\\bjw97\PycharmProjects\RLFN\data\\USR-248_HR')

img1 = utils_image.imread_uint(os.path.join(lr_dir, lr_result[0]))
img2 = utils_image.imread_uint(os.path.join(hr_dir, hr[0]))
print(utils_image.calculate_psnr(img1, img2))
print(utils_image.calculate_ssim(img1, img2))

for i in range(len(hr)):
    img1 = utils_image.imread_uint(os.path.join(lr_dir, lr_result[i]))
    img2 = utils_image.imread_uint(os.path.join(hr_dir, hr[i]))
    print(lr_result[i],"PSNR : ",utils_image.calculate_psnr(img1, img2))
    print(lr_result[i],"SSIM : ",utils_image.calculate_ssim(img1, img2))


