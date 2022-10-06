import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset


def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# 데이터 셋 생성 클래스
class Dataset(object):
    # (이미지 디렉토리, 패치 사이즈, 스케일, 텐서플로우를 이용한 이미지 로더 사용 여부) 초기화
    def __init__(self, images_dir, image_size, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])

    # lr & hr 이미지를 읽고 크롭하여 lr & hr 이미지를 반환하는 함수
    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)


def set_dataloader(image_size, upscale_factor, aug_factor, batch_size, datatype = 'train_HR'):
    #/home/shh950422/images/train/DIV2K_train_HR/
    #root_dir = 'C:\\Users\\bjw97\PycharmProjects\RLFN_Final\data'
    root_dir = '/home/shh950422/images/train/' 
    data_dir = os.path.join(root_dir, 'DIV2K_train_HR')

    image_size = calculate_valid_crop_size(image_size, upscale_factor)
    dataset = Dataset(data_dir, image_size, upscale_factor)
    if datatype == 'train_HR':
        dataset = ConcatDataset([dataset] * aug_factor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    return dataloader


