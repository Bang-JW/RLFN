import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256, help='INSERT IMAGE_SIZE')
parser.add_argument('--aug_factor', type=int, default=2, help='INSERT AUGUMENTATION FACTOR')
parser.add_argument('--upscale_factor', type=int, default=4, help='INSERT UPSCALE FACTOR')
parser.add_argument('--batch_size', type=int, default=32, help='INSERT BATCH SIZE')
parser.add_argument('--epochs', type=int, default=1000, help='INSERT EPOCHS')
args = parser.parse_args()
