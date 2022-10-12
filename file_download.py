import math
import wget
from zipfile import ZipFile
from requests import get
from tqdm import tqdm

def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
    return progress

def download(url):
    wget.download(url, bar=bar_custom)

def ext_zip(filename):
    with ZipFile(filename, 'r') as zip:
        zip.extractall('images')
        print('unzip over')

if __name__ == '__main__':
    url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
    url_val = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
    #wget.download(url_val, bar=bar_custom)
    ext_zip('DIV2K_valid_HR.zip')
