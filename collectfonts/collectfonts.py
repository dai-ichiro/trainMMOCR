import os
import glob
import shutil
from torchvision.datasets.utils import download_url

# Copy from Windows 11
font_index = [84, 85, 179, 180, 181]
fonts = glob.glob('c://Windows/Fonts/*.ttf')

os.makedirs('fonts', exist_ok=True)

for i in font_index:
    fname = os.path.basename(fonts[i])
    fname = fname.replace('.TTF', '.ttf')
    shutil.copy(fonts[i], os.path.join('fonts', fname))
    
# Download from mmocr
font_url = 'https://download.openmmlab.com/mmocr/data/font.TTF'
font_fname = 'mmocr.ttf'
download_url(font_url, root = 'fonts', filename = font_fname)

# Download from TextRecognitionDataGenerator
font_url = 'https://raw.githubusercontent.com/Belval/TextRecognitionDataGenerator/master/trdg/fonts/ja/TakaoMincho.ttf'
font_fname = font_url.split('/')[-1]
download_url(font_url, root = 'fonts', filename = font_fname)