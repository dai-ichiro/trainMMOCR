from mmocr.utils.ocr import MMOCR
from mim.commands.download import download
import os
import numpy as np
import cv2

img = 'sampleimage/test1.png'

# Detection: fcenet
det_checkpoint_name = 'fcenet_r50dcnv2_fpn_1500e_ctw1500'
os.makedirs('models', exist_ok=True)
det_checkpoint = download(package='mmocr', configs=[det_checkpoint_name], dest_root="models")[0]

ocr = MMOCR(
    det_config = os.path.join('models', det_checkpoint_name + '.py'),
    det_ckpt = os.path.join('models', det_checkpoint),
    recog = None,
    device = 'cuda'
)

result = ocr.readtext(img, imshow=True) # -> list (len: 1)
result = result[0]                      # -> dict (key: [boundary_result])
result = result['boundary_result']      # -> list (len: number of bboxes)

os.makedirs('trim', exist_ok=True)
original_image = cv2.imread(img)
for i, each_array in enumerate(result):

    poly = np.array(each_array[:-1]).reshape(-1, 1, 2).astype(np.float32) 

    x, y, width, height = cv2.boundingRect(poly)
    trim_image = original_image[y:y+height, x:x+width, :]

    cv2.imwrite(os.path.join('trim', f'{i}.jpg'), trim_image)