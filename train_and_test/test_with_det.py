from mmocr.utils.ocr import MMOCR
from mim.commands.download import download
import os

cfg = 'japanese_cfg.py'
img = 'sampleimage/test1.png'
checkpoint = 'output/latest.pth'

# Detection: PANet_IC15
det_checkpoint_name = 'fcenet_r50dcnv2_fpn_1500e_ctw1500'
os.makedirs('models', exist_ok=True)
det_checkpoint = download(package='mmocr', configs=[det_checkpoint_name], dest_root="models")[0]

ocr = MMOCR(
    det_config = os.path.join('models', det_checkpoint_name + '.py'),
    det_ckpt = os.path.join('models', det_checkpoint),
    recog_config = cfg,
    recog_ckpt= checkpoint,
    device = 'cuda'
)

result = ocr.readtext(img, print_result=True, imshow=True)