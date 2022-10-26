import sys
from mmocr.utils.ocr import MMOCR

img = sys.argv[1]

cfg = 'new_SATRN_cfg.py'
checkpoint = 'satrn_output/latest.pth'

ocr = MMOCR(
    det = None,
    recog_config = cfg,
    recog_ckpt= checkpoint,
    device = 'cuda'
)

result = ocr.readtext(img, print_result=True, imshow=True)
