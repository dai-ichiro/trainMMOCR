from mmocr.utils.ocr import MMOCR

cfg = 'japanese_cfg.py'
img = 'demo/3.jpg'
checkpoint = 'output/latest.pth'

ocr = MMOCR(
    det = None,
    recog_config = cfg,
    recog_ckpt= checkpoint,
    device = 'cuda'
)

result = ocr.readtext(img, print_result=True, imshow=True)
