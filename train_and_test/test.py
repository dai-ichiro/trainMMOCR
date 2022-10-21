from mmocr.utils.ocr import MMOCR

cfg = 'japanese_cfg.py'
img = 'img/img_0.jpg'
checkpoint = 'output/latest.pth'
out_file = 'result.json'

ocr = MMOCR(
    det = None,
    recog_config = cfg,
    recog_ckpt= checkpoint,
    device = 'cuda'
)

result = ocr.readtext(img, print_result=True, imshow=True)