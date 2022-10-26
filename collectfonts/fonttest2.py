import os
import glob
from PIL import Image, ImageDraw, ImageFont

os.makedirs('testresults', exist_ok=True)

fonts = glob.glob('fonts/*.ttf')

text = '「アート」\n【あーと】\n東京駅\n壱弐参'

for i, _font in enumerate(fonts):

    im = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    font = ImageFont.truetype(_font, 48)

    x = 20
    y = 20

    draw.multiline_text((x, y), text, fill=(0, 0, 255), font=font)
        
    im.save(os.path.join('testresults', f'{i}.jpg'))

