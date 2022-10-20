import os
import glob
import shutil

font_index = [135, 136, 137]
fonts = glob.glob('c://Windows/Fonts/*.ttf')

os.makedirs('fonts', exist_ok=True)

for i in font_index:
    fname = os.path.basename(fonts[i])
    fname = fname.replace('.TTF', '.ttf')
    shutil.copy(fonts[i], os.path.join('fonts', fname))
    
    