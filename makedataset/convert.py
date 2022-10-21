import glob
import os
import shutil
import random

from torch import rand

all_images = glob.glob('out/*.jpg')

os.makedirs('img', exist_ok=True)

label_list = []
for i, image in enumerate(all_images):
    text = os.path.basename(image).split('_')[0]
    out_fname = f'img_{i}.jpg'
    shutil.copy(image, os.path.join('img', out_fname))
    label_list.append(f'{out_fname} {text}')

num = len(label_list)
train_num = int(num * 0.9)

train_list = random.sample(label_list, train_num)
test_set = set(label_list) - set(train_list)

with open('train_label.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_list))

with open('test_label.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_set))
