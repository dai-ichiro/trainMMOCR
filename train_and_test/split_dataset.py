import random

with open('labels.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]

num = len(lines)
train_num = int(num * 0.95)

train_list = random.sample(lines, train_num)
test_set = set(lines) - set(train_list)

with open('train_label.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_list))

with open('test_label.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_set))
