import os
from mmcv import Config
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector

def main():
    cfg = Config.fromfile('SATRN_japanese_cfg.py')

    os.makedirs('satrn_output', exist_ok=True)

    ####
    ## modify configuration file
    ####

    # set output dir
    cfg.work_dir = 'output'

    # Path to annotation file
    cfg.train.ann_file= 'train/labels.txt'
    cfg.test.ann_file = 'test/labels.txt'

    # Paht to image folder
    cfg.train.img_prefix = 'train'
    cfg.test.img_prefix = 'test'

    # Modify label_convertor
    cfg.label_convertor.dict_file='dicts.txt'
    cfg.label_convertor.max_seq_len = 40
    
    # Modify data
    cfg.data.train.datasets = [cfg.train]
    cfg.data.val.datasets = [cfg.test]
    cfg.data.test.datasets = [cfg.test]

    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # Others
    cfg.optimizer.lr = 0.001 /8
    cfg.seed = 0
    cfg.runner.max_epochs = 1 # default 5 
    cfg.data.samples_per_gpu = 16
    cfg.log_config.interval = 1000

    cfg.dump('new_SATRN_cfg.py')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    model.init_weights()

    train_detector(model, datasets, cfg, validate=True)

if __name__ == '__main__':
    main()
