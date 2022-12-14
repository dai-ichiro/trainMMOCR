import os
from mmcv import Config
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector

def main():
    cfg = Config.fromfile('SAR_japanese_cfg.py')

    os.makedirs('output', exist_ok=True)

    ####
    ## modify configuration file
    ####

    # set output dir
    cfg.work_dir = 'output_retrain'

    # Path to annotation file
    cfg.train_ann_file = 'train/labels.txt'
    cfg.train.ann_file= 'train/labels.txt'

    cfg.test_ann_file = 'test/labels.txt'
    cfg.test.ann_file = 'test/labels.txt'

    # Paht to image folder
    cfg.train.img_prefix = 'train'
    cfg.test.img_prefix = 'test'

    # Dict file
    cfg.dict_file = 'dicts.txt'
    cfg.label_convertor.dict_file = 'dicts.txt'
    cfg.model.label_convertor.dict_file = 'dicts.txt'

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
    cfg.data.samples_per_gpu = 8
    cfg.log_config.interval = 1000

    cfg.load_from = 'output/latest.pth'

    cfg.dump('new_SAR_cfg.py')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)

    model.CLASSES = datasets[0].CLASSES
    #model.init_weights()
    
    train_detector(model, datasets, cfg, validate=True)

if __name__ == '__main__':
    main()
