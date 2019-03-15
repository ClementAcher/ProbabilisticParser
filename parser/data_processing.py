from os.path import join

import numpy as np

# Size of each dataset
TRAIN_DATASET_SIZE = 0.8
DEV_DATASET_SIZE = 0.1
TEST_DATASET_SIZE = 0.1

DATASET_NAMES = ['train.txt', 'dev.txt', 'test.txt']
DATA_PATH = './../data/'


def split_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = f.readlines()

    train_idx = int(TRAIN_DATASET_SIZE * len(dataset))
    dev_idx = train_idx + int(DEV_DATASET_SIZE * len(dataset))

    datasets = np.split(dataset, (train_idx, dev_idx))

    for split_ds, name in zip(datasets, DATASET_NAMES):
        with open(join(DATA_PATH, name), 'w') as f:
            f.writelines(split_ds)

    return True

split_dataset('./../data/sequoia-corpus+fct.mrg_strict')
