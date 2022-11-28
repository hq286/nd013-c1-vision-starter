import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger

TRAIN_PERCENTAGE = 0.7 
VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
SEED = 42

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in "data" into train and val sets.
    try:
        all_tfrecords = glob.glob('{}/*.tfrecord'.format(data_dir + "/" + "processed"))
    except Exception as error:
        print('Unable to access data files!')
    np.random.seed(SEED)
    np.random.shuffle(all_tfrecords)
    print('There are ' + str(len(all_tfrecords)) + ' tfrecords in total')

    # split the data
    dataset_size = len(all_tfrecords)
    train_size = int(TRAIN_PERCENTAGE*dataset_size)
    val_size = int(VAL_PERCENTAGE*dataset_size)
    train_data, val_data, test_data = np.split(all_tfrecords, [train_size, train_size + val_size])
    
    # move files into destination directories
    train_path = os.path.join(data_dir, 'train')
    for tfrecord in train_data:
        shutil.move(tfrecord, train_path)
    
    val_path = os.path.join(data_dir, 'val')
    for tfrecord in val_data:
        shutil.move(tfrecord, val_path)
    
    test_path = os.path.join(data_dir, 'test')
    for tfrecord in test_data:
        shutil.move(tfrecord, test_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)