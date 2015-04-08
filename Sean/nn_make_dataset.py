# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:45:36 2015

@author: seanvig2
"""

import os
from pylearn2.utils import serial

from py_utils import heightdataset


def main():
    path = os.path.join(heightdataset.jpl_dir, 'datasets')

    train = heightdataset.HeightDataset(which_set='train')

    train.use_design_loc(os.path.join(path, 'height_dataset_train.npy'))
    train_pkl_path = os.path.join(path, 'height_dataset_train.pkl')
    serial.save(train_pkl_path, train)

    valid = heightdataset.HeightDataset(which_set='valid')

    valid.use_design_loc(os.path.join(path, 'height_dataset_valid.npy'))
    valid_pkl_path = os.path.join(path, 'height_dataset_valid.pkl')
    serial.save(valid_pkl_path, valid)


if __name__ == "__main__":
    main()
