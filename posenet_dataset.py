# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


from drgk_pose import data as DS

datasets_root = "D:/two_keypoint"  # 这个需要专门指定目录


class Options(object):
    def __init__(self):
        self.image_size, self.batch_size = DS.IMAGE_SIZE, 16
        self.lr = 1e-4
        self.iteration = 2
        self.ckpt_dir = "ckpt"
        self.image_channel = DS.CHANNELS
        self.num_outputs = DS.NUM_JOINTS
        self.dataset_name = 'two_point'


def get_config(is_train=True):
    opt = Options()
    return opt


def build_Data_From_JSONs():  # 有必要注释一下目录结构 和约定目录结构 TODO
    tf.enable_eager_execution()
    # 从LabelMe的JSONs文件来创建数据集
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'pose', dataset_name)
    source_dir = os.path.join(root, 'data', 'train')
    output_dir = os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size))
    if not os.path.exists(output_dir):   #
        os.makedirs(output_dir)
    DS.produce_dataset_from_jsons(dataset_name=dataset_name, json_source_dir=source_dir, target_directory=output_dir)
    pass


def build_Data_From_CSVs():   # 有必要注释一下目录结构 TODO
    tf.enable_eager_execution()
    opts = get_config(is_train=True)

    output_dir = os.path.join(datasets_root, 'normal_tf_records_{}'.format(opts.image_size))
    datasets_names = ['byqywb', 'byqdwb', 'dlqqyjcb', 'ljjsq1', 'ljjsq2']
    for dataset_name in datasets_names:
        root = os.path.join(datasets_root, dataset_name)
        csv_file_path = os.path.join(root, "train.csv")
        image_source_dir = os.path.join(root, "images")
        if not os.path.exists(output_dir):   #
            os.makedirs(output_dir)
        DS.produce_dataset_from_csv(dataset_name=dataset_name, csv_file_path=csv_file_path, image_source_dir=image_source_dir, target_directory=output_dir)
