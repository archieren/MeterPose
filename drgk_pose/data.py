from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import io
import threading
import cv2
import random
import sys
import numpy as np
import tensorflow as tf
import logging
import json
import csv
# from labelme import utils as LU
import base64


logger = logging.getLogger(__name__)

NUM_JOINTS = 2
CHANNELS = 3
IMAGE_SIZE = 32*7
HEAT_MAP_SIZE = 8*7
SIGMA = 2

# 这个要约定一下，TODO
POINT_INDEX = {'1': 1,
               '2': 2,
               'c': 1,  # 中心
               'b': 2,  # 黑针头,
               'r': 3   # 红针头
               }


# dataset_utils
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    # if not isinstance(value, list):
    # value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encode_image_to_string(image):
    encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 元组
    encoded = encoded[1]  # 元组的第二项
    encoded = np.array(encoded)
    encoded = encoded.tostring()  # an bytes object
    return encoded


def get_annotations_dict(annotation_json_path):
    """
    从LabelMe文件中读取相关数据.了解LabelMe的文件格式是必要的.
    """
    if not os.path.exists(annotation_json_path):
        return None
    # 读入json文件
    with open(annotation_json_path, 'r') as f:
        json_text = json.load(f)
    #
    shapes = json_text.get('shapes', None)
    if shapes is None:
        return None
    # 读入图像数据.json文件中存放的是相对路径,要调整成绝对路径
    #
    image_relative_path = json_text.get('imagePath', None)
    if image_relative_path is None:
        return None
    image_name = image_relative_path.split('/')[-1]  # 只需要文件名
    image_format = image_name.split('.')[-1].replace('jpg', 'jpeg')

    # labelme里的imageData是经过base64处理的
    encoded_jpg = base64.b64decode(json_text.get('imageData'))  # an bytes object
    image = cv2.imdecode(np.fromstring(encoded_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # tf.image.decode_jpeg(encoded_jpg)
    height_o = image.shape[0]
    width_o = image.shape[1]
    channels = image.shape[2]
    assert channels == CHANNELS

    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    # 拆分开来，以表示中间的数据类型的变化，没搞清中间的技术细节
    encoded_jpg_resized = cv2.imencode('.jpg', image_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 元组
    encoded_jpg_resized = encoded_jpg_resized[1]  # 元组的第二项
    encoded_jpg_resized = np.array(encoded_jpg_resized)
    encoded_jpg_resized = encoded_jpg_resized.tostring()  # an bytes object
    # 只是验证一下
    # print(cv2.imdecode(np.fromstring(encoded_jpg_resized,dtype=np.uint8),cv2.IMREAD_COLOR).shape)
    #
    point_count = 0
    points = np.zeros((NUM_JOINTS, 2), dtype=np.int)
    for mark in shapes:
        shape_type = mark.get('shape_type')
        if not (shape_type == 'point'):
            continue
        else:
            point_count += 1
        point_index = int(POINT_INDEX[mark.get('label').split('_')[-1]])  # label的值是point_i,取索引号
        assert point_index <= NUM_JOINTS
        point = np.array(mark.get('points'), dtype=np.int)[0][::-1]  # 调整成[y,x]，即Height x Width坐标框架了！
        print("{}".format(image_name))
        point = (point * IMAGE_SIZE // [height_o, width_o])
        points[point_index-1] = point
    assert point_count == NUM_JOINTS  # 必须要定义NUM_JOINTS个Point!

    annotation_dict = {'height': height_o,
                       'width': width_o,
                       'channels': channels,
                       'filename': image_name,
                       'encoded_original_jpg': encoded_jpg,
                       'encoded_jpg': encoded_jpg_resized,
                       'format': image_format,
                       'points': points}
    return annotation_dict


def generate_target(points):
    assert points.shape[0] == NUM_JOINTS
    # 目标热图只采用 Gaussian类型
    target = np.zeros((HEAT_MAP_SIZE, HEAT_MAP_SIZE, NUM_JOINTS), dtype=np.float32)
    feat_stride = IMAGE_SIZE / HEAT_MAP_SIZE
    temperature_size = SIGMA * 3
    TOP, BOTTOM = LEFT, RIGHT = Y_, X_ = 0, 1  # 纯粹为了可读
    for point_id in range(NUM_JOINTS):
        mu_y = int(points[point_id][Y_]/feat_stride+0.5)  # 调整到HeatMap的坐标系，四舍五入.
        mu_x = int(points[point_id][X_]/feat_stride+0.5)
        # 检查Gaussian_bounds是否落在HEATMAP之外,直接跳出运行,不支持不可见JOINT POINT
        left_top = [int(mu_y - temperature_size), int(mu_x - temperature_size)]
        right_bottom = [int(mu_y + temperature_size), int(mu_x + temperature_size)]
        if left_top[Y_] >= HEAT_MAP_SIZE or left_top[X_] >= HEAT_MAP_SIZE or right_bottom[Y_] < 0 or right_bottom[X_] < 0:
            assert False

        # 生成Gaussian_Area
        size = SIGMA * 6 + 1  # temperature*2+1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIGMA ** 2))
        # 确定可用Gaussian_Area
        g_y = max(0, -left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_SIZE) - left_top[Y_]
        g_x = max(0, -left_top[X_]), min(right_bottom[X_], HEAT_MAP_SIZE) - left_top[X_]
        #
        img_y = max(0, left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_SIZE)
        img_x = max(0, left_top[X_]), min(right_bottom[X_], HEAT_MAP_SIZE)

        target[img_y[TOP]:img_y[BOTTOM], img_x[LEFT]:img_x[RIGHT], point_id] = g[g_y[TOP]:g_y[BOTTOM], g_x[LEFT]:g_x[RIGHT]]
    return target


def create_tf_example(annotation_dict):
    points = annotation_dict['points']
    target = generate_target(points)
    target = target.flatten()

    features = {'image/height': _int64_feature(annotation_dict['height']), 'image/width': _int64_feature(annotation_dict['width']), 'image/channels': _int64_feature(annotation_dict['channels']), 'image/filename': _bytes_feature(annotation_dict['filename']), 'image/encoded_original_jpg': _bytes_feature(annotation_dict['encoded_original_jpg']), 'image/encoded_jpg': _bytes_feature(annotation_dict['encoded_jpg']), 'image/format':  _bytes_feature(annotation_dict['format']), 'image/points': float_list_feature(target)
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parser(record):
    features = {'image/height': tf.FixedLenFeature((), tf.int64),
                'image/width': tf.FixedLenFeature((), tf.int64),
                'image/channels': tf.FixedLenFeature((), tf.int64),
                'image/filename': tf.FixedLenFeature((), tf.string),
                'image/encoded_original_jpg': tf.FixedLenFeature((), tf.string),
                'image/encoded_jpg': tf.FixedLenFeature((), tf.string),
                'image/format':  tf.FixedLenFeature((), tf.string),
                'image/points': tf.VarLenFeature(tf.float32)
                }
    # Parse example
    example = tf.parse_single_example(record, features)
    height, width = example['image/height'], example['image/width']
    # Decode image
    img = tf.image.decode_jpeg(example['image/encoded_jpg'])
    img = tf.cast(img, tf.float32)
    img = tf.divide(img, 127.5)
    img = tf.subtract(img, 1)  # tf.ones_like(img)
    img = tf.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))  # 这是坑点,在Eager模式下不需要此句话,但在Graph模式下却是必需的！
    # KeyPoint Heat_Map
    joint_heatmap_map = example['image/points']
    joint_heatmap_map = tf.sparse_tensor_to_dense(joint_heatmap_map, default_value=0.0)
    joint_heatmap_map = tf.reshape(joint_heatmap_map, (HEAT_MAP_SIZE, HEAT_MAP_SIZE, NUM_JOINTS))

    return img, joint_heatmap_map, height, width


def find_json_files(data_dir):
    pattern = os.path.join(data_dir, '*.json')
    filenames = tf.gfile.Glob(pattern)
    return filenames


def process_json_files_one_thread(thread_index, ranges, name, filenames, num_shards, output_directory):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s_%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            ann = get_annotations_dict(filename)
            example = create_tf_example(ann)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
        writer.close()
        shard_counter = 0


def process_json_files(name, filenames, output_directory, num_threads=1, num_shards=1):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    # Launch a thread for each batch.
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    # Create a generic TensorFlow-based utility for converting all image codings.
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames, num_shards, output_directory)
        t = threading.Thread(target=process_json_files_one_thread, args=args)
        t.start()
        threads.append(t)
    # Wait for all the threads to terminate.
    coord.join(threads)


def __produce_dataset_from_jsons(name, json_source_dir, target_directory):  # 单线程暂时够了，就不用这个接口！
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    filenames = find_json_files(data_dir=json_source_dir)
    process_json_files(name=name, filenames=filenames, output_directory=target_directory)


def produce_dataset_from_jsons(dataset_name, json_source_dir, target_directory):
    filenames = find_json_files(data_dir=json_source_dir)
    output_filename = '%s.tfrecord' % (dataset_name)
    output_file = os.path.join(target_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    for filename in filenames:
        ann = get_annotations_dict(filename)
        example = create_tf_example(ann)
        writer.write(example.SerializeToString())
    writer.close()


def get_annotations_dict_from_jpg(file_path, points):
    pass


def produce_dataset_from_csv(dataset_name, csv_file_path, image_source_dir, target_directory):
    col_types = [str, int, int, int, int]   # filename,p0_x,p0_y,p1_x,p1_y
    with open(csv_file_path) as f:
        output_filename = '%s.tfrecord' % (dataset_name)
        output_file = os.path.join(target_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        csv_f = csv.reader(f)
        _ = next(csv_f)    # 跳过第一行
        for raw_row in csv_f:
            row = tuple(convert(value) for convert, value in zip(col_types, raw_row))
            image_file_name = row[0]

            image = cv2.imread(os.path.join(image_source_dir, image_file_name))
            height, width, channels = image.shape[0], image.shape[1], image.shape[2]

            image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

            image_buffer = encode_image_to_string(image)
            image_resized_buffer = encode_image_to_string(image_resized)

            points = np.zeros((NUM_JOINTS, 2), dtype=np.int)
            points[0, :] = np.array([row[2], row[1]]) * IMAGE_SIZE // [height, width]
            points[1, :] = np.array([row[4], row[3]]) * IMAGE_SIZE // [height, width]

            annotation_dict = {'height': height,
                               'width': width,
                               'channels': channels,
                               'filename': image_file_name,
                               'encoded_original_jpg': image_buffer,
                               'encoded_jpg': image_resized_buffer,
                               'format': 'jpeg',
                               'points': points}

            example = create_tf_example(annotation_dict)
            writer.write(example.SerializeToString())
        writer.close()
    pass
