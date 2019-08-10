# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from drgk_pose import data as DS
from drgk_pose import utils
from drgk_pose import model_in_keras_way as model_keras
import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

KM = tf.keras.models
KB = tf.keras.backend


# from tensorflow.python.platform import tf_logging
# tf_logging.set_verbosity('INFO')
TFRECORD_DATASET_NAMES_FOR_TRAIN = ['two_point', 'byqywb', 'byqdwb', 'dlqqyjcb', 'ljjsq1', 'ljjsq2']
TFRECORD_DATASET_NAMES_FOR_EVAL = ['two_point']
TFRECORD_DATASET_NAMES_FOR_PREDICT = ['two_point']

# 尤其注意这里,输出节点的名称确定，还是需要看Tensorboard里的内容.有点莫名其妙.
INPUT_NAME = 'input_1'
FROZEN_GRAPH_NAME = 'keras_model_frozen.pb'
OUTPUT_NAME = 'heatmap/Conv2D'


class Options(object):
    def __init__(self):
        self.image_size, self.batch_size = DS.IMAGE_SIZE, 16
        self.lr = 1e-4
        self.iteration = 20
        self.ckpt_dir = "ckpt"
        self.image_channel = DS.CHANNELS
        self.num_outputs = DS.NUM_JOINTS
        self.dataset_name = 'two_point'


def get_config(is_train=True):
    opt = Options()
    return opt


def get_root(opts):
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'pose', dataset_name)
    return root


def get_train_dataset(root, opts, dataset_names, parser=DS.parser):
    dataset_files = [os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size), '{}.tfrecord'.format(data_name))
                     for data_name in dataset_names
                     ]

    dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = dataset.repeat()  # Keras模式下用这个,这个dataset无限长，它按epochs*steps_per_epoch来迭代的！
    dataset = dataset.map(parser).shuffle(buffer_size=20000)
    dataset = dataset.batch(opts.batch_size)
    return dataset


def get_predict_image_paths(root):
    pass


def get_predict_dataset(root, image_paths):
    def load_and_preprocess_image(filepath):
        img = tf.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [DS.IMAGE_SIZE, DS.IMAGE_SIZE])
        img = tf.cast(img, tf.float32)
        img = tf.divide(img, 127.5)
        img = tf.subtract(img, 1)
        return img

    # image_paths = [os.path.join(root,'data','pred','Byqywb_1(1)_1.jpg')]
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image).batch(1)
    return image_ds


def build_prn_model(root, opts, is_compiling_needed=True):  # 从保存的weights重建，
    prn_model = model_keras.PRN_Model(image_shape=(opts.image_size, opts.image_size, opts.image_channel), num_outputs=opts.num_outputs)
    checkpoint_dir = os.path.join(root, 'checkpoints_{}_in_keras_way'.format(opts.image_size))
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    # 保存点
    prn_model.def_checkpoint(checkpoint_dir)  # 没必要封装,但...
    if is_compiling_needed:
        prn_model.compile()
    prn_model.restore_weights()
    return prn_model


def load_keras_model(root, opts):  # 从保存的model重建
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'pose', dataset_name)
    keras_model_path = os.path.join(root, 'checkpoints_{}_in_keras_way'.format(opts.image_size), 'model.h5')
    keras_model = KM.load_model(keras_model_path)
    # print(keras_model.outputs)
    # print(keras_model.inputs)
    # keras_model.summary()
    return keras_model


def init_keras_session():
    KB.clear_session()
    config = tf.ConfigProto()
    # 注意:Keras对内存的控制有问题!!!
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    KB.set_session(sess)  # set this TensorFlow session as the default session for Keras


def predict_with_keras_model():
    opts = get_config(is_train=True)
    root = get_root(opts)
    init_keras_session()
    prn_model = load_keras_model(root, opts)
    image_path = os.path.join(root, 'data', 'pred', 'Byqywb_1(1)_1.jpg')
    image_paths = [image_path]
    image_o = cv2.imread(image_path)
    image_o = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
    height, width = image_o.shape[0], image_o.shape[1]

    predict_dataset = get_predict_dataset(root, image_paths)
    pre = prn_model.predict(predict_dataset, steps=1)

    image = utils.draw_an_image(image_o, pre[0], height, width)
    plt.imshow(image)
    plt.show()


def train_PoseResNet_in_keras_way():
    init_keras_session()
    opts = get_config(is_train=True)
    root = get_root(opts)
    prn_model = build_prn_model(root, opts)

    def parser(record):  # parser需要改一下返回结果！
        b_x, b_target, _, _ = DS.parser(record)
        return b_x, b_target  # {INPUT_NAME:b_x}

    train_dataset = get_train_dataset(root, opts, TFRECORD_DATASET_NAMES_FOR_TRAIN, parser=parser)
    prn_model.fit(train_dataset, epochs=opts.iteration, steps_per_epoch=2000)
    prn_model.save_model()  #


def export_frozen_graph_with_a_keras_model():
    init_keras_session()
    # 此句非常重要,千万不要忘了！否则会有意想不到的问题！
    KB.set_learning_phase(0)
    opts = get_config(is_train=True)
    root = get_root(opts)
    _ = load_keras_model(root, opts)  # 这种方法并不好，得到的结果臃肿
    frozen_graph_path = os.path.join(root, 'GraphExported', FROZEN_GRAPH_NAME)  # 'frozen_keras_model.pb'

    sess = KB.get_session()
    with sess.as_default():
        output_graph_def = utils.freeze_session(sess, output_names=[OUTPUT_NAME])
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    pass


def export_frozen_graph_with_svaved_weights():
    init_keras_session()
    KB.set_learning_phase(0)  # !!!
    opts = get_config(is_train=True)
    root = get_root(opts)
    _ = build_prn_model(root, opts, is_compiling_needed=False)  # 这种方式得到冻结图,就很简约!
    frozen_graph_path = os.path.join(root, 'GraphExported', FROZEN_GRAPH_NAME)  # 'frozen_keras_model.pb'

    sess = KB.get_session()
    with sess.as_default():
        output_graph_def = utils.freeze_session(sess, output_names=[OUTPUT_NAME])
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    pass


def show_some_predictions():
    init_keras_session()
    KB.set_learning_phase(0)
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'pose', dataset_name)

    image_path = os.path.join(root, 'data', 'pred', 'Byqywb_1(1)_1.jpg')
    image_o = cv2.imread(image_path)
    image_o = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
    height, width = image_o.shape[0], image_o.shape[1]
    print("{}.{}".format(height, width))
    image = image_o/127.5 - 1.0
    image = cv2.resize(image, (DS.IMAGE_SIZE, DS.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    frozen_graph_dir = os.path.join(root, 'GraphExported')
    frozen_graph_path = os.path.join(frozen_graph_dir, FROZEN_GRAPH_NAME)
    graph, sess = utils.load_a_frozen_graph(frozen_graph_path)

    with graph.as_default():
        with sess.as_default():
            fetches = {}  # 要取那些tensor呢？
            key = OUTPUT_NAME
            tensor_name = key + ':0'
            fetches[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # input_image_tensor = tf.get_default_graph().get_tensor_by_name(IMAGE_INPUT_TENSOR_NAME+':0')
            input_image_tensor = tf.get_default_graph().get_tensor_by_name(INPUT_NAME+':0')
            feeds = {input_image_tensor: np.expand_dims(image, 0)}

            output_dict = sess.run(fetches=fetches, feed_dict=feeds)
            print("Output_Dict keys:{}".format(output_dict.keys()))
            new_image = utils.draw_an_image(image_o, output_dict[key][0], height, width)

            plt.subplot(121)
            plt.imshow(new_image)
            plt.subplot(122)
            plt.imshow(image_o)
            plt.show()

    pass


def main(_):
    # train_PoseResNet_in_keras_way()

    # export_frozen_graph_with_a_keras_model()
    # export_frozen_graph_with_svaved_weights()
    predict_with_keras_model()
    show_some_predictions()


if __name__ == '__main__':
    tf.app.run()
