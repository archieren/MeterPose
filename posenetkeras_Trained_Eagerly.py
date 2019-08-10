# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from drgk_pose import data as DS
from tqdm import tqdm
from drgk_pose import utils
from drgk_pose import model_keras_eagerly as model_keras
import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

KM = tf.keras.models
KB = tf.keras.backend


# from tensorflow.python.platform import tf_logging
# tf_logging.set_verbosity('INFO')
TFRECORD_DATASET_NAMES_FOR_TRAIN = ['byqywb', 'byqdwb', 'dlqqyjcb', 'ljjsq1', 'ljjsq2']  # ,'two_point'
TFRECORD_DATASET_NAMES_FOR_EVAL = ['two_point']
TFRECORD_DATASET_NAMES_FOR_PREDICT = ['two_point']

# 尤其注意这里,输出节点的名称确定，还是需要看Tensorboard里的内容.有点莫名其妙.
INPUT_NAME = 'input_1'
UNFROZEN_GRAPH_NAME = 'eager_model_unfrozen.pb'
FROZEN_GRAPH_NAME = 'eager_model_frozen.pb'

OUTPUT_NODE_NAMES = OUTPUT_NAME = 'heatmap/Conv2D'


class Options(object):
    def __init__(self):
        self.image_size, self.batch_size = DS.IMAGE_SIZE, 16
        self.lr = 1e-4
        self.iteration = 1
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


def get_dataset(root, dataset_names, parser=DS.parser, is_repeated=False):
    opts = get_config(is_train=True)
    dataset_files = [os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size), '{}.tfrecord'.format(data_name))
                     for data_name in dataset_names
                     ]

    dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = dataset.map(parser).shuffle(buffer_size=20000)
    if is_repeated:
        dataset = dataset.repeat()  # Keras模式下用这个,这个dataset无限长，它按epochs*steps_per_epoch来迭代的！
    else:
        dataset = dataset.repeat(1)  # Eagerly,Estimator 模式下用这个.
    dataset = dataset.batch(opts.batch_size)
    return dataset


def init_keras_session():
    KB.clear_session()
    config = tf.ConfigProto()
    # 注意:Keras对内存的控制有问题!!!
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    KB.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_PoseResNet_eagerly():
    tf.enable_eager_execution()
    init_keras_session()
    opts = get_config(is_train=True)
    root = get_root(opts)
    prn_model = model_keras.PRN_Model(image_shape=(opts.image_size, opts.image_size, opts.image_channel), num_outputs=opts.num_outputs)
    #
    checkpoint_dir = os.path.join(root, 'checkpoints_{}_in_eager_mode'.format(opts.image_size))
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    prn_model.def_check_point_eagerly(checkpoint_dir)
    if tf.train.latest_checkpoint(checkpoint_dir) is not None:
        prn_model.restore_check_point_eagerly()

    #
    dataset = get_dataset(root, TFRECORD_DATASET_NAMES_FOR_TRAIN)  # 注意，没有指定新的Parser!!

    saved_image_dir = os.path.join(root, 'saved_images_{}'.format(opts.image_size))
    if not os.path.exists(saved_image_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_image_dir)

    for _ in range(opts.iteration):
        bar = tqdm(dataset)
        for (batch_num, batch_data) in enumerate(bar):
            b_x, b_target, b_height, b_width = batch_data
            prn_model.train_eagerly(b_x, b_target)
            loss = prn_model.get_loss()
            bar.set_description("Loss: {:<10f} ".format(loss))
            bar.refresh()
            if batch_num % 32 == 0:
                prn_model.save_images(b_x, b_height, b_width, saved_image_dir)
        prn_model.save_check_point_eagerly()
    prn_model.save_model_eagerly()
    pass


def export_frozen_graph_with_keras_models():
    """
    那么可以下结论,无论Keras Model以EagerExcutionMode还是GraphExcutionMode方式训练的，只要存为model.h5(此句需要解释),
    后面都可以统一的freeze-session方式来冻结图.
    """
    init_keras_session()
    KB.set_learning_phase(0)  # 此句非常重要,千万不要忘了！否则会有意想不到的问题！
    opts = get_config(is_train=True)
    root = get_root(opts)
    keras_model_path = os.path.join(root, 'checkpoints_{}_in_eager_mode'.format(opts.image_size), 'eager_model.h5')
    keras_model = KM.load_model(keras_model_path)
    frozen_graph_path = os.path.join(root, 'GraphExported', FROZEN_GRAPH_NAME)  # 'frozen_eager_model.pb')
    print(keras_model.outputs)
    print(keras_model.inputs)
    keras_model.summary()
    sess = KB.get_session()
    with sess.as_default():
        output_graph_def = utils.freeze_session(sess, output_names=[OUTPUT_NAME])
        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    pass


def show_some_predictions():
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

    init_keras_session()
    frozen_graph_dir = os.path.join(root, 'GraphExported')
    frozen_graph_path = os.path.join(frozen_graph_dir, FROZEN_GRAPH_NAME)
    graph, sess = utils.load_a_frozen_graph(frozen_graph_path)

    with graph.as_default():
        with sess.as_default():
            fetches = {}  # 要取那些tensor呢？
            key = OUTPUT_NAME
            tensor_name = key + ':0'
            fetches[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            input_image_tensor = tf.get_default_graph().get_tensor_by_name(INPUT_NAME+':0')
            feeds = {input_image_tensor: np.expand_dims(image, 0)}

            output_dict = sess.run(fetches=fetches, feed_dict=feeds)
            print("Output_Dict keys:{}".format(output_dict.keys()))
            new_image = utils.draw_an_image(image_o, output_dict[OUTPUT_NAME][0], height, width)

#            plt.ion()
            plt.subplot(121)
            plt.imshow(new_image)
            plt.subplot(122)
            plt.imshow(image_o)
            plt.show()

    pass


def Summarize_PoseResNet():
    opts = get_config(is_train=True)
    prn_model = model_keras.PRN_Model(image_shape=(opts.image_size, opts.image_size, opts.image_channel), num_outputs=opts.num_outputs)
    prn_model.PRN.summary(line_length=192)
    print("Input_Names: {}".format(prn_model.PRN.input_names))
    print("Output_Names: {}".format(prn_model.PRN.output_names))


def main(_):

    # Summarize_PoseResNet()
    # train_PoseResNet_eagerly()
    export_frozen_graph_with_keras_models()
    show_some_predictions()


if __name__ == '__main__':
    tf.app.run()
