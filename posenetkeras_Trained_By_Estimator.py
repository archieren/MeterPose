# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from drgk_pose import data as DS
from drgk_pose import utils
from drgk_pose import model_keras_by_estimator as model_keras
import os
import tensorflow as tf
import tensorflow_estimator as tfe
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.tools import freeze_graph as FG

KM = tf.keras.models
KB = tf.keras.backend


# 尤其注意这里,输出节点的名称确定，还是需要看Tensorboard里的内容.有点莫名其妙.
INPUT_NAME = 'input_images'
FROZEN_GRAPH_NAME = "estimator_model_frozen.pb"

UNFROZEN_GRAPH_NAME = 'estimator_model_unfrozen.pb'
OUTPUT_NODE_NAMES = OUTPUT_NAME = 'heatmap/Conv2D'


# from tensorflow.python.platform import tf_logging
# tf_logging.set_verbosity('INFO')
TFRECORD_DATASET_NAMES_FOR_TRAIN = ['byqywb', 'byqdwb', 'dlqqyjcb', 'ljjsq1', 'ljjsq2']  # ,'two_point'
TFRECORD_DATASET_NAMES_FOR_EVAL = ['two_point']
TFRECORD_DATASET_NAMES_FOR_PREDICT = ['two_point']


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


def get_train_dataset(root, dataset_names, parser=DS.parser):
    opts = get_config(is_train=True)
    dataset_files = [os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size), '{}.tfrecord'.format(data_name))
                     for data_name in dataset_names
                     ]

    dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = dataset.map(parser).shuffle(buffer_size=20000)
    dataset = dataset.batch(opts.batch_size)
    return dataset


def get_predict_dataset(image_paths):
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
    image_ds = path_ds.map(load_and_preprocess_image).batch(32)
    return image_ds


def init_keras_session():
    KB.clear_session()
    config = tf.ConfigProto()
    # 注意:Keras对内存的控制有问题!!!
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    KB.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_PoseResNet_with_estimator():
    # 这种方式可能最快
    init_keras_session()
    opts = get_config(is_train=True)
    root = get_root(opts)
    checkpoint_dir = os.path.join(root, 'checkpoints_{}_in_estimator_mode'.format(opts.image_size))
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)

    prn_model = model_keras.PRN_Model(image_shape=(opts.image_size, opts.image_size, opts.image_channel), num_outputs=opts.num_outputs)
    prn_model_estimator = prn_model.model_to_estimator(checkpoint_dir)

    def parser(record):  # parser需要改一下返回结果！
        b_x, b_target, _, _ = DS.parser(record)
        return b_x, b_target

    def input_fn(dataset_names):
        dataset = get_train_dataset(root, dataset_names, parser=parser)
        iterator = dataset.make_one_shot_iterator()
        b_x, b_target = iterator.get_next()
        return b_x, b_target

    def input_fn_for_predict():
        images_dir = os.path.join(root, 'data', 'pred')
        pattern = os.path.join(images_dir, '*.jpg')
        filenames = tf.gfile.Glob(pattern)
        all_image_paths = list(filenames)
        image_ds = get_predict_dataset(all_image_paths)
        iterator = image_ds.make_one_shot_iterator()
        b_x = iterator.get_next()
        return b_x  # 预测不需要labels部分

    def serving_input_fn():
        placeholder = tf.placeholder(name=INPUT_NAME, dtype=tf.float32, shape=[None, DS.IMAGE_SIZE, DS.IMAGE_SIZE, 3])
        # serving_features must match features in model_fn when mode == tf.estimator.ModeKeys.PREDICT.  # 没搞懂
        serving_features = {prn_model.get_input_name(): placeholder}
        return tfe.estimator.export.build_raw_serving_input_receiver_fn(serving_features)

    for _ in range(opts.iteration):  # opts.iteration
        prn_model_estimator.train(input_fn=lambda: input_fn(TFRECORD_DATASET_NAMES_FOR_TRAIN))
        eval_result = prn_model_estimator.evaluate(input_fn=lambda: input_fn(TFRECORD_DATASET_NAMES_FOR_EVAL))
        print('\nEvalResult: categorical_accuracy {categorical_accuracy:0.3f} loss {loss:0.10f} \n'.format(**eval_result))
        predict_result = prn_model_estimator.predict(input_fn=input_fn_for_predict)
        print(predict_result)
    prn_model_estimator.export_savedmodel(checkpoint_dir, serving_input_fn())

    """
    pre = prn_model_estimator.predict(input_fn=input_fn_for_predict)
    for (_,pi) in enumerate(pre):
        print(pi['heatmap'].shape)
    """

    pass


def _export_frozen_inference_graph():
    opts = get_config(is_train=True)
    root = get_root(opts)
    init_keras_session()
    # checkpoint_dir = os.path.join(root,'checkpoints_{}_in_estimator_mode'.format(opts.image_size))
    graph_dir = os.path.join(root, 'GraphExported')
    # OUTPUT_UNFROZEN_GRAPH_PATH  =   os.path.join(graph_dir,UNFROZEN_GRAPH_NAME)
    OUTPUT_FROZEN_GRAPH_PATH = os.path.join(graph_dir, FROZEN_GRAPH_NAME)
    input_saved_model_dir = os.path.join(root, 'checkpoints_{}_in_estimator_mode'.format(opts.image_size), '1561627272')

    # 解释一下 S1 和 S2,他们是互斥两种方式！！
    FG.freeze_graph(
        # -----------------------------S1
        input_graph=None,  # S1
        input_saver=None,
        input_binary=True,
        input_checkpoint=None,
        input_meta_graph=None,
        # ------------------------------ Common
        output_node_names=OUTPUT_NODE_NAMES,
        restore_op_name=None,  # Unused
        filename_tensor_name=None,  # Unused
        output_graph=OUTPUT_FROZEN_GRAPH_PATH,
        clear_devices=True,
        initializer_nodes="",
        variable_names_whitelist="",
        variable_names_blacklist="",
        # --------------------------------S2
        input_saved_model_dir=input_saved_model_dir,  # S2
        saved_model_tags=tf.saved_model.tag_constants.SERVING
    )


'''
def load_a_saved_model(export_dir):
    with tf.Session() as sess:
        # Load saved_model MetaGraphDef from export_dir.
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

        # Get SignatureDef for serving (here PREDICT_METHOD_NAME is used as export_outputs key in model_fn).
        #sigs = meta_graph_def.signature_def[tf.saved_model.signature_constants.PREDICT_METHOD_NAME]
        #print('hello{}'.format(sigs.__name__))
        # Get the graph for retrieving input/output tensors.
        g = tf.get_default_graph()

        # Retrieve serving input tensors, keys must match keys defined in serving_features (when building input receiver fn).
        input_images = g.get_tensor_by_name('input_images:0')

        # Retrieve serving output tensors, keys must match keys defined in ExportOutput (e.g. PredictOutput) in export_outputs.
        heatmap = g.get_tensor_by_name('heatmap/Conv2D:0')
        print(input_images.shape)
        print(heatmap.shape)
        return g,sess
'''


def show_some_predictions_with_frozen_graph():
    init_keras_session()
    opts = get_config(is_train=True)
    root = get_root(opts)

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
            ops = tf.get_default_graph().get_operations()
            all_tensor_name = {output.name for op in ops for output in op.inputs}
            for name in all_tensor_name:
                if 'Iter' in name:
                    print(name)

            fetches = {}  # 要取那些tensor呢？
            key = OUTPUT_NAME
            tensor_name = key + ':0'
            fetches[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # input_image_tensor = tf.get_default_graph().get_tensor_by_name(IMAGE_INPUT_TENSOR_NAME+':0')
            input_image_tensor = tf.get_default_graph().get_tensor_by_name(INPUT_NAME+':0')
            feeds = {input_image_tensor: np.expand_dims(image, 0)}

            output_dict = sess.run(fetches=fetches, feed_dict=feeds)
            print("Output_Dict keys:{}".format(output_dict.keys()))
            new_image = utils.draw_an_image(image_o, output_dict[OUTPUT_NAME][0], height, width)

            plt.subplot(121)
            plt.imshow(new_image)
            plt.subplot(122)
            plt.imshow(image_o)
            plt.show()


def show_some_predictions_with_saved_model():
    init_keras_session()
    opts = get_config(is_train=True)
    root = get_root(opts)

    image_path = os.path.join(root, 'data', 'pred', 'Byqywb_1(1)_1.jpg')
    image_o = cv2.imread(image_path)
    image_o = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
    height, width = image_o.shape[0], image_o.shape[1]
    print("{}.{}".format(height, width))
    image = image_o/127.5 - 1.0
    image = cv2.resize(image, (DS.IMAGE_SIZE, DS.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    export_dir = os.path.join(root, 'checkpoints_{}_in_estimator_mode'.format(opts.image_size), '1561627272')
    with tf.Session() as sess:
        _ = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        graph = sess.graph  # tf.get_default_graph()

        fetches = {}  # 要取那些tensor呢？
        key = OUTPUT_NAME
        tensor_name = key+':0'
        fetches[key] = graph.get_tensor_by_name(tensor_name)

        input_image_tensor = graph.get_tensor_by_name(INPUT_NAME + ':0')
        feeds = {input_image_tensor: np.expand_dims(image, 0)}

        output_dict = sess.run(fetches=fetches, feed_dict=feeds)
        new_image = utils.draw_an_image(image_o, output_dict[OUTPUT_NAME][0], height, width)

        plt.subplot(121)
        plt.imshow(new_image)
        plt.subplot(122)
        plt.imshow(image_o)
        plt.show()


def Summarize_PoseResNet():
    KB.clear_session()
    opts = get_config(is_train=True)
    prn_model = model_keras.PRN_Model(image_shape=(opts.image_size, opts.image_size, opts.image_channel), num_outputs=opts.num_outputs)
    prn_model.PRN.summary(line_length=192)
    print("Input_Names: {}".format(prn_model.PRN.input_names))
    print("Output_Names: {}".format(prn_model.PRN.output_names))


def main(_):

    # Summarize_PoseResNet()
    # train_PoseResNet_with_estimator()
    _export_frozen_inference_graph()
    show_some_predictions_with_frozen_graph()
    show_some_predictions_with_saved_model()

    # test_load_a_saved_model()


if __name__ == '__main__':
    tf.app.run()
