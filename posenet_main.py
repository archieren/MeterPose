# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_estimator as TFE
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib as PL

from drgk_pose import model
from drgk_pose import utils
from drgk_pose import data as DS

from tensorflow.python.tools import freeze_graph as FG

# from tensorflow.python.platform import tf_logging
# tf_logging.set_verbosity('INFO')
TFRECORD_DATASET_NAMES_FOR_TRAIN = ['two_point', 'byqywb', 'byqdwb', 'dlqqyjcb', 'ljjsq1', 'ljjsq2']
TFRECORD_DATASET_NAMES_FOR_EVAL = ['two_point']
TFRECORD_DATASET_NAMES_FOR_PREDICT = ['two_point']


INPUT_NAME = 'input_images'
UNFROZEN_GRAPH_NAME = 'tf_model_unfrozen.pb'
FROZEN_GRAPH_NAME = 'tf_model_frozen.pb'
OUTPUT_NODE_NAMES = OUTPUT_NAME = 'resnet_model/heatmaps'


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


def get_dataset(root, dataset_names, parser=DS.parser, is_training=True):
    opts = get_config(is_train=True)
    dataset_files = [os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size), '{}.tfrecord'.format(data_name))
                     for data_name in dataset_names
                     ]

    dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = dataset.map(parser)
    if is_training:
        dataset = dataset.repeat(opts.iteration-1)  # 这样每次就训练一个epoch就行了！
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(opts.batch_size)
    return dataset


def posenet_model_fn(features, labels, mode, params):
    # features = tf.feature_column.input_layer(features, params['feature_columns'])
    return model.posenet_model_fn(features, labels, mode)


def get_predict_image_paths(root):
    pass


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


def posenet_predict(poser, root):
    root_path = PL.Path(root)/'data'/'pred'
    output_dir = os.path.join(root, 'saved_images_224')
    all_image_paths = list(root_path.glob('**/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]  # 必须的

    def input_fn_for_predict():
        image_ds = get_predict_dataset(all_image_paths)
        iterator = image_ds.make_one_shot_iterator()
        b_x = iterator.get_next()
        return b_x

    pre = poser.predict(input_fn=input_fn_for_predict)
    for (index, pi) in enumerate(pre):
        image = cv2.imread(all_image_paths[index])
        height, width = image.shape[0], image.shape[1]
        image = utils.draw_an_image(image, pi['heatmaps'], height, width)
        cv2.imwrite(os.path.join(output_dir, 'image_{}.jpg'.format(index)), image)
        """
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.title(all_image_paths[index])
        plt.imshow(image)
        plt.show()
        """
    print(pre)


def posenet_train(poser, root):
    def parser(record):  # parser需要改一下返回结果！
        b_x, b_target, _, _ = DS.parser(record)
        return b_x, b_target  # {INPUT_NAME:b_x}

    def input_fn_for_train():
        dataset = get_dataset(root, TFRECORD_DATASET_NAMES_FOR_TRAIN, parser=parser, is_training=True)
        iterator = dataset.make_one_shot_iterator()
        b_x, b_target = iterator.get_next()
        return b_x, b_target

    poser.train(input_fn=input_fn_for_train)


def posenet_run():
    # my_feature_columns=[]
    opts = get_config(is_train=True)
    root = get_root(opts)

    checkpoint_dir = os.path.join(root, 'checkpoints_{}'.format(opts.image_size))
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)

    poser = TFE.estimator.Estimator(
        model_fn=posenet_model_fn,
        model_dir=checkpoint_dir,
        params=None
    )
    posenet_train(poser, root)
    posenet_predict(poser, root)  # 居然不能单独运行


'''
def export_frozen_inference_graph():    #最好别用这个，输出的图比较大
    def freeze_graph(checkpoint_dir, frozen_graph_dir,output_node_names):

        #这是个简化的冻结图的算法.利用MetaGraph+CheckPoint来生成.
        #有些概念上的东西需要澄清：
        #Graph and Graph_Def
        #MetaGraph,MetaGraph_Def,CheckPoint and Saver
        #但这个算法得到的冻结图，好像没那么紧凑.
        #代码留在这,只是给自己留个样板

        input_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        output_graph = os.path.join(frozen_graph_dir,FROZEN_GRAPH_NAME)  #"frozen_model.pb")

        clear_devices = True
        with tf.Session() as sess:
            # Import the meta graph in the current default Graph and Restore the weights
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
            saver.restore(sess, input_checkpoint)

            output_graph_def = utils.freeze_session(sess,output_names=output_node_names.split(","))
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
    #KB.set_learning_phase(0)    !!!!????
    opts=get_config(is_train=True)
    dataset_name=opts.dataset_name
    root=os.path.join(os.getcwd(), 'work','pose',dataset_name)
    checkpoint_dir = os.path.join(root,'checkpoints_{}'.format(opts.image_size))
    frozen_graph_dir=os.path.join(root,'GraphExported')
    freeze_graph(checkpoint_dir,frozen_graph_dir,OUTPUT_NAME)
    pass
'''


def _export_frozen_inference_graph():
    opts = get_config(is_train=True)
    root = get_root(opts)

    checkpoint_dir = os.path.join(root, 'checkpoints_{}'.format(opts.image_size))
    graph_dir = os.path.join(root, 'GraphExported')
    OUTPUT_UNFROZEN_GRAPH_PATH = os.path.join(graph_dir, UNFROZEN_GRAPH_NAME)
    OUTPUT_FROZEN_GRAPH_PATH = os.path.join(graph_dir, FROZEN_GRAPH_NAME)

    def export_inference_graph():
        graph = tf.Graph()
        with graph.as_default():
            # 注意这里的思路：
            # 1.不是从checkpoint中的meta文件来恢复graph_def，而是用网络图的定义函数来得到Graph_Def，
            #       --怎么说呢，这是有好处的，比如可以定义输入，可以设置training，等等！！！
            # 2.然后，再从checkpoint中取出训练好的参数值，进行赋值指派等等！！！(看 _export_frozen_inference_graph())
            placeholder = tf.placeholder(name=INPUT_NAME, dtype=tf.float32, shape=[None, DS.IMAGE_SIZE, DS.IMAGE_SIZE, 3])
            posenetmodel = model.PoseNetModel(resnet_size=50, data_format='channels_last', num_points=model.NUM_POINTS)
            posenetmodel(placeholder, training=False)  # 这一步就生成Graph了
            graph_def = graph.as_graph_def()

            with tf.gfile.GFile(OUTPUT_UNFROZEN_GRAPH_PATH, 'wb') as f:
                f.write(graph_def.SerializeToString())

    export_inference_graph()

    tf.reset_default_graph()
    input_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    FG.freeze_graph(OUTPUT_UNFROZEN_GRAPH_PATH,     # input_graph
                    None,                           # input_saver          "TensorFlow saver file to load."
                    True,                           # input_binary         "Whether the input files are in binary format."
                    input_checkpoint,               # input_checkpoint
                    OUTPUT_NODE_NAMES,              # output_node_names    "The name of the output nodes, comma separated."
                    "save/restore_all",             # restore_op_name      "The name of the master restore operator."
                    "save/Const:0",                 # filename_tensor_name "The name of the tensor holding the save path."
                    OUTPUT_FROZEN_GRAPH_PATH,       # output_graph
                    True,                           # clear_devices        "Whether to remove device specifications."
                    "",                             # initializer_nodes    "comma separated list of initializer nodes to run before freezing."
                    # variable_names_whitelist="",
                    # variable_names_blacklist="",
                    # input_meta_graph=None,
                    # input_saved_model_dir=None,
                    # saved_model_tags=tag_constants.SERVING,
                    # checkpoint_version=saver_pb2.SaverDef.V2
                    )


def show_some_predictions():
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'pose', dataset_name)

    image_path = os.path.join(root, 'data', 'pred', 'byqywb_4_39_4.jpg')
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
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_name = {output.name for op in ops for output in op.inputs}
            for name in all_tensor_name:
                if 'Iter' in name:
                    print(name)

            fetches = {}  # 要取那些tensor呢？
            key = OUTPUT_NAME
            tensor_name = key + ':0'
            fetches[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

            # input_image_tensor = tf.get_default_graph().get_tensor_by_name(IMAGE_INPUT_TENSOR_NAME+':0')
            input_image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(INPUT_NAME+':0')
            feeds = {input_image_tensor: np.expand_dims(image, 0)}

            output_dict = sess.run(fetches=fetches, feed_dict=feeds)
            print("Output_Dict keys:{}".format(output_dict.keys()))
            new_image = utils.draw_an_image(image_o, output_dict[OUTPUT_NAME][0], height, width)

            # plt.ion()
            plt.subplot(121)
            plt.imshow(new_image)
            plt.subplot(122)
            plt.imshow(image_o)
            plt.show()


def just_play():
    opts = get_config(is_train=True)
    root = get_root(opts)
    print(root)
    print('---------')
    root = PL.Path(root)
    for item in root.iterdir():
        print(item)
    print("==========")
    all_image_paths = list(root.glob('**/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    print(len(all_image_paths))
    # for item in all_image_paths:
    #    print(item)


if __name__ == '__main__':
    # posenet_run()

    # export_frozen_inference_graph()

    # _export_frozen_inference_graph()
    show_some_predictions()
    # just_play()
