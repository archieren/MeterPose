# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import cv2
import numpy as np


def _fix_batch_norm_nodes(input_graph_def):
    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()  # tf.graph_util.remove_training_nodes(graph.as_graph_def())

        _fix_batch_norm_nodes(input_graph_def)   # 我估计这是古董级的Hack

        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            output_names,
            freeze_var_names
        )
        return frozen_graph


def load_a_frozen_graph(path_to_frozen_graph):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(path_to_frozen_graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    graph = tf.Graph()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=graph, config=config)

    with graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    '''
    graph = tf.Graph()
    with graph.as_default():  #这个是必须的.
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')  #The name var will prefix every op/nodes in your graph
    '''

    return graph, sess


def draw_an_image(x, heatmap, h, w):
    # 此image来源于原始的图像
    image = x
    # image = np.array(x)
    # image = np.array(255*(image+1)/2,dtype=np.uint8)
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    for i in range(heatmap.shape[2]):
        heatmap_i = np.array(heatmap[:, :, i])
        hs, ws = np.where(heatmap_i == heatmap_i.max())  # np.where返回的是元组，每一元素又是相应维度的坐标列表
        # print(hs.shape)
        # 注意数组的坐标和图像坐标的对应关系
        # 图像的坐标系和array是一样的的,(height,width) 或 (row,colum) 或 (y,x)
        # 但opencv的函数，对坐标的引用，又是按常规情形使用的,
        p_x, p_y = ws[0], hs[0]  # 只取一个最大值点 ，
        # print("{}-{}".format(p_y,p_x))
        p_x, p_y = int(p_x*w/heatmap_i.shape[0]), int(p_y*h/heatmap_i.shape[1])
        # print("....{}-{}".format(p_y,p_x))
        cv2.circle(image, (p_x, p_y), 5, (0, 0, 255), 2)
    return image
