from __future__ import division


import tensorflow as tf
import numpy as np
import drgk_pose.networks_resnet_keras as RES
KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers


class PoseResNetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom Pose_Res_Net like architecture.
            See paper "Simple Baselines for Human Pose Estimation and Tracking"

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras PoseResNet `Model`.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        input = KL.Input(shape=input_shape)
        conv1 = RES._conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = KL.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = RES._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        block = PoseResNetBuilder.make_deconv_layers()(block)
        heatmap = KL.Conv2D(filters=num_outputs, kernel_size=(1, 1), padding="same", use_bias=False, strides=1, name="heatmap")(block)

        model = KM.Model(inputs=input, outputs=heatmap)
        return model

    @staticmethod
    def make_deconv_layers(num_layers=3):
        def f(input):
            for _ in range(num_layers):
                input = KL.Convolution2DTranspose(filters=256, kernel_size=(4, 4), padding="same", use_bias=False, strides=2)(input)
                input = KL.BatchNormalization()(input)
                input = KL.ReLU()(input)
            return input
        return f

    @staticmethod
    def build_pose_resnet_50(input_shape, num_outputs):
        return PoseResNetBuilder.build(input_shape, num_outputs, RES.bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_pose_resnet_101(input_shape, num_outputs):
        return PoseResNetBuilder.build(input_shape, num_outputs, RES.bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_pose_resnet_152(input_shape, num_outputs):
        return PoseResNetBuilder.build(input_shape, num_outputs, RES.bottleneck, [3, 8, 36, 3])
