
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_estimator as TFE
from drgk_pose import networks_posenet

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = NUM_POINTS = 2
NUM_GEN_LAYERS = 3


class PoseNetModel(networks_posenet.Model):
    def __init__(self,
                 resnet_size,  # =50
                 data_format,  # =None
                 num_points,  # =NUM_POINTS
                 resnet_version=networks_posenet.DEFAULT_VERSION,
                 dtype=networks_posenet.DEFAULT_DTYPE,
                 num_gen_layers=NUM_GEN_LAYERS):
        # 简化为PoseNet的参数设置。
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True
        super(PoseNetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_points,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype,
            num_gen_layers=num_gen_layers
        )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
        resnet_size: The number of convolutional layers needed in the model.

    Returns:
        A list of block sizes to use in building the model.

    Raises:
        KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys()))
        raise ValueError(err)


def posenet_model_fn(features, labels, mode, num_points=NUM_POINTS, data_format='channels_last', resnet_size=50):
    tf.compat.v1.summary.image('images', features, max_outputs=8)
    assert features.dtype == networks_posenet.DEFAULT_DTYPE

    model = PoseNetModel(resnet_size=resnet_size, data_format=data_format, num_points=num_points)
    heatmaps_pred = model(features, mode == TFE.estimator.ModeKeys.TRAIN)  # training = mode == tf.estimator.ModeKeys.Train

    predictions = {'heatmaps': heatmaps_pred, 'image_source': features}
    if mode == TFE.estimator.ModeKeys.PREDICT:
        return TFE.estimator.EstimatorSpec(
            mode=mode, predictions=predictions
            # ,export_outputs={'predict':TFE.estimator.export.PredictOutput(predictions)}
        )

    l2_loss = tf.losses.mean_squared_error(labels=labels, predictions=heatmaps_pred)

    accuracy = tf.compat.v1.metrics.accuracy(labels=labels, predictions=heatmaps_pred, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.identity(accuracy[1], name='train_accuracy')  # 创建一个叫train_accuracy的tensor
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    if mode == TFE.estimator.ModeKeys.EVAL:
        return TFE.estimator.EstimatorSpec(mode, loss=l2_loss, eval_metric_ops=metrics)

    assert mode == TFE.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9999)
    # 下面两句，必须琢磨一下，至少BN的正常运用，是需要他们的！
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(l2_loss, global_step=tf.train.get_global_step())
    train_op = tf.group(minimize_op, update_ops)
    return TFE.estimator.EstimatorSpec(mode=mode, loss=l2_loss, train_op=train_op)
