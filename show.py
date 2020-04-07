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

prn_model = model_keras.PRN_Model(image_shape=(32*7, 32*7, 3), num_outputs=3)
prn_model.PRN.summary()