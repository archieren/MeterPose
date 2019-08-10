# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import numpy as np
import tensorflow as tf

# from drgk_anomaly.networks_keras import GanomalyGAN as ggan
# from drgk_anomaly.networks_keras import OC_NN as ocnn

# import tensorflow_estimator.python.estimator.keras as TFKE
from drgk_pose.networks_posenet_keras import PoseResNetBuilder as PRN

KL = tf.keras.layers
KB = tf.keras.backend
KC = tf.keras.callbacks
KM = tf.keras.models
KO = tf.keras.optimizers
KU = tf.keras.utils
TFKE = tf.keras.estimator
KLOSS = tf.keras.losses
KLoss = tf.keras.losses


def l2_loss(y_true, y_pred):
    return KB.mean(KB.square(y_pred - y_true))


class PRN_Model(object):
    """Train,Eval... the network.
    """

    def __init__(self, image_shape=(224, 224, 3), num_outputs=2, depth=64):
        self._lr = 0.0002
        self._beta1 = 0.5
        self._beta2 = 0.9999

        self.image_shape = image_shape
        self.num_outputs = num_outputs

        self.PRN = PRN.build_pose_resnet_50(self.image_shape, self.num_outputs)

        self.optimizer = KO.Adam(lr=self._lr, beta_1=self._beta1, beta_2=self._beta2)
        self.loss = 0.0

        # eager方式下，采用自定义的设置.(尽管还是可以用fit方式)
        self.optimizer_eager = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)

        self.checkpoint_dir = None
        self.checkpoint_prefix = None
        self.checkpoint = None
        self.checkpoint_manager = None

        self.model_weights_file = None  # 仅含权重,
        self.model_file = None  # 含权重,模型配置乃至优化器配置
        self.model_checkpoint = None

        pass

    def get_input_name(self):
        return self.PRN.input_names[0]

    def get_loss(self):
        return self.loss

    # Eagerly
    def train_eagerly(self, x, target):
        with tf.GradientTape() as tape:
            heatmap = self.PRN(x, training=True)
            loss = l2_loss(y_true=target, y_pred=heatmap)
            self.loss = loss
            grad = tape.gradient(loss, self.PRN.trainable_variables)
            self.optimizer_eager.apply_gradients(zip(grad, self.PRN.trainable_variables))
        pass

    def def_check_point_eagerly(self, checkpoint_dir):
        """
        在Eager执行模式下,最好采用tf.train.Checkpoint等来保存,而不用老式的tf.train.Saver来完成.
        但有个问题,在eager执行方式下,就没有MetaGraph，也不可能有MetaGraph存在！
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer_eager, net=self.PRN)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        pass

    def save_check_point_eagerly(self):
        self.checkpoint_manager.save()
        pass

    def restore_check_point_eagerly(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        pass

    def save_model_eagerly(self):
        model_file = os.path.join(self.checkpoint_dir, "eager_model.h5")
        self.PRN.save(model_file)

    '''
    #In_Keras_Mode
    def compile(self):
        self.PRN.compile(optimizer=self.optimizer,loss='mse',metrics=['accuracy'])

    def fit(self,train_dataset,epochs=1,steps_per_epoch=1000,validation_data=None):
        if self.model_checkpoint is None :
            callbacks=None
        else:
            callbacks=[self.model_checkpoint]

        if not (validation_data is None):
            validation_steps=1
        else :
            validation_steps=None

        self.PRN.fit(x=train_dataset
                    ,epochs=epochs
                    ,steps_per_epoch=steps_per_epoch
                    ,callbacks=callbacks  # 用来保存model
                    ,validation_data=validation_data
                    ,validation_steps=validation_steps
                    )

    def eval(self,eval_dataset):
        return self.PRN.evaluate(eval_dataset)

    def predict(self,pre_dataset):
        return self.PRN.predict(pre_dataset,steps=1)
    def def_ModelCheckPoint(self,checkpoint_dir):
        #name_of_model_file="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
        #name_of_model_file='model-{epoch:02d}.h5'
        #name_of_model_file='model-checkpoint.h5'
        #name_of_model_file='model-weights.h5'
        assert self.model_weights_file is not None
        self.model_checkpoint=KC.ModelCheckpoint(
            self.model_weights_file #os.path.join(checkpoint_dir, name_of_model_file)
            #,monitor='val_acc'
            #,save_best_only=True
            #,mode='auto'
            ,save_weights_only=True)


    def def_weights_and_model_file_in_keras_mode(self,checkpoint_dir):
        self.checkpoint_dir=checkpoint_dir
        self.model_weights_file=os.path.join(self.checkpoint_dir, "model_weights.h5")
        self.model_file=os.path.join(self.checkpoint_dir, "model.h5")
        pass

    def save_weights_in_keras_mode(self): #没必要用了
        self.PRN.save_weights(self.model_weights_file,save_format='h5')
        pass

    def restore_weights_in_keras_mode(self): #没必要用了
        if os.path.exists(self.model_weights_file):
            self.PRN.load_weights(self.model_weights_file)
        pass

    def save_model_in_keras_mode(self):
        self.PRN.save(self.model_file)
        pass

    '''

    def save_images(self, b_x, b_height, b_width, output_dir):
        b_heatmap = self.PRN(b_x, training=False)

        count = b_x.shape[0]
        for i in range(count):
            image = self.draw_an_image(b_x[i], b_heatmap[i], b_height[i], b_width[i])
            cv2.imwrite(os.path.join(output_dir, 'image_{}.jpg'.format(i)), image)

    def draw_an_image(self, x, heatmap, h, w):
        # image=np.zeros(shape=x.shape,dtype=np.float)
        image = np.array(x)
        image = np.array(255*(image+1)/2, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        assert self.num_outputs == heatmap.shape[2]
        for i in range(self.num_outputs):
            heatmap_i = np.array(heatmap[:, :, i])
            hs, ws = np.where(heatmap_i == heatmap_i.max())  # np.where返回的是元组，每一元素又是相应维度的坐标列表
            # 注意数组的坐标和图像坐标的对应关系
            # 图像的坐标系和array是一样的的,(height,width) 或 (row,colum) 或 (y,x)
            # 但opencv的函数，对坐标的引用，又是按常规情形使用的,
            p_x, p_y = ws[0], hs[0]  # 只取一个最大值点 ，
            p_x, p_y = int(p_x*self.image_shape[0]/heatmap_i.shape[0]), int(p_y*self.image_shape[1]/heatmap_i.shape[1])
            cv2.circle(image, (p_x, p_y), 5, (0, 0, 255), 1)

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

        return image
