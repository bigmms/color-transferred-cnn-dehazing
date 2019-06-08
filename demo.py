# -*- coding: utf-8 -*-
from tflearn.data_utils import *
from os.path import join
import numpy as np
from skimage import io, transform
from keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
import time
from functools import wraps
import warnings
from tensorflow.python.ops.image_ops import rgb_to_hsv
import tensorflow as tf
from keras import backend as K
warnings.filterwarnings("ignore")


def time_this_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


class DeHaze:
    def __init__(self):
        self.inputImagsPath, self.outputImagsPath = './Test', './TestOutput'
        self.Models = {'model_L': './Model/dehaze_rough_l.h5', 'model_A': './Model/dehaze_rough_a.h5',
                       'model_B': './Model/dehaze_rough_b.h5', 'model_refine': './Model/dehaze_refine_mse_30.h5'}
        self.first_StepResize = (228, 304, 3)
        self.extensionCoef_x, self.extensionCoef_y = 6, 8

    def SSIM_Loss(self, y_true, y_pred):
        y_pred_hsv = rgb_to_hsv(y_pred)

        # mae_loss
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        # tv_loss
        shape = tf.shape(y_pred)
        height, width = shape[1], shape[2]
        y = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(y_pred, [0, 1, 0, 0],
                                                                                          [-1, -1, -1, -1])
        x = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(y_pred, [0, 0, 1, 0],
                                                                                         [-1, -1, -1, -1])
        tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

        # ssim_loss
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
        patches_true = tf.extract_image_patches(y_true, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        patches_pred = tf.extract_image_patches(y_pred, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        covar_true_pred = std_pred * std_true
        ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
        denom = (K.square(u_true) + K.square(u_pred) + c1) * (var_pred + var_true + c2)
        ssim /= denom

        size = tf.size(y_pred_hsv)
        light_loss = tf.nn.l2_loss(y_pred_hsv[:, :, :, 2]) / tf.to_float(size)

        total_loss = -0.07 * light_loss + 1.0 * mae - 0.0005 * tv_loss
        return total_loss

    ''' Loading dataset　'''
    @time_this_function
    def loadImages(self):
        self.firstStepInputImages = []
        self.inputImagesList = os.listdir(self.inputImagsPath)
        for pk, pil in enumerate(self.inputImagesList):
            prou_img = transform.resize(io.imread(join(self.inputImagsPath, pil)), self.first_StepResize)
            self.firstStepInputImages.append(rgb2lab(np.uint8(prou_img * 255.0)))
        print("Loading...Done")

    ''' Haze Removal Part '''
    @time_this_function
    def first_step(self):
        print('#testing images: %s' % (len(self.firstStepInputImages)))
        self.firstStepInputImages = np.reshape(self.firstStepInputImages, [-1]+list(self.first_StepResize))

        l_pres = load_model(self.Models['model_L']).predict(self.firstStepInputImages)
        a_pres = load_model(self.Models['model_A']).predict(self.firstStepInputImages)
        b_pres = load_model(self.Models['model_B']).predict(self.firstStepInputImages)

        predicts = [[l[0], l[1], a[0], a[1], b[0], b[1]] for l, a, b in zip(l_pres, a_pres, b_pres)]
        self.firstOutputImages = [self.restoreCImg(iv, predicts[ik]) for ik, iv in enumerate(self.firstStepInputImages)]

    ''' Texture Refinement Part '''
    @time_this_function
    def second_step(self):
        self.secondInputImages = []
        for pk, pil in enumerate(self.firstOutputImages):
            imgc = np.pad(np.reshape(pil, newshape=[-1]+list(pil.shape)), [[0, 0], [self.extensionCoef_x, self.extensionCoef_x], [self.extensionCoef_y, self.extensionCoef_y], [0, 0]], mode='reflect')
            self.secondInputImages.append(np.reshape(imgc, newshape=list(imgc.shape[1:4])) / 255.0)

        model = load_model(self.Models['model_refine'], custom_objects={'SSIM_Loss': self.SSIM_Loss})
        prou_imgs = np.reshape(self.secondInputImages, [-1]+(list(self.secondInputImages[0].shape)))
        self.secondOutputImages = np.clip(model.predict(prou_imgs), 0, 1)
        [io.imsave(join(self.outputImagsPath, self.inputImagesList[ik]), iv[self.extensionCoef_x:iv.shape[0] - self.extensionCoef_x, self.extensionCoef_y:iv.shape[1] - self.extensionCoef_y, :]) for ik, iv in enumerate(self.secondOutputImages)]

    ''' Color Transfer '''
    def restoreCImg(self, haze_img_lab=None, avg_stds=None):
        pre_img = np.zeros(haze_img_lab.shape)
        avg_clean, std_clean = np.zeros([3]), np.zeros([3])
        avg_haze, std_haze = np.zeros([3]), np.zeros([3])
        for channel in range(3):
            avg_clean[channel], std_clean[channel] = avg_stds[channel * 2], avg_stds[channel * 2 + 1]
            avg_haze[channel], std_haze[channel] = np.mean(haze_img_lab[:, :, channel]), np.std(haze_img_lab[:, :, channel])
            pre_img[:, :, channel] = (haze_img_lab[:, :, channel] - avg_haze[channel]) * (std_clean[channel] / std_haze[channel]) + avg_clean[channel]
        return np.clip(np.uint8(lab2rgb(pre_img) * 255.0), np.uint8(0), np.uint8(255))

    def __del__(self):
        print("Done")

    def run(self):
        self.loadImages()
        self.first_step()
        self.second_step()


if __name__ == '__main__':
    deHaze = DeHaze()
    deHaze.run()
