from __future__ import print_function
# from __future__ import absolute_import
from keras.optimizers import SGD, Adadelta
from keras.regularizers import l2
from keras.models import Sequential, Model, model_from_yaml
from keras.utils import plot_model
from keras.layers import merge, Dense, Dropout, Flatten, concatenate, add, Concatenate, subtract, average, dot
import numpy as np
import scipy
import sys
import os
import argparse
from random import randint, uniform
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE

import cv2
import keras.backend as K
from keras_preprocessing.image import load_img
from skimage.feature import local_binary_pattern as lbp
from skimage.feature import hog
from ess_func import comput_ch_LBP, comput_gray_LBP
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
RS = 123


print(sys.path)

from ess_func import read_pairs, sample_people, prewhiten, store_loss, hog_to_tensor, custom_loss
import seaborn as sns

# set the image parameters
img_rows = 120
img_cols = 120
img_dim_color = 3
# mix_prop = 1.0                                                    # set the value of the mixing proportion

# ----------------------------------------------------------------------------------------------------------------
# def fashion_scatter(x, colors):
#     # choose a color palette with seaborn.
#     num_classes = len(np.unique(colors))
#     palette = np.array(sns.color_palette("hls", num_classes))
#
#     # create a scatter plot.
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')
#
#     # add the labels for each digit corresponding to the label
#     txts = []
#
#     for i in range(num_classes):
#
#         # Position of each label at median of data points.
#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize=24)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
#         txts.append(txt)
#     plt.savefig('./no disparity.png')
#     plt.show()
#
#     return f, ax, sc, txts









#
# from  cnn_networks_mt.Auto_encod import Noise_estimator
#
# from cnn_networks_mt.CNN_small_GAP_NDisp import cnn_hybrid_color_single
# Res_model = Noise_estimator(img_rows, img_cols, img_dim_color)
#
#
# Res_model.load_weights('/home/yaurehman2/Documents/Face_liveness_Noise_modeling/training_files_mult/auto_encod_ckpt/VGG16_A_GAP_OULU_solocam_autoencod_20.h5')
#
# output = cnn_hybrid_color_single(Res_model.input, Res_model.output)  # load the model
#
# model_final = Model(inputs=Res_model.input,
#                     outputs=output)
# model_final.summary()
# model_final.load_weights('/home/yaurehman2/Documents/Face_liveness_Noise_modeling/training_files_mult/multi_scale/oulu/VGG16_A_solo_GAP_binary_mult_noise_disp_adaptive_relu_3x3_20.h5')
# layer_name = 'lambda_2'






from  CNN_networks.CNN_small_GAP_hog_lbp_rgb import cnn_hybrid_color_single


Res_model = cnn_hybrid_color_single(img_rows, img_cols, img_dim_color)

# for layer in Res_model.layers:
#     layer.trainable = False
# Res_model.trainable = False

# print(Res_model.summary())



model_final = Res_model

# print(model_final.summary())  # print the model summary


model_final.load_weights('/liveness_perturbation/livenet_hybrid_ckpt/oulu/protocol_1_eq_samp/color_lbp_rgb_online_p1_8_6_5x5_120x120_30.h5')

layer_name = 'lbp_rgb_fusion_1'


new_model = Model(model_final.input, model_final.get_layer(layer_name).output)

file_path = '/Newwork/OULU_FACE/train/81.jpg'

feature_vector = []
labels = []
counter = 0

img = load_img(file_path,
               grayscale=False,
               target_size=(img_rows,img_cols),
               interpolation='nearest'
               )
img =np.asarray(img).astype('float32')/255

img = np.expand_dims(img, axis=0)
lbp_im_data = comput_ch_LBP(img, p=6, r=8)

v = new_model.predict([img, lbp_im_data])

output = np.squeeze(v, axis=0)

plt.rcParams['figure.figsize'] = (15, 15)
f, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(output[:,:,i], cmap='jet')

        # ax[i, j].set_xlabel('feature map: %s' %counter)
        ax[i, j].axis('off')
plt.savefig('./with perturbation')

# f = plt.figure(1)
# ax = f.add_subplot(111)
# plt.imshow(output)
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# ax.set_frame_on(False)
plt.show()
#
#
# # f.savefig("./Figures/display2_generated.tiff",bbox_inches='tight', pad_inches=0)
#
# plt.show()
# cv2.imshow('output_image', np.uint8(output*255))
# cv2.waitKey()

