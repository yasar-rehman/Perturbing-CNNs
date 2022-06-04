from __future__ import division
from keras.engine import Layer, InputSpec
from keras import initializers
from keras import backend as K
import tensorflow as tf
from keras.layers import Input, Subtract
import numpy as np
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, concatenate, Lambda, GlobalMaxPool2D, Flatten, Embedding, division
import functools

from tensorflow.python.ops.image_ops_impl import ResizeMethod


class LRN2D(Layer):

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """

        :param alpha: Hyper parameter
        :param k: constant
        :param beta: constant
        :param n: adjacent kernel maps at same spatial position
        :param kwargs: Variable number of key-worded arguments
        """
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)

        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k

        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta

        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class gated_pool(Layer):

    # Implementation of Gated pooling,  for learning a parameter per layer option

    def __init__(self, pool_size, strides, padding ='same', data_format=None,**kwargs):

        super(gated_pool,self).__init__(**kwargs)
        self.pool_size = pool_size,
        self.strides = strides
        self.padding = padding
        self.data_format = data_format


    def build(self, input_shape):
        # input_shape is a 4D tensor with [batch size, height, width ,channels]
        self.mask = self.add_weight(name='kernel1',
                                    shape=(self.pool_size[0][0], self.pool_size[0][0],1,1),
                                    initializer='uniform',
                                    trainable=True) #only false for testing

        super(gated_pool, self).build(input_shape)

    def call(self, x, **kwargs):

        nb_batch, input_row, input_col, nb_filter = K.int_shape(x)   # get the output shape

        # output_size = input_row // 2  # output size should be reduced to half

        xs = []

        for c in tf.split(x, nb_filter, 3):
            conv1 = K.conv2d(c,
                             self.mask,
                             strides=(self.strides[0], self.strides[0]),
                             padding='same')


            xs.append(conv1)

        output = K.sigmoid(K.concatenate(xs, axis=3))


        pool_max = K.pool2d(x,
                            pool_size=(self.pool_size[0][0],self.pool_size[0][0]),
                            strides=(self.strides[0],self.strides[0]),
                            padding='same',
                            pool_mode='max')
        pool_avg = K.pool2d(x,
                            pool_size=(self.pool_size[0][0], self.pool_size[0][0]),
                            strides=(self.strides[0], self.strides[0]),
                            padding='same',
                            pool_mode='avg')

        f_gated = tf.add(tf.multiply(output, pool_max), tf.multiply((1-output), pool_avg))

        return f_gated

    def compute_output_shape(self, input_shape):
        rows = np.int(np.ceil(((input_shape[1] - self.pool_size[0][0]) / self.strides[0])) + 1)
        cols = np.int(np.ceil(((input_shape[2] - self.pool_size[0][0]) / self.strides[0])) + 1)
        return (input_shape[0], rows, cols, input_shape[3])


class hog_rgb_blend(Layer):
    def __init__(self, window_size, strides, single_lbp = True, padding ='same',**kwargs ):
        super(hog_rgb_blend,self).__init__()
        self.window_size = window_size
        self.strides = strides
        self.single_lbp = single_lbp
        self.padding = padding



    def build(self, input_shape):
        # input_shape is a 4D tensor with [batch size, height, width ,channels]
        self.mask = self.add_weight(name='kernel1',
                                    shape=(self.window_size[0], self.window_size[0], 2, 1),
                                    initializer='uniform',
                                    trainable=True)  # only false for testing

        super(hog_rgb_blend,self).build(input_shape)

    def call(self, x, **kwargs):
        rgb_data = x[0]
        hog_data = x[1]
        nb_batch, input_row, input_col, nb_filter = K.int_shape(rgb_data)  # get the output shape

        _, hog_row, hog_col, hog_filter = K.int_shape(hog_data)

        if self.single_lbp:
            hog_data = tf.split(hog_data, hog_filter, axis=3)
            hog_data = hog_data[1]


        if (input_row != hog_row) | (input_col != hog_col):
            hog_data = tf.image.resize_images(hog_data, (input_row, input_col), method=ResizeMethod.AREA)

        xs = []
        for c in tf.split(rgb_data, nb_filter,3):
            hyb_data = K.concatenate([c,hog_data],axis=-1)
            conv1 = K.conv2d(hyb_data,
                                 self.mask,
                                 strides=(self.strides[0], self.strides[1]),
                                 padding=self.padding)

            wt = K.sigmoid(conv1)
            # out = tf.add(tf.multiply(wt, c), tf.multiply((1-wt), hog_data))
            out = tf.multiply(wt,c)
            xs.append(out)

        output = K.concatenate(xs, axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]




class lbp_rgb_fusion(Layer):
    def __init__(self, window_size, strides, padding ='same',**kwargs ):
        super(lbp_rgb_fusion,self).__init__()
        self.window_size = window_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # input_shape is a 4D tensor with [batch size, height, width ,channels]
        self.mask = self.add_weight(name='kernel1',
                                    shape=(self.window_size[0], self.window_size[0], 2, 1),
                                    initializer='uniform',
                                    trainable=True)  # only false for testing

        super(lbp_rgb_fusion,self).build(input_shape)

    def call(self, x, **kwargs):
        rgb_data = x[0]
        lbp_data = x[1]
        nb_batch, input_row, input_col, nb_filter = K.int_shape(rgb_data)  # get the output shape
        _, lbp_row, lbp_col, lbp_filter = K.int_shape(lbp_data)


        # if (input_row != lbp_row) | (input_col != lbp_col):
        #     lbp_data = tf.image.resize_images(lbp_data, (input_row, input_col), method=ResizeMethod.AREA)

        xs = []

        for c in tf.split(rgb_data, nb_filter,3):
            hyb_data = K.concatenate([c,lbp_data],axis=-1)

            conv1 = K.conv2d(hyb_data,
                                 self.mask,
                                 strides=(self.strides[0], self.strides[1]),
                                 padding=self.padding)

            wt = K.sigmoid(conv1)
            out = tf.multiply(wt,c)
            xs.append(out)

        output = K.concatenate(xs, axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]








class global_gated_pool(Layer):

    # Implementation of Gated pooling,  for learning a parameter per layer option

    def __init__(self, **kwargs):

        super(global_gated_pool, self).__init__(**kwargs)


    def build(self, input_shape):
        # input_shape is a 4D tensor with [batch size, height, width ,channels]

        # however we just
        self.mask = self.add_weight(name='kernel_gp',
                                    shape=(1, input_shape[3]),
                                    initializer='uniform',
                                    trainable=True)  # only false for testing

        super(global_gated_pool, self).build(input_shape)

    def call(self, x, **kwargs):


        gb_pool = GlobalAveragePooling2D()(x)
        mx_pool = GlobalMaxPool2D()(x)
        # print(gb_pool)
        output = tf.add(tf.multiply(self.mask, mx_pool),tf.multiply((1-self.mask),gb_pool))

        return output


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


class disparity(Layer):
    def __init__(self, **kwargs):
        super(disparity, self).__init__(**kwargs)

    def build(self, input_shape):
        # input contain two elements (batch x feature vector) and labels

        super(disparity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # repeat the labels to make it equivalent to the feature vectors
        print(inputs)
        self.feat_vect_r = inputs[0]
        self.feat_vect_l = inputs[1]

        der_k = np.asarray([[1.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -2.0, -1.0]])
        der_k = np.expand_dims(np.expand_dims(der_k, axis=-1),-1)

        der_kern = np.repeat(der_k,8,axis=-2)


        kernel = tf.constant(der_kern, dtype=1)


        self.d_3d = self.feat_vect_r - self.feat_vect_l

        der = tf.nn.depthwise_conv2d(self.feat_vect_l, kernel, [1, 1, 1, 1], padding='SAME')


        self.depth_d_3d = self.d_3d / (K.abs(der)+0.5)
        # theta = tf.math.acos(K.clip(self.feat_vect, -1, 1))

        return  self.depth_d_3d



    def compute_output_shape(self, input_shape):
        return input_shape[0]




class contrastive_c_loss(Layer):
    def __init__(self, classes,**kwargs):
        self.classes = classes

        super(contrastive_c_loss,self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self. centers  =  self.add_weight(name='centers_1',
                                    shape=(self.classes, input_shape[0][1]),
                                    initializer='random_normal',
                                    trainable=True)
        super(contrastive_c_loss,self).build(input_shape)

    def call(self,x,**kwargs):

        return self.centers

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return self.centers.shape


def contrastive_center_loss(centers, features, labels):
    labels = K.reshape(labels,(-1,))
    labels = K.cast(labels, 'int32')

    # gather the centers corresponding to labels

    centroids_batch = tf.gather(centers, labels)
    #######################################################3


    # center_loss
    #l2_loss = 0.5*K.sum(K.square(features-centroids_batch), keepdims=True)
    # print(l2_loss)
    # compute the numerator
    print(centroids_batch)
    ####################################################################3
    a = K.sum(K.square(tf.subtract(features,centroids_batch)),axis=-1, keepdims=True)         # (batch size x 1)
    print(a)
    # compute the denomenator

   # expand the feature dimensions to match with centers dimensions
    features = K.expand_dims(features, axis=-1)        #(batch size x  2 x  1)

    print(features)

    # transform to have the features as the last dimension
    features = K.permute_dimensions(features, (0, 2, 1))  # batch size x 1 x 2
    # expand the center dimensions to match with features dimensions
    centers_ = K.expand_dims(centers,axis=0)              # 1 x 10 x 2
    dummy_var = tf.subtract(features, centers_)
    print(dummy_var)
    centers_all = K.sum(K.sum(K.square(dummy_var), axis=-1),axis=-1, keepdims=True)  # (batch size, 1)  ||x - cj||
    print(centers_all)

    b = centers_all - a
    print(b)

    contrast_loss = a/(b+1)
    print(contrast_loss)

    contrast_loss = 0.5*K.sum(contrast_loss, axis=0, keepdims=True)
    print(contrast_loss)
    #
    # l2_loss = distance_same / (inter_distances + 1)     # (batch_size, 1, 1)
    # print(l2_loss)
    #
    # l2_loss = 0.5 * K.sum(l2_loss,axis=0)  # compute the contrastive center loss
    # print(l2_loss)

    return contrast_loss
def center_loss(centers, features, labels):
    labels = K.reshape(labels,(-1,))
    labels = K.cast(labels, 'int32')

    # gather the centers corresponding to labels

    centroids_batch = tf.gather(centers, labels)
    #######################################################3
    # center_loss
    #l2_loss = 0.5*K.sum(K.square(features-centroids_batch), keepdims=True)
    # print(l2_loss)
    # compute the numerator
    ####################################################################3
    a = 0.5*tf.reduce_sum(K.square(tf.subtract(features,centroids_batch)),axis=-1, keepdims=True)         # (batch size x 1)
    # contrast_loss = 0.5*K.sum(a, axis=0, keepdims=True)
    return a


def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred): # with tensorflow
        _EPSILON = K.epsilon()
        y_pred = K.clip(y_pred, _EPSILON, 1. - _EPSILON) # improve the stability of the focal loss and see issues 1 for more information

        out = -(y_true * alpha * K.pow(1.0 - y_pred, gamma) * K.log(y_pred) + (1.0 - y_true) * (1.0 - alpha) * K.pow(y_pred, gamma) * K.log(1.0 - y_pred))
        # pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        # pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        # out =  -K.sum(alpha* K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha)* K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return K.mean(out, axis=-1)
    return focal_loss_fixed
