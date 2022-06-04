from __future__ import print_function
# from __future__ import absolute_import
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential, Model, model_from_yaml
from keras.utils import plot_model
import numpy as np
import scipy
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt
from keras_preprocessing import image
import tensorflow as t
import random
import cv2
import keras.backend as K
import tensorflow as tf
import pandas as pd
from skimage.feature import local_binary_pattern as lbp
from skimage.feature import hog
from sklearn.utils import class_weight
# for reproducibility purpose
import random as rn
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

# -----------------------------------------------------------------------------------------------
# import the essential functions required for computation
# sys.path.insert(0, os.path.expanduser('~//CNN_networks'))
# sys.export PYTHONPATH=/home/yaurehman2/PycharmProjects/face_anti_sp_newidea

print(sys.path)
from  CNN_networks.CNN_small_GAP_hog_lbp_rgb import cnn_hybrid_color_single


from ess_func import read_pairs, sample_people, prewhiten, store_loss, hog_to_tensor, custom_loss



def get_data_label(training_data, training_labels):
    # get the length of the training data
    len_tr = len(training_data)

    # get the number equal to the length of the training data
    indices_tr = np.arange(len_tr)

    # initialize the image counter
    images_read = 0
    train_img_data = []

    for i in indices_tr:
        # print(training_data[i], '\t' ,training_labels[i])
        if training_labels[i] > 0:
            training_labels[i] = 1

        train_img_data.append([training_data[i], training_labels[i]])

        images_read += 1
        sys.stdout.write('train images read = {0}\r'.format(images_read))
        sys.stdout.flush()

    return train_img_data


def data_balance(data, labels, seed):
    live_samples = []
    live_labels = []

    printed1_samples = []
    printed1_labels = []

    printed2_samples = []
    printed2_labels = []

    display1_samples = []
    display1_labels = []
    display2_samples = []
    display2_labels = []

    attack_samples = []
    attack_labels = []

    # separate the number of live images
    for i in range(len(data)):
        if labels[i] == 0:
            live_samples.append(data[i])
            live_labels.append(labels[i])

    # count the number of live images
    nrof_live = len(live_samples)

    # find a balancing ratio
    balance_ratio = nrof_live // 4

    for i in range(len(data)):
        if labels[i] == 1:
            printed1_samples.append(data[i])
            printed1_labels.append(labels[i])

        elif labels[i] == 2:
            printed2_samples.append(data[i])
            printed2_labels.append(labels[i])

        elif labels[i] == 3:
            display1_samples.append(data[i])
            display1_labels.append(labels[i])

        elif labels[i] == 4:
            display2_samples.append(data[i])
            display2_labels.append(labels[i])

    # shuffle the attack samples first
    shuffle_indices = np.arange(len(live_samples))
    random.seed(seed)
    random.shuffle(shuffle_indices)


    # use the balancing ratio
    counter = 0
    for i in shuffle_indices:
        # print(printed1_samples[i], '\n', printed2_samples[i] )

        attack_samples.append(printed1_samples[i])
        attack_labels.append(printed1_labels[i])

        attack_samples.append(printed2_samples[i])
        attack_labels.append(printed2_labels[i])

        attack_samples.append(display1_samples[i])
        attack_labels.append(display1_labels[i])

        attack_samples.append(display2_samples[i])
        attack_labels.append(display2_labels[i])

        counter += 1
        if counter == balance_ratio:
            break


    balanced_data = live_samples + attack_samples
    balanced_labels = live_labels + attack_labels

    return balanced_data, balanced_labels




# -----------------------------------------------------------------------------------------------

def main(args):
    # set the image parameters
    img_rows = args.img_rows
    img_cols = args.img_cols
    img_dim_color = args.img_channels
    # mix_prop = 1.0                                                    # set the value of the mixing proportion


    #############################################################################################################
    ##################################  DEFINING MODEL  ##########################################################
    ##############################################################################################################


    output_1 = cnn_hybrid_color_single(args.img_rows, args.img_cols, args.img_channels)

    model_final = output_1
    print(model_final.summary())  # print the model summary

    plot_model(model_final, to_file='./hog_lbp_rgb-learned',
               show_shapes=True)  # save the model summary as a png file

    # set the learning rate
    lr = args.learning_rate

    # set the optimizer
    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9)

    # model compilation
    model_final.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    # print the value of the learning rate
    print(K.get_value(optimizer.lr))




    # --------------------------------------------------
    #############################################################################################################
    ########################## GETTING TRAINING DATA AND TESTING DATA  ##########################################
    ##############################################################################################################

    # get the training data by calling the pairs function

    train_pairs_r, training_data_r, training_label_r = read_pairs(args.tr_img_lab_r)
    # training_data_r, training_label_r = data_balance(training_data_r, training_label_r, seed= 4)


    #######################################################33
    live_samples_ub = 0
    attack_samples_ub = 0

    live_samples_r = []
    live_labels_r = []
    attack_samples_r = []
    attack_labels_r = []
    # Balance the data

    for i in range(len(training_data_r)):
        if training_label_r[i] == 0:
            live_samples_r.append(training_data_r[i])
            live_labels_r.append(training_label_r[i])

            live_samples_ub += 1
        else:
            attack_samples_r.append(training_data_r[i])
            attack_labels_r.append(training_label_r[i])

            attack_samples_ub += 1

    print("Live samples are %g ,\t attack samples are %g" % (live_samples_ub, attack_samples_ub))

    # compute the difference;
    if live_samples_ub < attack_samples_ub:
        # compute the ratio
        diff = np.int(attack_samples_ub / live_samples_ub)
        print("The difference is %f" % diff)



    dummy = []

    for i in range(len(training_label_r)):
        if training_label_r[i] > 0:
            dummy.append(1)
        else:
            dummy.append(0)

    print(np.unique(dummy))
    weight = class_weight.compute_class_weight('balanced', np.unique(dummy), dummy)
    print(weight)


   #####################################################################################################333


    # given input data the below gives you data and binary labels in a list
    train_img_data_rgb = get_data_label(training_data_r, training_label_r)
    # --------------------------------------------------------------------------------------------

    # read the test data
    test_pairs, test_data_r, test_labels_r = read_pairs(args.tst_img_lab_r)

    # given input data the below gives you data and binary labels in a list
    test_img_data_rgb = get_data_label(test_data_r, test_labels_r)



    # --------------------------------------------------------------------------------------------------
    # make all the data in panda data frame format
    train_df_r = pd.DataFrame(train_img_data_rgb)
    train_df_r.columns = ['id', 'label']

    test_df_r = pd.DataFrame(test_img_data_rgb)
    test_df_r.columns = ['id', 'label']

    datagen = image.ImageDataGenerator()

    train_generator_r = datagen.flow_from_dataframe(
        dataframe=train_df_r,
        directory=None, x_col='id', y_col='label', has_ext=True, batch_size=args.batch_size, seed=42, shuffle=True,
        class_mode="sparse", target_size=(args.img_rows, args.img_cols), color_mode='rgb', interpolation='nearest',
        drop_duplicates=True)

    test_datagen = image.ImageDataGenerator()

    test_generator_r = test_datagen.flow_from_dataframe(
        dataframe=test_df_r,
        directory=None, x_col='id', y_col='label', has_ext=True, batch_size=args.batch_size, seed=42, shuffle=True,
        class_mode="sparse", target_size=(args.img_rows, args.img_cols), color_mode='rgb', interpolation='nearest',
        drop_duplicates=True)


    accuracy = 0
    # --------------------------------------------------------------------------------------------------
    batch_num = 0
    while batch_num < args.max_epochs:

        start_time = time.time()  # initialize the clock
        acc = []
        loss = []

        sub_count = 0

        total_batch = train_generator_r.n // train_generator_r.batch_size

        for i in range(train_generator_r.n // train_generator_r.batch_size):
            x_r, y = next(train_generator_r)

            x_hsv = []

            x_rgb_lbp = []
            x_hsv_lbp = []
            x_ycrcb_lbp = []


            for j in range(x_r.shape[0]):

                x_hsv_data = cv2.cvtColor(x_r[j], cv2.COLOR_RGB2HSV)
                x_ycrcb_data = cv2.cvtColor(x_r[j], cv2.COLOR_RGB2YCrCb)

                r, g, b = cv2.split(x_r[j])
                hue, sat, int = cv2.split(x_hsv_data)
                y_i, cr, cb = cv2.split(x_ycrcb_data)


               # compute the lbp features for rgb image
                lbp_data_r = np.expand_dims(lbp(image=r, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_g = np.expand_dims(lbp(image=g, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_b = np.expand_dims(lbp(image=b, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)

                # compute the lbp features for hsv image
                lbp_data_hue = np.expand_dims(lbp(image=hue, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_sat = np.expand_dims(lbp(image=sat, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_int = np.expand_dims(lbp(image=int, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)

                # compute the lbp features for ycrcb image
                lbp_data_y = np.expand_dims(lbp(image=y_i, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_cr = np.expand_dims(lbp(image=cr, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)
                lbp_data_cb = np.expand_dims(lbp(image=cb, P=args.lbp_pts, R=args.lbp_r, method='uniform'),axis=-1)

                rgb_lbp_f = np.concatenate((lbp_data_r, lbp_data_g, lbp_data_b), axis=-1)
                hsv_lbp_f = np.concatenate((lbp_data_hue, lbp_data_sat, lbp_data_int), axis=-1)
                ycrcb_lbp_f = np.concatenate((lbp_data_y, lbp_data_cr, lbp_data_cb), axis=-1)

                x_rgb_lbp.append(rgb_lbp_f)
                x_hsv_lbp.append(hsv_lbp_f)
                x_ycrcb_lbp.append(ycrcb_lbp_f)

                x_hsv.append(x_hsv_data)





            x_l_rgb = np.asarray(x_rgb_lbp)
            x_l_hsv = np.asarray(x_hsv_lbp)
            x_l_ycrcb = np.asarray(x_ycrcb_lbp)

            x_feature_m = np.concatenate((x_l_rgb, x_l_hsv,x_l_ycrcb), axis=-1)

            x_hsv = np.asarray(x_hsv).astype('float32')/255




            x_rgb = x_r.astype('float32') / 255





            tr_acc1 = model_final.fit([x_hsv, x_feature_m],
                                      y,
                                      class_weight={0: weight[0], 1: weight[1]},
                                      epochs=1,
                                      verbose=0)
            # class_weight = {0: weight[0], 1: weight[1]},


            acc.append(tr_acc1.history['acc'])
            loss.append(tr_acc1.history['loss'])


            sub_count += 1
            sys.stdout.write('batch_count = {0} of {1} \r'.format(sub_count, total_batch))
            sys.stdout.flush()

        train_acc = np.sum(np.asarray(acc))*100 / (train_generator_r.n //train_generator_r.batch_size)
        train_loss = np.sum(np.asarray(loss))*100 / (train_generator_r.n //train_generator_r.batch_size)

        print('training_acc: {0} \t training_loss: {1}'.format(train_acc, train_loss))

        print ('______________________________________________________________________')
        print('Running the evaluations')

        test_acc = []
        test_loss = []
        sub_count = 0

        for i in range(test_generator_r.n // test_generator_r.batch_size):
            x_r, y = next(test_generator_r)

            x_hsv = []
            x_rgb_lbp = []
            x_hsv_lbp = []
            x_ycrcb_lbp = []


            for j in range(x_r.shape[0]):

                x_hsv_data = cv2.cvtColor(x_r[j], cv2.COLOR_RGB2HSV)
                x_ycrcb_data = cv2.cvtColor(x_r[j], cv2.COLOR_RGB2YCrCb)

                r, g, b = cv2.split(x_r[j])
                hue, sat, int = cv2.split(x_hsv_data)
                y_i, cr, cb = cv2.split(x_ycrcb_data)

                # compute the lbp features for rgb image
                lbp_data_r = np.expand_dims(lbp(image=r, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_g = np.expand_dims(lbp(image=g, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_b = np.expand_dims(lbp(image=b, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)

                # compute the lbp features for hsv image
                lbp_data_hue = np.expand_dims(lbp(image=hue, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_sat = np.expand_dims(lbp(image=sat, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_int = np.expand_dims(lbp(image=int, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)

                # compute the lbp features for ycrcb image
                lbp_data_y = np.expand_dims(lbp(image=y_i, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_cr = np.expand_dims(lbp(image=cr, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)
                lbp_data_cb = np.expand_dims(lbp(image=cb, P=args.lbp_pts, R=args.lbp_r, method='uniform'), axis=-1)

                rgb_lbp_f = np.concatenate((lbp_data_r, lbp_data_g, lbp_data_b), axis=-1)
                hsv_lbp_f = np.concatenate((lbp_data_hue, lbp_data_sat, lbp_data_int), axis=-1)
                ycrcb_lbp_f = np.concatenate((lbp_data_y, lbp_data_cr, lbp_data_cb), axis=-1)

                x_rgb_lbp.append(rgb_lbp_f)
                x_hsv_lbp.append(hsv_lbp_f)
                x_ycrcb_lbp.append(ycrcb_lbp_f)

                x_hsv.append(x_hsv_data)


            x_l_rgb = np.asarray(x_rgb_lbp)
            x_l_hsv = np.asarray(x_hsv_lbp)
            x_l_ycrcb = np.asarray(x_ycrcb_lbp)

            x_hsv = np.asarray(x_hsv).astype('float32')/255

            x_feature_m = np.concatenate((x_l_rgb, x_l_hsv, x_l_ycrcb), axis=-1)

            # x_rgb = np.asarray(rgb_data)

            x_rgb = x_r.astype('float32') / 255


            tst_loss = model_final.evaluate([x_hsv,x_feature_m],
                                            y,
                                            verbose=0)
            x1 = model_final.metrics_names

            test_acc.append(tst_loss[x1.index('acc')])
            test_loss.append(tst_loss[x1.index('loss')])


            sub_count +=  1
            sys.stdout.write('epoch_count = {0}\r'.format(sub_count))
            sys.stdout.flush()

        test_acc = np.sum(np.asarray(test_acc)) * 100 / (test_generator_r.n // test_generator_r.batch_size)
        test_loss = np.sum(np.asarray(test_loss)) * 100 / (test_generator_r.n // test_generator_r.batch_size)

        print('test_acc: {0} \t test_loss: {1}'.format(test_acc, test_loss))



        batch_num += 1

        # **********************************************************************************************
        # learning rate schedule update: if learning is done using a single learning give the batch_num below a
        # high value
        if np.mod(batch_num,2):
            lr = lr*0.5
            K.set_value(optimizer.lr, lr)
            print(K.get_value(optimizer.lr))

        # ************************************************************************************************
        # -----------------------------------------------------------------------------------------------

        end_time = time.time() - start_time

        print("Total time taken %f :" % end_time)

        if test_acc > accuracy:
            model_final.save_weights(
                '/home/yaurehman2/Documents/liveness_perturbation/livenet_hybrid_ckpt/oulu/protocol_1/'
                'learned_lbp_rgb_protocol1_all_color_modified_hsv'
                + str(args.lbp_pts) + '_' + str (args.lbp_r) + '_' + '5x5_160x160_online_lbp_rgb_lev1_' + str(args.max_epochs) + '.h5')
            accuracy = test_acc
            print('saved')

def parser_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tr_img_lab_r', type=str,
                        help='directory from where to get the training paths and ground truth',
                        default='/home/yaurehman2/Documents/Newwork/OULU_FACE_Protocol1/train.txt')

    # /home/yaurehman2/Documents/Newwork/REPLY_ATTACK_FACE_Mod_corr/train.txt
    # /home/yaurehman2/Documents/Newwork/OULU_FACE/train.txt
    # /home/yaurehman2/Documents/Newwork/CASIA_FACE_4C/train_4C_M.txt
    parser.add_argument('--tst_img_lab_r', type=str,
                        help='direcotry where test iamges are stored ',
                        default='/home/yaurehman2/Documents/Newwork/OULU_FACE_Protocol1/dev.txt')



    # """**************************************************************************************************************"""

    """Specify the parameters for the CNN Net"""

    parser.add_argument('--batch_size', type=int,
                        help='input batch size to the network', default=32)

    parser.add_argument('--test_batch_size', type=int,
                        help='input test batch size to the network', default=12000)

    parser.add_argument('--max_epochs', type=int,
                        help='maximum number of epochs for training', default=15)

    parser.add_argument('--epoch_batch', type=int,
                        help='Maximum epoch per batch per iteration', default=12000)

    # """**************************************************************************************************************"""

    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate', default=0.01)

    parser.add_argument('--lbp_pts', type=int,
                        help='number of points for lbp', default=8)

    parser.add_argument('--lbp_r', type=int,
                        help='radius for lbp', default=2)

    parser.add_argument('--img_rows', type=int,
                        help='image height', default=64)

    parser.add_argument('--img_cols', type=int,
                        help='image width', default=64)

    parser.add_argument('--img_channels', type=int,
                        help='number of input channels in an image', default=3)

    parser.add_argument('--epoch_flag', type=int,
                        help='determine when to change the learning rate', default=1)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))



