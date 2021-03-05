from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
import skimage.io
import tensorflow.keras.backend as K
from online_training_ArcFace_Stage_2 import *
import random
import math, time
# ----------------------------------------------------------------------------
color = True
label_split = '5_rgb'
count = 1000
LOAD_MODEL = './models/DogFaceNet_ArcFace_Triplet/triplet_arc_loss_0.33.h5'  # model path
MODEL_NAME = 'DogFaceNet_ArcFace_Triplet'
DATA_PATH = './data/Dog FaceNet DB1/more_than_5_rgb/'
TEST_SPLIT = 0.1

# ----------------------------------------------------------------------------
assert os.path.isdir(DATA_PATH), '[Error] Provided PATH for dataset does not exist.'
print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(DATA_PATH):
    dirs.sort()
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
assert len(labels)!=0, '[Error] No data provided.'
print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))
nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

# ----------------------------------------------------------------------------
nbof_test = int(TEST_SPLIT*nbof_classes)
keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]
filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

# ----------------------------------------------------------------------------
alpha = 0.3

def TripletLoss():
    def triplet_loss(y_true,y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sum(K.square(a-p),-1)
        an = K.sum(K.square(a-n),-1)
        return K.sum(tf.nn.relu(ap - an + alpha))
    return triplet_loss

def TripletAcc():
    def triplet_acc(y_true,y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sum(K.square(a-p),-1)
        an = K.sum(K.square(a-n),-1)
        return K.less(ap + alpha,an)
    return triplet_acc

def TripletArcLoss():
    def triplet_arc_loss(y_true,y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sqrt(K.sum(K.square(a-p),-1))
        an = K.sqrt(K.sum(K.square(a-n),-1))
        cos_ap = (tf.constant(2.) - ap) / tf.constant(2.)
        cos_an = (tf.constant(2.) - an) / tf.constant(2.)
        theta_ap = tf.math.acos(cos_ap)
        theta_an = tf.math.acos(cos_an)
        return K.sum(tf.nn.relu(theta_ap - theta_an + alpha))
    return triplet_arc_loss

def TripletArcAcc():
    def triplet_arc_acc(y_true, y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sqrt(K.sum(K.square(a - p), -1))
        an = K.sqrt(K.sum(K.square(a - n), -1))
        cos_ap = (tf.constant(2.) - ap) / tf.constant(2.)
        cos_an = (tf.constant(2.) - an) / tf.constant(2.)
        theta_ap = tf.math.acos(cos_ap)
        theta_an = tf.math.acos(cos_an)
        return K.less(theta_ap + alpha, theta_an)
    return triplet_arc_acc

#----------------------------------------------------------------------------
model = tf.keras.models.load_model(LOAD_MODEL, custom_objects={'triplet_arc_loss': TripletArcLoss(), 'triplet_arc_acc': TripletArcAcc()})
# model = tf.keras.models.load_model(LOAD_MODEL, custom_objects={'triplet_loss': TripletLoss(), 'triplet_acc': TripletAcc()})

#----------------------------------------------------------------------------
filenames_test_2D = []
labels_test_2D = []
test_class_list = []
test_class_list2 = []
label_num = 0
for i in range(len(filenames_test)):
    if labels_test[i] == label_num:
        test_class_list.append(filenames_test[i])
        test_class_list2.append(labels_test[i])
    else:
        filenames_test_2D.append(test_class_list)
        labels_test_2D.append(test_class_list2)
        test_class_list = [filenames_test[i]]
        test_class_list2 = [labels_test[i]]
        label_num = labels_test[i]
filenames_test_2D.append(test_class_list)
labels_test_2D.append(test_class_list2)

acc1_max, acc5_max, acc1_min, acc5_min, acc1_sum, acc5_sum = 0, 0, 100, 100, 0, 0

def load_single_image(filename, color):
    if color:
        h, w, c = (224, 224, 3)
    elif not color:
        h, w, c = (224, 224, 1)
    if 'jpg' in filename:
        temp = skimage.io.imread(filename).reshape(1, h, w, c)
        image = temp / 255.0
    return image

for m in range(count):
    print(m)
    sec1 = time.time()
    sub_train_filenames_test = np.empty(0)
    sub_train_labels_test = np.empty(0)
    sub_test_filenames_test = np.empty(0)
    sub_test_labels_test = np.empty(0)
    for i in range(len(filenames_test_2D)):
        k = random.sample(filenames_test_2D[i], len(filenames_test_2D[i]) - 1)
        for j in range(len(filenames_test_2D[i])):
            if filenames_test_2D[i][j] in k:
                sub_test_filenames_test = np.append(sub_test_filenames_test, filenames_test_2D[i][j])
                sub_test_labels_test = np.append(sub_test_labels_test, labels_test_2D[i][j])
            else:
                sub_train_filenames_test = np.append(sub_train_filenames_test, filenames_test_2D[i][j])
                sub_train_labels_test = np.append(sub_train_labels_test, labels_test_2D[i][j])

    check_dists = []
    num = 0
    tr, fa, tr5, fa5 = 0, 0, 0, 0

    emb_train_list = []
    for i in sub_train_filenames_test:
        emb_train_list.append(model.predict(load_single_image(i, color)))
    emb_test_list = []
    for i in sub_test_filenames_test:
        emb_test_list.append(model.predict(load_single_image(i, color)))

    dist_dir = {}
    for i in sub_train_filenames_test:
        temp_label = i.split(label_split+'/')[1].split('/')[-1]
        dist_dir[temp_label] = []

    for v, i in enumerate(sub_test_filenames_test):
        num += 1
        new_dists = {}
        new_dists2 = {}
        test_label = i.split(label_split+'/')[1].split('/')[0]
        test_label2 = i.split(label_split+'/')[1].split('/')[-1]
        emb_test = emb_test_list[v]
        for k in range(len(sub_train_filenames_test)):
            emb_train = emb_train_list[k]
            temp_dist = math.sqrt(np.sum((emb_train[0]-emb_test[0])**2))
            temp_label = sub_train_filenames_test[k].split(label_split+'/')[1].split('/')[0]
            temp_label2 = sub_train_filenames_test[k].split(label_split+'/')[1].split('/')[-1]
            check_dists.append(temp_dist)
            try:
                if new_dists[temp_label] > temp_dist:
                    new_dists[temp_label] = temp_dist
            except:
                new_dists[temp_label] = temp_dist
                new_dists2[temp_label2] = temp_dist

        sorted_dic = sorted(new_dists.items(), key=lambda x: x[1])
        sorted_dic2 = sorted(new_dists2.items(), key=lambda x: x[1])
        predicted_label = sorted_dic[0][0]
        predicted_label2 = sorted_dic2[0][0]
        predicted_labels_5 = [sorted_dic[0][0],sorted_dic[1][0],sorted_dic[2][0],sorted_dic[3][0],sorted_dic[4][0]]

        dist_dir[predicted_label2].append(test_label2)
        if test_label == predicted_label:
            tr += 1
        else:
            fa += 1

        if test_label in predicted_labels_5:
            tr5 += 1
        else:
            fa5 += 1

    if tr + fa == num and tr5 + fa5 == num:
        acc1 = (tr/num) * 100
        acc5 = (tr5/num) * 100
        print('[%s model result]'%MODEL_NAME)
        print('acc1: %.2f%% (tr: %d)\nacc5: %.2f%% (tr5: %d)'%(acc1,tr,acc5,tr5))

        if acc1 > acc1_max:
            acc1_max = acc1
        if acc1 < acc1_min:
            acc1_min = acc1
        if acc5 > acc5_max:
            acc5_max = acc5
        if acc5 < acc5_min:
            acc5_min = acc5
        acc1_sum += acc1
        acc5_sum += acc5

    sec2 = time.time()
    # print('epoch time: %.2f'%(sec2-sec1))

print('Summary : %s'%MODEL_NAME)
print('[1] acc1 MAX: %.2f %%'%acc1_max)
print('[2] acc1 MIN: %.2f %%'%acc1_min)
print('[3] acc1 AVG: %.2f %%'%(acc1_sum/count))
print('[4] acc5 MAX: %.2f %%'%acc5_max)
print('[5] acc5 MIN: %.2f %%'%acc5_min)
print('[6] acc5 AVG: %.2f %%'%(acc5_sum/count))