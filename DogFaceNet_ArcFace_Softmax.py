from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, BatchNormalization
import tensorflow as tf
import math
import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training_ArcFace_Softmax import *
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

#----------------------------------------------------------------------------
NET_NAME    = 'ArcFace_Softmax'
VERSION_NAME = 'v1'
PATH = './data/Dog FaceNet DB1/more_than_5_rgb/'
PATH_LOG = './logs/DogFaceNet_ArcFace_Softmax/' + VERSION_NAME +'/'
PATH_MODEL = './models/DogFaceNet_ArcFace_Softmax/' + VERSION_NAME +'/'

if not os.path.isdir(PATH_LOG):
    os.makedirs(PATH_LOG)
if not os.path.isdir(PATH_LOG + "metrics"):
    os.makedirs(PATH_LOG + "metrics")
if not os.path.isdir(PATH_MODEL):
    os.makedirs(PATH_MODEL)

SIZE = (224,224,3)
TEST_SPLIT  = 0.1
TOTAL_EPOCHS = 3
STEPS_PER_EPOCH = 300
VALIDATION_STEPS = 30

#----------------------------------------------------------------------------
assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'
print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
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

#----------------------------------------------------------------------------
nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]
num_classes = nbof_classes-nbof_test

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(num_classes))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

labels_rev = []
lab = -1
s = 0
for i in labels_train:
    if i != s:
        s = i
        lab = lab + 1
        labels_rev.append(lab)
    elif i == s:
        labels_rev.append(lab)
labels_rev = np.array(labels_rev, dtype='int32')

labels_rev2 = []
lab = 0
s = 0
for i in labels_test:
    if i != s:
        s = i
        lab = lab + 1
        labels_rev2.append(lab)
    elif i == s:
        labels_rev2.append(lab)
labels_rev2 = np.array(labels_rev2, dtype='int32')

#----------------------------------------------------------------------------
logist_scale = 32
w_decay = 5e-4
alpha = 0.3
beta = 0.3
ArcFace_weight = K.variable(1.0)
embds_weight = K.variable(1.0)

def SoftmaxLoss():
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss

def TripletLoss():
    """tiplet loss"""
    def triplet_loss(y_true,y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sum(K.square(a-p),-1)
        an = K.sum(K.square(a-n),-1)
        return K.sum(tf.nn.relu(ap - an + alpha))
    return triplet_loss

def TripletAcc():
    """triplet accuracy"""
    def triplet_acc(y_true,y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sum(K.square(a-p),-1)
        an = K.sum(K.square(a-n),-1)
        return K.less(ap + alpha,an)
    return triplet_acc

def TripletArcAcc():
    """triplet Arc accuracy"""
    def triplet_arc_acc(y_true, y_pred):
        a = y_pred[0::3]
        p = y_pred[1::3]
        n = y_pred[2::3]
        ap = K.sqrt(K.sum(K.square(a - p), -1))
        an = K.sqrt(K.sum(K.square(a - n), -1))
        cos_ap = (2. - ap) / 2.
        cos_an = (2. - an) / 2.
        theta_ap = math.acos(cos_ap)
        theta_an = math.acos(cos_an)
        return K.less(theta_ap + alpha, theta_an)
    return triplet_arc_acc

#----------------------------------------------------------------------------
print('Defining model {:s} ...'.format(NET_NAME))
def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

emb_size = 32
inputs = Input(shape=SIZE, name='input')
class_labels = Input([None], name='label')

x = Conv2D(16, (7, 7), (2, 2), use_bias=False, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3))(x)

for layer in [16,32,64,128,512]:

    x = Conv2D(layer, (3, 3), strides=(2,2), use_bias=False, activation='relu', padding='same')(x)
    r = BatchNormalization()(x)

    x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
    x = BatchNormalization()(x)
    r = Add()([r,x])

    x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
    x = BatchNormalization()(x)
    x = Add()([r,x])

x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(emb_size, use_bias=False)(x)
logist = ArcMarginPenaltyLogists(num_classes=num_classes, margin=alpha, logist_scale=logist_scale, name='ArcFace')(x, labels=class_labels)

ArcFace_model = tf.keras.Model([inputs, class_labels], logist)
ArcFace_model.compile(loss={'ArcFace':SoftmaxLoss()},
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics={'ArcFace':SparseCategoricalAccuracy()})

model_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG)

#----------------------------------------------------------------------------
histories = []
crt_loss = 100.0
crt_acc = 100.0
batch_size = 3*10
nbof_subclasses = 40

train_data = ArcFace_image_generator(filenames_train, labels_rev, len(filenames_train)//STEPS_PER_EPOCH)
histories += [ArcFace_model.fit_generator(train_data,
                                          epochs=TOTAL_EPOCHS,
                                          steps_per_epoch=STEPS_PER_EPOCH)]

ArcFace_model.save('{:s}{:s}.h5'.format(PATH_MODEL,NET_NAME))