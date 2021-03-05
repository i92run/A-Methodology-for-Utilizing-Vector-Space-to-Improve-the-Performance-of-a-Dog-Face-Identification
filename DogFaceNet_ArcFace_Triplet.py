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
from online_training_ArcFace_Triplet import *

#----------------------------------------------------------------------------
NET_NAME    = 'ArcFace_Triplet'
VERSION_NAME = 'v2'
PATH = './data/Dog FaceNet DB1/more_than_5_rgb/'
PATH_LOG = './logs/DogFaceNet_ArcFace_Triplet/' + VERSION_NAME +'/'
PATH_MODEL = './models/DogFaceNet_ArcFace_Triplet/' + VERSION_NAME +'/'

if not os.path.isdir(PATH_LOG):
    os.makedirs(PATH_LOG)
if not os.path.isdir(PATH_LOG + "metrics"):
    os.makedirs(PATH_LOG + "metrics")
if not os.path.isdir(PATH_MODEL):
    os.makedirs(PATH_MODEL)

SIZE = (224,224,3)
TEST_SPLIT  = 0.1
epoch = 1000
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

#----------------------------------------------------------------------------
alpha = 0.3
beta = 0.3
ArcFace_weight = K.variable(1.0)
embds_weight = K.variable(1.0)

def TripletArcLoss():
    """tiplet loss"""
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
    """triplet Arc accuracy"""
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
print('Defining model {:s} ...'.format(NET_NAME))

emb_size = 32
inputs = Input(shape=SIZE, name='input')

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
embds = Lambda(lambda x: tf.nn.l2_normalize(x,axis=-1), name='ArcFace_embds')(x)

ArcFace_model = tf.keras.Model(inputs, embds)
ArcFace_model.compile(loss=TripletArcLoss(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=[TripletArcAcc()])

model_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG)
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_MODEL + 'triplet_arc_acc_{val_triplet_arc_acc:.2f}.h5',
    save_weights_only=False,
    monitor='val_triplet_arc_acc',
    mode='max',
    save_best_only=True)

model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_MODEL + 'triplet_arc_loss_{val_loss:.2f}.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

#----------------------------------------------------------------------------
histories = []
crt_loss = 100.0
crt_acc = 100.0
batch_size = 3*10
nbof_subclasses = 40

data = online_adaptive_hard_image_generators()
train_data = online_adaptive_hard_image_generators()
train_datas = train_data.online_adaptive_hard_image_generator(filenames_train, labels_train, ArcFace_model, crt_loss, batch_size, nbof_subclasses=nbof_subclasses)

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.int32)
sess = tf.Session()
tf.summary.scalar('hard triplet ratio', X1)
tf.summary.scalar('nbof hard triplets', X2)
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter(PATH_LOG + "metrics", sess.graph)
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        train_data.hard_triplet_ratio = np.exp(-train_data.loss * 10 / batch_size)
        train_data.nbof_hard_triplets = int(batch_size // 3 * train_data.hard_triplet_ratio)
        print("Current hard triplet ratio: " + str(train_data.hard_triplet_ratio))
        print("Current nbof triplet ratio: " + str(train_data.nbof_hard_triplets))
    def on_epoch_end(self, epoch, logs=None):
        summary = sess.run(merge, feed_dict={X1: train_data.hard_triplet_ratio,X2: train_data.nbof_hard_triplets})
        writer.add_summary(summary, epoch)
        train_data.loss = logs['loss']
        if epoch % 10 == 9:
            ArcFace_model.save('{:s}{:s}_{:d}.h5'.format(PATH_MODEL, NET_NAME, epoch))

for images_batch,labels_batch in data.online_adaptive_hard_image_generator(
    filenames_train,
    labels_train,
    ArcFace_model,
    crt_acc,
    batch_size,
    nbof_subclasses=nbof_subclasses):
    h = ArcFace_model.train_on_batch(images_batch,labels_batch)
    break

histories += [ArcFace_model.fit_generator(
    train_datas,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=epoch,
    validation_data=image_generator(filenames_test,labels_test,batch_size,use_aug=False),
    validation_steps=VALIDATION_STEPS,
    verbose=2,
    callbacks=[model_tensorboard, model_checkpoint_callback1, model_checkpoint_callback2, CustomCallback()])]