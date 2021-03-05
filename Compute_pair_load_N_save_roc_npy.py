from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training_ArcFace_Triplet import *
import cv2

# ----------------------------------------------------------------------------
MODEL_NAME = 'DogFaceNet_ArcFace_Stage_2'
LOAD_MODEL = './models/DogFaceNet_ArcFace_Stage_2/triplet_loss_0.26.h5'
PATH = './data/Dog FaceNet DB1/more_than_5_rgb/'
SIZE = (224,224,3)
TEST_SPLIT = 0.1

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

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

#----------------------------------------------------------------------------
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
model = tf.keras.models.load_model(LOAD_MODEL, custom_objects={'triplet_loss': TripletArcLoss(), 'triplet_acc': TripletArcAcc()})

#----------------------------------------------------------------------------
with open('./results/ROC/pairs_1001.txt',mode='r') as f:
    temp = f.readlines()

pairs = []
for file in temp:
    pairs.append(file.replace('\n','').replace('//192.168.0.36/research/Dataset/dog/DogFaceNet_Dataset_224_1', './data/Dog FaceNet DB1'))

issame = np.load('results/ROC/issame_1001.npy')

#----------------------------------------------------------------------------
print('Verification task, model evaluation...')

predict=model.predict_generator(predict_generator(pairs, 32), steps=np.ceil(len(pairs)/32))
emb1 = predict[0::2]
emb2 = predict[1::2]

diff = np.square(emb1-emb2)
dist = np.sum(diff,1)

best = 0
best_t = 0
thresholds = np.arange(0.001,8,0.001)
for i in range(len(thresholds)):
    less = np.less(dist, thresholds[i])
    acc = np.logical_not(np.logical_xor(less, issame))
    acc = acc.astype(float)
    out = np.sum(acc)
    out = out/len(acc)
    if out > best:
        best_t = thresholds[i]
        best = out

print('Done.\n')
print("Best threshold: " + str(best_t))
print("Best accuracy: " + str(best))

t = best_t
fa = []
fr = []
for i in range(len(dist)):
    if issame[i] == 0 and dist[i]<t:
        fa += [i]
    if issame[i] == 1 and dist[i]>t:
        fr += [i]

temp1 = []
temp2 = []
for i in fa:
    temp1.append(i)
    temp2.append('fa')

for j in fr:
    temp1.append(j)
    temp2.append('fr')

dic = {'i':temp1,'fafr':temp2}
import pandas as pd
df = pd.DataFrame(dic)
df.to_csv('results/ROC/%s_fafr_1101.csv'%MODEL_NAME)

s = 10
sr = 20
n = 8
print('Ground truth: {:s}'.format(str(issame[s:(n+s)])))
threshold = 0.3
less = np.less(dist, threshold)
acc = np.logical_not(np.logical_xor(less, issame))
acc = acc.astype(float)
out = np.sum(acc)
out = out/len(acc)

print("Accuracy: " + str(out))

#----------------------------------------------------------------------------
tprs = np.empty(len(thresholds))
fprs = np.empty(len(thresholds))

p = np.sum(issame.astype(float))
n = np.sum(np.logical_not(issame).astype(float))

for i in range(len(thresholds)):
    logical_pred = np.less(dist, thresholds[i])
    tp = np.sum(np.logical_and(logical_pred,issame).astype(float))
    fp = np.sum(np.logical_and(logical_pred,np.logical_not(issame)).astype(float))
    tprs[i] = tp/p
    fprs[i] = fp/n

np.save('results/ROC/%s_roc.npy'%MODEL_NAME, [tprs, fprs])