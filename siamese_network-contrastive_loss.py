#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.clear_all_output();')


# In[1]:


import os 
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from glob import glob
import matplotlib.image as image
import time
# import splitfolders 

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import tensorflow.keras.backend as Backend

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda


# In[2]:


dir_name = "./data_2500"

files='**/*.jpg'
filenames = glob(os.path.join(dir_name, files))
classes = [os.path.basename(os.path.dirname(name)) for name in filenames]
count = list(Counter(classes).items())
print("Class count:", count);


# In[3]:


class_names = os.listdir(dir_name) # Get names of classes
class_name2id = { label: index for index, label in enumerate(class_names) } # Map class names to integer labels
print("Classes:", class_name2id)

labels = [class_name2id[c] for c in classes]
Counter(labels)


# In[4]:


# config
IMG_SHAPE = (102, 102, 3)
IMG_SIZE = (102, 102) 
BATCH_SIZE = 16
# EPOCHS = 32
EPOCHS = 20


# In[5]:


def get_image(filename):
    img_obj = load_img(filename, target_size=IMG_SIZE) # image object
    numpy_image = img_to_array(img_obj) # image object -> pixel array
    return numpy_image

def map_dataset(filenames):
    data = []
    for i in range(len(filenames)):
        data.append(get_image(filenames[i]))
#         if(i % 1000 == 0): print(i, 'images loaded')
    return np.array(data)/255.0

def get_files_labels(path):
    files = glob(os.path.join(path, '**/*.jpg'))
    classes = [os.path.basename(os.path.dirname(name)) for name in files]
    labels = [class_name2id[c] for c in classes]
    return files, np.array(labels)
    


# In[6]:


train_path = "splitted_data/train"
train_files, train_y = get_files_labels(train_path)
print("Train:", Counter(train_y))

test_path = "splitted_data/test"
test_files, test_y = get_files_labels(test_path)
print("Test:", Counter(test_y))

val_path = "splitted_data/val"
val_files, val_y = get_files_labels(val_path)
print("Validation:", Counter(val_y))


# In[7]:


get_ipython().run_cell_magic('time', '', 'train_X = map_dataset(train_files)\ntest_X = map_dataset(test_files)\nval_X = map_dataset(val_files)')


# In[8]:


print(train_X.shape)
print(train_y.shape)
# print(train_X[0])


# In[ ]:





# # Make Pairs for Siamese Network

# In[9]:


def make_pos_neg_pairs(X, y):
    pairs = [] # (image, image) pair 
    labels = [] # 0 means negative pair, otherwise, positive
    
    class_count = len(np.unique(y))
    class_indexes = [np.where(y == i)[0] for i in range(0, class_count)]
    
    for i in range(len(X)):
        cur_img = X[i]
        cur_label = y[i]
#         print(cur_label)
        
        # positive: pick an image from same class randomly
        pos_i = np.random.choice(class_indexes[cur_label]) 
        pos_img = X[pos_i]
        pairs.append([cur_img, pos_img])
        labels.append([1])
        
        # negative: pick an image from other classes randomly
        neg_ids = np.where(y != cur_label)[0]
        neg_img = X[np.random.choice(neg_ids)]        
        pairs.append([cur_img, neg_img])
        labels.append([0])
        
    return (np.array(pairs), np.array(labels))


# In[10]:


get_ipython().run_cell_magic('time', '', '\n(train_paris, train_labels) = make_pos_neg_pairs(train_X, train_y)\n(val_paris, val_labels) = make_pos_neg_pairs(val_X, val_y)')


# In[11]:


np.unique(train_labels, return_counts=True)


# In[12]:


print(train_paris.shape)
print(train_labels.shape)
print(val_paris.shape)
print(val_labels.shape)


# In[13]:


def show(ax, image, title):
    ax.imshow(image)
    ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
fig = plt.figure(figsize=(9, 9))
axs = fig.subplots(3, 3)
for i in range(3):
    show(axs[i, 0], train_paris[2*i][0], "anchor")
    show(axs[i, 1], train_paris[2*i][1], "positive")
    show(axs[i, 2], train_paris[2*i+1][1], "negative")


# # Build Siamese Network

# In[14]:


def siamese_model(shape, embedding_d=512): 
    inputs = Input(shape)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    
#     x = GlobalAveragePooling2D()(x)
    x = Flatten()(x) # instade of GlobalAveragePooling2D
    
    x = Dense(2048, activation="relu")(x)

    outputs = Dense(embedding_d)(x)
    
    model = Model(inputs, outputs)
    
    return model


# In[15]:


def cal_euclidean_distance(vector_tuple):
    (vector_1, vector_2) = vector_tuple
    sqr_sum = Backend.sum(Backend.square(vector_1 - vector_2), axis=1, keepdims=True)
    distance = Backend.sqrt(Backend.maximum(sqr_sum, Backend.epsilon()))
    return distance   


# In[16]:


input_1 = Input(shape=IMG_SHAPE)
input_2 = Input(shape=IMG_SHAPE)
ext_features = siamese_model(IMG_SHAPE)
features_1 = ext_features(input_1)
features_2 = ext_features(input_2)
euc_distance = Lambda(cal_euclidean_distance)([features_1, features_2])
sia_model = Model(inputs=[input_1, input_2], outputs=euc_distance)
sia_model.summary()


# In[17]:


ext_features.summary()


# # Trianing Model 

# In[18]:


import time


# In[19]:


def cal_loss(y, pred, margin=1):
    y = tf.cast(y, pred.dtype)
    square = Backend.square(pred)
    margin_square = Backend.square(Backend.maximum(margin - pred, 0))
    loss = Backend.mean(y * square + (1 - y) * margin_square)
    return loss


# In[20]:


sia_model.compile(loss=cal_loss, optimizer="adam")

start_time = time.time()

history = sia_model.fit(
    [train_paris[:, 0], train_paris[:, 1]], train_labels[:],
    validation_data=([val_paris[:, 0], val_paris[:, 1]], val_labels[:]),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS)

print(f"\nTrained for {(time.time() - start_time) / 60} minutes ")


# In[21]:


# print(history.history["loss"])


# In[21]:


plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")


# In[ ]:


# (test_paris, test_labels) = make_pos_neg_pairs(test_X, test_y)


# In[ ]:


# for i in range(5):
#     imageA = np.expand_dims(test_paris[i][0], axis=0)
#     imageB = np.expand_dims(test_paris[i][1], axis=0)
#     preds = sia_model.predict([imageA, imageB])
#     proba = preds[0][0]
    
#     fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 4))
#     plt.suptitle("Distance: {:.2f}".format(proba))
    
#     ax = fig.add_subplot(1, 2, 1)
#     plt.imshow(test_paris[i][0])
#     plt.axis("off")
    
#     ax = fig.add_subplot(1, 2, 2)
#     plt.imshow(test_paris[i][1])
#     plt.axis("off")
    
#     plt.show()
    


# In[27]:


# ext_features.save('ext_features.h5')


# # Single layer classifier

# In[2]:


# ext_features = tf.keras.models.load_model("ext_features.h5")


# In[13]:


# val_fetures


# In[22]:


# train_fetures = ext_features(train_X)
val_fetures = ext_features(val_X)
test_fetures = ext_features(test_X)

train_fetures = ext_features(train_X[:2000])
train_fetures2 = ext_features(train_X[2000:4000])
train_fetures3 = ext_features(train_X[4000:6000])
train_fetures4 = ext_features(train_X[6000:])

# val_fetures1 = ext_features(val_X[:500])
# val_fetures2 = ext_features(val_X[500:])
# test_fetures1 = ext_features(test_X[:500])
# test_fetures2 = ext_features(test_X[500:])


# In[21]:


# train_fetures = ext_features(train_X[:500])
# train_fetures2 = ext_features(train_X[500:1000])
# train_fetures3 = ext_features(train_X[1000:1500])
# train_fetures4 = ext_features(train_X[1500:2000])
# train_fetures5 = ext_features(train_X[2000:2500])
# train_fetures6 = ext_features(train_X[2500:3000])
# train_fetures7 = ext_features(train_X[3000:3500])
# train_fetures8 = ext_features(train_X[3500:4000])
# train_fetures9 = ext_features(train_X[4000:4500])
# # train_fetures10 = ext_features(train_X[4000:4500])
# train_fetures11 = ext_features(train_X[4500:5000])
# train_fetures12 = ext_features(train_X[5000:5500])
# train_fetures13 = ext_features(train_X[5500:6000])
# train_fetures14 = ext_features(train_X[6000:6500])
# train_fetures15 = ext_features(train_X[6500:7000])
# train_fetures16 = ext_features(train_X[7000:7500])
# train_fetures17 = ext_features(train_X[7500:8000])

# train_fetures0 = np.concatenate((train_fetures,train_fetures2,train_fetures3, train_fetures4, train_fetures5, 
#                                  train_fetures6, train_fetures7, train_fetures8, train_fetures9,
#                                 train_fetures11, train_fetures12, train_fetures13, train_fetures14, train_fetures15,
#                                 train_fetures16, train_fetures17))


# In[23]:


train_fetures0 = np.concatenate((train_fetures,train_fetures2,train_fetures3, train_fetures4))


# In[24]:


train_fetures0.shape, train_X.shape, train_y.shape


# In[35]:


# class_model = tf.keras.Sequential(name="class_model")
# class_model.add(tf.keras.layers.Dense(128, activation='relu', name="dense_layer1"))
# class_model.add(tf.keras.layers.BatchNormalization())
# class_model.add(tf.keras.layers.Dropout(0.5))
# class_model.add(tf.keras.layers.Dense(4, activation='softmax', name="predictions"))
# # class_model.add(layers.Dense(4, name="layer3"))


# In[25]:


class_model = tf.keras.Sequential(name="class_model")
# class_model.add(tf.keras.layers.Dense(128, activation='relu', name="dense_layer1"))
# class_model.add(tf.keras.layers.BatchNormalization())
# class_model.add(tf.keras.layers.Dropout(0.5))
class_model.add(tf.keras.layers.Dense(4, activation='softmax', name="predictions"))
# class_model.add(layers.Dense(4, name="layer3"))


# In[26]:


from tensorflow.keras.optimizers import SGD, Adam
import time

# SGD(lr=0.001, momentum=0.9)
# SGD(lr=0.00001, momentum=0.99)
class_model.compile(optimizer=Adam(lr=0.001), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    metrics=["accuracy"]
                    )

start_time = time.time()

hist = class_model.fit(x=train_fetures0, y= train_y, 
                          epochs=30, 
                          validation_data=(val_fetures, val_y),
)

print(f"\nTrained for {(time.time() - start_time) / 60} minutes ")


# In[27]:


from sklearn.metrics import classification_report

y_pred = class_model.predict(test_fetures)

# calculate classification accuracy
report = classification_report(test_y, np.argmax(y_pred, axis=1), target_names=class_names, digits=4)
print(report)


# In[28]:


plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")


# In[29]:


plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
plt.plot(hist.history["accuracy"], label="train_accuracy")
plt.plot(hist.history["val_accuracy"], label="val_accuracy")
plt.title("Training accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend(loc="lower left")


# In[ ]:





# # References
# - https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
