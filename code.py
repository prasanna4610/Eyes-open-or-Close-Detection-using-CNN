#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


img_array = cv2.imread("D:\Internship\dataset\closedeyes.jpg", cv2.IMREAD_GRAYSCALE)


# In[3]:


plt.imshow(img_array, cmap="gray")


# In[4]:


img_array.shape


# In[ ]:





# In[5]:


import os


# In[11]:


Datadirectory = r"D:\Internship\dataset\train"
Classes = ['Closed_Eyes', 'Open_Eyes']
# Datadirectory = 'dataset_new/train/'
# Classes = ['Closed', 'Open']
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
    backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
    plt.imshow(img_array, cmap="gray")
    plt.show()
    break
    break


# In[12]:


img_size = 224
new_array = cv2.resize(backtorgb, (img_size,img_size))
plt.imshow(new_array, cmap="gray")
plt.show()


# In[16]:


training_data = []

def create_training_data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size,img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                    pass


# In[17]:


create_training_data()


# In[18]:


print(len(training_data))


# In[ ]:





# In[19]:


import random
random.shuffle(training_data)


# In[21]:


X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)


# In[22]:


X.shape


# In[23]:


X = X/255.0


# In[24]:


Y = np.array(y)


# In[ ]:





# In[25]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:





# In[26]:


model = tf.keras.applications.mobilenet.MobileNet()


# In[27]:


model.summary()


# In[28]:


base_input = model.layers[0].input


# In[29]:


base_output = model.layers[-4].output


# In[30]:


Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)


# In[31]:


new_model = keras.Model(inputs = base_input, outputs = final_output)


# In[32]:


new_model.summary()


# In[33]:


new_model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[34]:


new_model.fit(X,Y, epochs = 2, validation_split = 0.1)


# In[36]:


new_model.save('model.h5')


# In[37]:


new_model = tf.keras.models.load_model('model.h5')


# In[ ]:





# In[40]:


img_array = cv2.imread(r"D:\Internship\dataset\closedeyes.jpg",cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
new_array = cv2.resize(backtorgb, (img_size, img_size))


# In[42]:


X_input = np.array(new_array).reshape(1, img_size, img_size, 3)


# In[43]:


X_input.shape


# In[44]:


plt.imshow(new_array)


# In[45]:


X_input = X_input/255.0


# In[46]:


prediction = new_model.predict(X_input)


# In[47]:


prediction


# In[ ]:





# In[48]:


img = cv2.imread(r"D:\Internship\dataset\openeyes.jpg")


# In[49]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[ ]:





# In[51]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


# In[52]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[53]:


eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)


# In[55]:


for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)


# In[56]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[59]:


eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)
for x, y,w, h in eyes:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyess = eyeCascade.detectMultiScale(roi_gray)
    if len(eyess) == 0:
        print("eyes not detected")
    else:
        for ex, ey, ew, eh in eyess :
            eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]


# In[60]:


plt.imshow(cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2RGB))


# In[61]:


eyes_roi.shape


# In[62]:


final_img = cv2.resize(eyes_roi, (224,224))
final_img = np.expand_dims(final_img, axis=0)
final_img = final_img/255.0


# In[63]:


new_model.predict(final_img)


# In[ ]:




