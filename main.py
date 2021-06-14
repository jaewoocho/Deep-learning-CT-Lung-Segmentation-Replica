import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Activation, Conv2D, Flatten, Dense, Maxpooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLRonPlateau

# _________________________________________________________________________________________
# Load DataSets

# Trainning datasets are preprocessed and downloaded from Github 

# CT Image data
x_train = np.load('#filepath/dataset/x_train.npy')
# Lung segmentation mask image 
y_train = np.load('#filepath/dataset/y_train.npy')

# Load x,y validation models
x_val = np.load('#filepath/dataset/x_val/npy')
y_val = np.load('#filepath/dataset/y_val/npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# (Trainning Sets, Length, Width, Color Channel)
# Training sets: 240
# Validation Sets: 27
# 256 x 256 size image 
# Channel 1 as gray scale 

# _________________________________________________________________________________________

# Build Model
# Input size 256 x 256 on channel 1 gray scale
inputs = input(shape=(256, 256, 1))

# Keras Conv2D is a 2D Convolution Layer
# This Layer creates a convolution kernel that is wind with layer input which helps produce a tensor of outputs

# Max pooling operation for 2D spatial data; Downsamples the input representation
# by taking the maximum value over the window defined by pool_size for each dimension along the features axis

# Creates a kernel to downsize image by 2 times 
net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2,  padding='same)(net)

# Creates a kernel to downsize image by 2 times 
net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2,  padding='same)(net)

# Creates a kernel to downsize image by 2 times 
net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2,  padding='same)(net)

# Dense Layer is added for better training results
net = Dense(128, activation = 'relu')(net)

# Creates a kernel to upsize iamge by 2 times
net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

# Creates a kernel to upsize iamge by 2 times
net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

#Creates a kernel to upsize image to match to channel 1 as the original image
net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

model.summary()


# _________________________________________________________________________________________

# Training model

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[ReduceLROnPlateagu(monitor=''val_loss, factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)])

# _________________________________________________________________________________________

# Evaluation
fig, ax = plt.subplots(2, 2, figsize=(10,7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.hisotry['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')


ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.hisotry['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')

# The Upper two graphs(Line) are the trainning results
# The Lower two graphs(dot) are the validation results 
# All of the graphs are highly accurately trained 

# _________________________________________________________________________________________
# Final Results with Prediction
preds = model.predict(x_val)

fig, ax= plt.subplots(len(x_val), 3, figsize=(10, 100))

for i, pred in enumerate(preds):
  # CT Data
  ax[i,0].imshow(x_val[i].squeeze(), cmap='gray')
  # Actual Data
  ax[i,1].imshow(y_val[i].squeeze(), cmap='gray')
  # Predicted Data
  ax[i,2].imshow(pred.squeeze(), cmap='gray')









