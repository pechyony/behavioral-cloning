import random
import math
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# dataset directory
input_dir = '/datadrive/Behavioral-Cloning/data_combined'

# size of the input image
cols = 320
rows = 160

# hyperparameters
top_crop = 70                # number of top rows to be removed 
bottom_crop = 25             # number of bottom rows to be removed
n_augmentations = 6          # number of rotated imaged to be created for each input image
alpha = 0.001                # regularization coefficient
learning_rate = 0.001        # learning rate
batch_size = 24              # mini-batch size
max_epochs = 100             # maximum number of training epochs
patience = 3                 # maximum number of epochs with no improvement in validation error
delta = 0.00001              # minimal change of validation error to be considered as 'improvement'
steering_correction = 0.2    # correction of steering angle for side images

# Load image
def get_image(source_path):
    filename = source_path.split('\\')[-1]
    current_path = input_dir + '/IMG/' + filename

    # cv2.imread creates BGR image, drive.py get images in RGB format,
    # so we need to convert the image from BGR to RGB
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)

    return image

# Transform image using a given transformation matrix
def transform(x, M):
    transformed = np.zeros_like(x)
    for i in range(0,3):
        transformed[:,:,i] = cv2.warpAffine(x[:,:,i],M,(cols,rows))

    return transformed

# Rotate image. Rotation angle ins drawn uniformly from [-45,45]
def rotate(x):
    angle = random.uniform(-45,45)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    return transform(x,M), angle/180*3.1415926

# Generator for training and validation images
def generator(samples, batch_size=24, train=True):
    num_samples = len(samples)
    cameras = ['center','left','right']
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        # Loop over batches of images
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # rad images from the current batch
                for camera in cameras:
                    image = get_image(batch_sample[camera])
                    images.append(image)
                 
                # create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample['steering'])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

                # add flipped images 
                augmented_images, augmented_angles = [], []
                for image, angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_angles.append(angle * -1)
            
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            # rotate training images
            if train:
                X_train_orig = np.copy(X_train)
                y_train_orig = np.copy(y_train)

                # for each image, create n_augmentations rotated images
                for j in range(n_augmentations):
                    X_train_rotated = np.zeros_like(X_train_orig)
                    y_train_rotated = np.zeros_like(y_train_orig)
                    for i in range(0,X_train_orig.shape[0]):
                        X_train_rotated[i,:,:,:], angle = rotate(X_train_orig[i,:,:,:])
                        y_train_rotated[i] = y_train_orig[i] + angle 

                    X_train = np.concatenate((X_train,X_train_rotated),axis=0)
                    y_train = np.concatenate((y_train,y_train_rotated),axis=0)

            # output next batch of images and labels
            yield sklearn.utils.shuffle(X_train, y_train)

# read the list of images
lines = []
with open(input_dir + '/driving_log.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        lines.append(line)

# split images into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator for training examples
train_generator = generator(train_samples, train=True)

# generator for validation examples
validation_generator = generator(validation_samples, train=False)

# define model architecture
model = Sequential()
model.add(Cropping2D(cropping=((top_crop,bottom_crop), (0,0)), input_shape=(rows,cols,3)))    # Cropping layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))                                                # Normalization layer
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu", kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu", kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu", kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Convolution2D(64,3,3,activation="relu", kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Convolution2D(64,3,3,activation="relu", kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Flatten())
model.add(Dense(1164, kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Dense(100, kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Dense(50, kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Dense(10, kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))
model.add(Dense(1, kernel_regularizer=regularizers.l2(alpha), bias_regularizer=regularizers.l2(alpha)))

# define optimizer and objective function
adam = optimizers.Adam(lr=learning_rate) 
model.compile(loss='mse', optimizer=adam)

# Keras callback for saving the best model seen so far
# the models are saved in model_<epoch number>.h5 files
best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

# Keras callback for early stopping: stop when there are 4 epochs with less than 0.00001 in the validation error
early_stopping = EarlyStopping(min_delta=delta, patience=patience, mode='min')

# train the model, compute validation error after each epoch, save the best model seen so far, stop
# when there are 4 epochs with less than 0.00001 in the validation error
model.fit_generator(train_generator, samples_per_epoch=round(len(train_samples)/batch_size),
                    validation_data=validation_generator, nb_val_samples=round(len(validation_samples)/batch_size),
                    nb_epoch=max_epochs, callbacks=[best_val, early_stopping])
