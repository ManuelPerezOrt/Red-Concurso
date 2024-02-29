import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten, Add
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import os
from glob import glob
import numpy as np

train_dir = 'train' 
test_dir = 'test'

cat_files_path = os.path.join(train_dir, 'cat/*')
dog_files_path = os.path.join(train_dir, 'dog/*')
cat_files_path_test = os.path.join(test_dir, 'cat/*')
dog_files_path_test = os.path.join(test_dir, 'dog/*')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))
cat_files_test = sorted(glob(cat_files_path_test))
dog_files_test = sorted(glob(dog_files_path_test))

n_files = len(cat_files) + len(dog_files)
print(n_files)
n_files_test = len(cat_files_test) + len(dog_files_test)
print(n_files_test)

ih, iw = 150, 150 
input_shape = (ih, iw, 3)

num_class = 2 
epochs = 30 

batch_size = 50 
num_train = n_files 
num_test =  n_files_test 
epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

print(train_dir)
gentrain = ImageDataGenerator(rescale=1. / 255.)

train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

print(test_dir)
gentest = ImageDataGenerator(rescale=1. / 255)

test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

inputs = keras.Input(shape=(ih, iw, 3))
x = inputs

# Primer bloque residual
x1 = Conv2D(10, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
x1 = Activation('relu')(x1)
x1 = Conv2D(10, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x1)
x = Conv2D(10, (1, 1), padding='same')(x)  # Cambia la cantidad de canales a 10
x = Add()([x, x1])
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Segundo bloque residual
x1 = Conv2D(10, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
x1 = Activation('relu')(x1)
x1 = Conv2D(10, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x1)
x = Add()([x, x1])
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Tercer bloque residual
x1 = Conv2D(20, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
x1 = Activation('relu')(x1)
x1 = Conv2D(20, (3, 3), padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x1)
x = Conv2D(20, (1, 1), padding='same')(x)  # Cambia la cantidad de canales a 20
x = Add()([x, x1])
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(64, kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)

model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

model.fit(      train,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test,
                validation_steps=test_steps
                )

model.save('cvsd.h5')
