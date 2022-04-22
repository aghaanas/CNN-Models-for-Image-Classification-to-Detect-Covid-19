# Solution by the following group members
# 1. Agha Ahmad (501119910)
# 2. Hina Shafique Awan (501118831)

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report

# Loading Dataset

TRAIN_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train"
TEST_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/test"

TRAIN_COVID_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/COVID19"
TRAIN_NORMAL_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/NORMAL"
TRAIN_PNE_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/PNEUMONIA"

VAL_NORMAL_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/test/COVID19"
VAL_PNEU_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/test/NORMAL"
VAL_COVID_PATH = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/test/PNEUMONIA"

# Image Preprocessing

train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
)
test_datagen = image.ImageDataGenerator(rescale = 1./255)

# Training Data Generation

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')

# Test Data Generation

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')


# ALEXNET Model

model = Sequential([
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Model Compilation

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 100
stepsperepoch=9
validationsteps=1
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
mc = ModelCheckpoint("alexnet.h5", monitor='val_loss',save_best_only=True, mode='min',verbose=1)

# Model Fitting

hist = model.fit_generator(
    train_generator,
    epochs=epochs,
    callbacks=[annealer,mc,es],
    steps_per_epoch=stepsperepoch,
    validation_data=test_generator,
    validation_steps = validationsteps
)

# Model Evaluating

preds = model.evaluate(test_generator)
print ("Validation Loss = " + str(preds[0]))
print ("Validation Accuracy = " + str(preds[1]))

# Plotting Accuracy and Loss

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('ALEXNET Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(["Train_acc","Validation_acc"])
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('ALEXNET Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["Train_loss","Validation Loss"])
plt.show()

# Classification Report

predictions = model.predict(test_generator)
pred_labels = np.argmax(predictions, axis = 1)
print(classification_report(test_generator.classes, pred_labels))