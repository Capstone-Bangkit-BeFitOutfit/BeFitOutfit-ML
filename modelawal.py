import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Parameter
num_classes = 10 
input_shape = (28, 28, 3)

# Data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  
        target_size=(28, 28),
        batch_size=64,
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(28, 28),
        batch_size=64,
        class_mode='categorical')

# Model 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train
model.fit(train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator, 
        validation_steps=len(validation_generator),  
        epochs=10)

# Evaluate on test set  
test_generator = test_datagen.flow_from_directory(
        'data/test', 
        target_size=(28, 28),
        batch_size=64,
        class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', test_acc)

# Save model
model.save('model.h5')