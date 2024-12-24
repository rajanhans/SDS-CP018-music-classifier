import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import numpy as np

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Assuming 10 genres

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("CNN model initialized successfully.")

# Define directories
train_dir = r'D:\SDS\AUDIO CLASSIFICATION\Spectrograms\train'
test_dir = r'D:\SDS\AUDIO CLASSIFICATION\Spectrograms\test'

# Check if the directories exist and are not empty
if not os.path.exists(train_dir):
    raise ValueError(f"Directory {train_dir} does not exist.")
if not os.listdir(train_dir):
    raise ValueError(f"Directory {train_dir} is empty.")

if not os.path.exists(test_dir):
    raise ValueError(f"Directory {test_dir} does not exist.")
if not os.listdir(test_dir):
    raise ValueError(f"Directory {test_dir} is empty.")

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

# Normalization for testing
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Predict the classes for the test set
Y_pred = model.predict(validation_generator, validation_generator.samples // validation_generator.batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true classes
y_true = validation_generator.classes

# Print the classification report
print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true[:len(y_pred)], y_pred, target_names=target_names))
