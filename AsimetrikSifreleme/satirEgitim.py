import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D, Conv2D, \
    Dropout, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.api.regularizers import l2


base_dir = '/Users/rabiagnbs/Desktop/Code/VeriBilimi/PycharmProjects/pythonProject/AsimetrikSifreleme/satirResimler'
label_map = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}


img_size = (376, 94)

images = []
labels = []

for label, class_name in label_map.items():
    class_dir = os.path.join(base_dir, str(label))
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = Image.open(img_path)
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        images.append(img_array)
        labels.append(label)


images = np.array(images)
labels = np.array(labels)


labels = to_categorical(labels, num_classes=len(label_map))


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = Sequential([
    Input(shape=(376, 94, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.4),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),


    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

sample_image = X_test[0].reshape(1, 376, 94, 3)
predicted_class = model.predict(sample_image).argmax()
print(f"Predicted Class: {predicted_class}")

train_acc_mean = np.mean(history.history['accuracy'])
val_acc_mean = np.mean(history.history['val_accuracy'])

print(f"Ortalama Train Accuracy: {train_acc_mean:.4f}")
print(f"Test Accuracy: {test_acc}")

model.save('iris_model.keras')

model_json = model.to_json()
with open("iris_structure.json", "w") as json_file:
    json_file.write(model_json)

plt.figure(figsize=(14, 3))
plt.subplot(1, 2, 1)
plt.suptitle('Train', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], color='r', label='Training Loss')
plt.plot(history.history['val_loss'], color='b', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], color='g', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='m', label='Validation Accuracy')
plt.legend(loc='lower right')

plt.show()
