import os
import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical, load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


label_map = {
    "SepalLengthCm": 0,
    "SepalWidthCm": 1,
    "PetalLengthCm": 2,
    "PetalWidthCm": 3
}


image_dir = "/Users/rabiagnbs/Desktop/Code/VeriBilimi/PycharmProjects/pythonProject/AsimetrikSifreleme/kırpılmış_resimler"


images = []
labels = []


for file_name in os.listdir(image_dir):
    for label_name, label_id in label_map.items():
        if label_name in file_name:

            img_path = os.path.join(image_dir, file_name)
            img = load_img(img_path, target_size=(128,128), color_mode="rgb")
            images.append(img_to_array(img))
            labels.append(label_id)


images = np.array(images)
labels = np.array(labels)

labels = to_categorical(labels, num_classes=len(label_map))
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)



model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(len(label_map), activation='softmax')
])

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

sample_image = X_test[0].reshape(1, 128, 128, 3)
predicted_class = model.predict(sample_image).argmax()
print(f"Predicted Class: {predicted_class}")

train_acc_mean = np.mean(history.history['accuracy'])
val_acc_mean = np.mean(history.history['val_accuracy'])

print(f"Ortalama Train Accuracy: {train_acc_mean:.4f}")
print(f"Test Accuracy: {test_acc}")


model.save('iris_model.keras')


model_json = model.to_json()


with open("../iris_structure.json", "w") as json_file:
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
"""
# Test verisinden bir örnek alalım (ilk örnek)
test_image_original = X_test[0]

# Modelin tahminini alalım (model, 0-1 arasında olasılık skorları döndürür)
results = model.predict(np.expand_dims(test_image_original, axis=0))

# Sınıf isimleri (class_names)
class_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Bar grafiği için indeksler ve renkler
ind = 0.1 + 0.6 * np.arange(len(class_names))
width = 0.4
color_list = ['red', 'darkorange', 'limegreen', 'navy']


for i in range(len(class_names)):
    plt.bar(ind[i], results[0][i], width, color=color_list[i])


plt.title("Sınıflandırma Sonuçları", fontsize=20)
plt.xlabel("Özellikler Kategorisi", fontsize=16)
plt.ylabel("Sınıflandırma Skoru", fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)


plt.show()

# En yüksek skorlu sınıfı yazdıralım
prin"("Sınıflandırma sonucu en yüksek oranla:", class_names[np.argmax(results)])
"""