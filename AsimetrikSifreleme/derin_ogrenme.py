import os
import numpy as np
from keras.src.models import Sequential
import tensorflow
from keras.src.layers import InputLayer, Dense, MaxPooling2D, Conv2D, Flatten
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt
import random

# Klasördeki tüm png resimleri oku
image_paths = [os.path.join('AsimetrikSifreleme/heatmap_images', fname) for fname in os.listdir('AsimetrikSifreleme/heatmap_images') if fname.endswith('.png')]

# Resimleri yüklemek ve ön işleme yapmak
def load_image(image_path):
    img = tensorflow.io.read_file(image_path)
    img = tensorflow.image.decode_png(img, channels=3)  # RGB görüntü
    img = tensorflow.image.resize(img, [224, 224])  # İstenilen boyutta yeniden boyutlandır
    img = img / 255.0  # Piksel değerlerini normalize et
    return img

# Resim dosya adlarını sınıf etiketlerine eşlemek
def get_class_from_filename(filename):
    for class_name in class_names:
        if class_name in filename:
            return class_name
    return "Unknown"  # Eğer eşleşme bulunamazsa, 'Unknown' sınıfı döndürülür

# Klasördeki tüm png resimlerini oku ve etiketleri oluştur
class_names = ["PetalLengthCm_Row", "PetalWidthCm_Row", "SepalLengthCm_Row", "SepalWidthCm_Row", "SpeciesEncoding_Row"]

# Resimleri yükleyip birleştir
images = [load_image(img_path) for img_path in image_paths]
images = tensorflow.stack(images)  # Resimleri bir tensöre dönüştür
images = images.numpy()  # NumPy dizisine dönüştür

# Etiketleri dosya adlarından al
labels = np.array([get_class_from_filename(os.path.basename(img_path)) for img_path in image_paths])

# Etiketleri kategorik hale getirme
labels = np.array([class_names.index(label) for label in labels])  # Etiketleri sayısal hale getir
labels = to_categorical(labels, num_classes=len(class_names))  # Kategorik hale getirme


# Modelin çıkış katmanını bu sınıf sayısına göre ayarlama
model = Sequential([
    InputLayer(input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # Çıkış katmanı sınıf sayısına göre ayarlandı
])


# Modeli derleme
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Modeli eğit
model.fit(images, labels, epochs=10, batch_size=32)

# Eğitim tamamlandıktan sonra modeli kaydetme
model.save('heatmap_model.keras')

# Modeli yükleme (önceden eğitilmiş modeliniz varsa bu satırı kullanabilirsiniz)
model = load_model('heatmap_model.keras')

# Test için görsellerden rastgele bir tahmin gerçekleştirme
def prepare_image(image):
    img = image.astype('float32')
    img = img / 255.0
    return img.reshape(1, 224, 224, 3)
# Test için görsellerden rastgele bir tahmin gerçekleştirme
plt.figure(figsize=(16, 16))

right = 0
mistake = 0
prediction_num = 20  # 20 resim üzerinde test edilecek

for i in range(prediction_num):
    index = random.randint(0, len(images) - 1)
    image = images[index]
    data = prepare_image(image)

    plt.subplot(5, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')

    ret = model.predict(data, batch_size=1)
    predicted_class = np.argmax(ret[0])  # Tahmin edilen sınıfın indeksi
    actual_class = np.argmax(labels[index])  # Gerçek sınıfın indeksi

    predicted_class_name = class_names[predicted_class]  # Sınıf adı

    if actual_class == predicted_class:
        plt.title(f"Sınıf: {predicted_class_name}\n", fontsize=12)  # Sınıf adı
        right += 1
    else:
        plt.title(f"{predicted_class_name} \n != {class_names[actual_class]}", color='#ff0000', fontsize=12)  # Hatalı tahmin
        mistake += 1


plt.show()
print("Doğru tahminlerin sayısı:", right)
print("Hata sayısı:", mistake)
print("Doğru tahmin oranı:", right / (mistake + right) * 100, '%')



# Yeni bir görüntüyü modelin tahmin edebilmesi için ön işleme tabi tutma
def preprocess_image(image_path):
    img = tensorflow.io.read_file(image_path)
    img = tensorflow.image.decode_png(img, channels=3)  # RGB görüntü
    img = tensorflow.image.resize(img, [224, 224])  # Modelin kabul ettiği boyuta ayarlayın
    img = img / 255.0  # Normalleştirme
    img = np.expand_dims(img, axis=0)  # Modelin beklediği 4D tensör boyutuna getirme (1, 224, 224, 3)
    return img

# Tahmin yapılacak görüntü
image_path = '/Users/rabiagnbs/Desktop/Code/VeriBilimi/PycharmProjects/pythonProject/AsimetrikSifreleme/heatmap_images/PetalLengthCm_Row_0.png'
image = preprocess_image(image_path)

# Modelin tahmini
prediction = model.predict(image)
predicted_class_index = np.argmax(prediction)  # Tahmin edilen sınıfın indeksi
predicted_class_name = class_names[predicted_class_index]  # Sınıf adını almak için

# Tahmin sonucunu görselleştirme
plt.imshow(tensorflow.image.decode_png(tensorflow.io.read_file(image_path), channels=3))
plt.axis('off')
plt.title(f"Tahmin edilen sınıf: {predicted_class_name}")
plt.show()

print("Tahmin edilen sınıf:", predicted_class_name)
