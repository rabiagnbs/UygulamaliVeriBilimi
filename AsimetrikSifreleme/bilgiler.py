import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
import struct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import struct
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

pd.set_option('display.max_columns', None) #Veri setinin tüm sütunlarını görüntüler.
pd.set_option('display.max_rows', None) #Veri Setinin tüm satırlarını görüntüler
pd.set_option('display.float_format', lambda x: '%.3f' % x) #Veri setindeki ondalıklı sayıların virgülden  sonraki 3 hanesini görüntüler.
pd.set_option('display.width', 500) #Konsolda 500 karakter gözükecek şekilde ayarlar.

df = pd.read_csv("/Users/rabiagnbs/Desktop/Iris.csv") #Veri okunur.
df.head() #Veri setinin ilk 5 satırı listelenir.

#Petal lenght taç yaprağı uzunluğu, petal width taç yaprağı genişliği
#Sepal length çanak yaprağı uzunluğu sepal width çanak yaprağı genişliği
#species türler.


#outlier_threasholds ile veri setindeki değerlerin alt ve üst satırını belirliyor.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range,3)
    low_limit = round(quartile1 - 1.5 * interquantile_range,3)
    return low_limit, up_limit



for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, outlier_thresholds(df, col))


#Aykırı değer var mı yok mu onun kontrolü yapılır.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, check_outlier(df, col))

#Aykırı değerleri gösteren fonksiyon
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if not outliers.empty:
        print(outliers)

grab_outliers(df, "SepalWidthCm")


#Aykırı değerlere baskılama yöntemi ile ayırt etme yöntemi
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "SepalWidthCm")

#Label Encoder ile Hedef Değişkenini string yapıdan Ordinal Kategorik değişkene çeviriyorum.
le = LabelEncoder()
df['Species_Encoding'] = le.fit_transform(df['Species']) + 1
df.head()

#YSA için Girdi ve çıktı verilerini seçiyorum
X=df.iloc[:, 1:5]
X.head()

y=df["Species_Encoding"]
y.head()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=17)

#Burada standarizasyon yapılır. Standarizasyon Verinin ortalamasını 0 standart sapmasını 1 olarak ayarlar.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # X_train üzerinde fit ve transform işlemi
X_test = scaler.transform(X_test)


#MLPClassifier ile model oluşumu gerçekleşiyor.
#hidden_layer_sizes gizli katmandaki nöron sayısını temsil eder.
mlpcl= MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='logistic',max_iter=10000)
#Model eğitim verisi ile eğitiliyor.
mlpcl.fit(X_train, y_train.values.ravel())
#Model Test Verisi üzerinden tahmin yapıyor.
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

#precision= TP/ TP+ FP
#recall = TP / TP + FN
#f1_score precision ve recall sonuçlarının harmonik ortalamasıdır.

#Yeni bir df oluşturuyorum gerçek ve tahmin edilen sonuçları buluyorum.
comparison_df = pd.DataFrame({
    'Gerçek Sonuçlar': y_test.values,
    'Tahmin Edilen Sonuçlar': predictions
})

#Başarı metrikleri ile modelin başarısını test ediyorum.
MSE = np.square(np.subtract(y_test, predictions)).mean()
RMSE = np.sqrt(MSE)
print(MSE, RMSE)

"""
private_key_a = ec.generate_private_key(ec.SECP256R1(), default_backend())  #Burada A için SECP256R1 eğrisine ait bir özel anahtar oluşturuluyor.
public_key_a = private_key_a.public_key() # A için Genel anahtar oluşturuluyor.

private_key_b = ec.generate_private_key(ec.SECP256R1(), default_backend()) #Burada B için SECP256R1 eğrisine ait bir özel anahtar oluşturuluyor.
public_key_b = private_key_b.public_key() # B için Genel anahtar oluşturuluyor.


shared_key_a = private_key_a.exchange(ec.ECDH(), public_key_b) #A kendi özel anahtarını B'nin genel anahtarını kullanarak paylaşılan anahtarı üretiyor.
shared_key_b = private_key_b.exchange(ec.ECDH(), public_key_a) #B kendi özel anahtarını A'nın genel anahtarını kullanarak paylaşılan anahtarı üretiyor.
#Bu iki anahtar birbirine eşit olacaktır.

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=os.urandom(16),
    iterations=10000,
    backend=default_backend()
)
aes_key = kdf.derive(shared_key_a) #burada shared_key ile AES anahtarı türetiliyor. Bu anahtar şifreleme ve şifre çözmede kullanılacak.
#hashes.SHA256() algoritması ile paylaşılan anahtar, 256- bit uzunluğunda bir anahtara dönüştürülür.
#salt: her türetme işleminde rastgele bir 16 byte verisi ekleyerek, aynı anahtarın tekrar kullanılmasını zorlaştırır.
#iterations: anahtar türetme işleminin güvenliğini artırmak için iterasyonu arttırırız.


def encrypt_value(key, value):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_value = iv + encryptor.update(struct.pack('>d', value)) + encryptor.finalize()
    return encrypted_value
#os.urandom(16) fonksiyonu ile 16 byte uzunluğunda bir iv rastgele oluşturur. Iv şifreleme işleminde kullanılan başlangıç değeridir ve her şifreleme için farklı olmalıdır. Bu değer string int karışıktır.
#AES şifreleme algoritması CFB modu ile başlatılır.
#encryptor.update() metodu şifrelenecek işlemi alır ve şifreleme işlemi başlatılır.
#encryptor.finalize() işlemi ile şifreleme tamamlanır.
#encrypt_value() şifrelenmiş veri ve iv birleşimidir.

def decrypt_value(key, encrypted_value):
    iv = encrypted_value[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_value = decryptor.update(encrypted_value[16:]) + decryptor.finalize()
    return struct.unpack('>d', decrypted_value)[0]
#şifreli veri, ilk 16 byte'ı IV olarak ayırır.
#AES algoritması ve CFB modu ile şifreli veri çözülür.
#decryptor.update() ile şifre çözme işlemi yapılır ve ardından
# decryptor.finalize() ile işlem tamamlanır.
#struct.unpack('>d', decrypted_value)[0] orijinal veriyi tekrar float tipinde veriye dönüştürür.


df_sifreli= pd.DataFrame()
df_sifreli['encrypted_SepalLengthCm'] = df['SepalLengthCm'].apply(lambda x: encrypt_value(aes_key, x))
df_sifreli['encrypted_SepalWidthCm'] = df['SepalWidthCm'].apply(lambda x: encrypt_value(aes_key, x))
df_sifreli['encrypted_PetalLengthCm'] = df['PetalLengthCm'].apply(lambda x: encrypt_value(aes_key, x))
df_sifreli['encrypted_PetalWidthCm'] = df['PetalWidthCm'].apply(lambda x: encrypt_value(aes_key, x))
df_sifreli['encrypted_SpeciesEncoding'] = df['Species_Encoding'].apply(lambda x: encrypt_value(aes_key, x))
df_sifreli.head()

first_encrypted_value = df_sifreli['encrypted_SpeciesEncoding'].iloc[1]
length_first_encrypted_value = len(first_encrypted_value)
print(length_first_encrypted_value)

#Verileri ('er alıp binary veri tipine dönüştüren fonksiyon.
def convert_to_binary(encrypted_data):
    return ''.join(f'{byte:08b}' for byte in encrypted_data)

df_binary=pd.DataFrame()
df_binary['binary_SepalLengthCm'] = df_sifreli['encrypted_SepalLengthCm'].apply(convert_to_binary)
df_binary['binary_SepalWidthCm'] = df_sifreli['encrypted_SepalWidthCm'].apply(convert_to_binary)
df_binary['binary_PetalLengthCm'] = df_sifreli['encrypted_PetalLengthCm'].apply(convert_to_binary)
df_binary['binary_PetalWidthCm'] = df_sifreli['encrypted_PetalWidthCm'].apply(convert_to_binary)
df_binary['binary_SpeciesEncoding'] = df_sifreli['encrypted_SpeciesEncoding'].apply(convert_to_binary)
df_binary.head()


first_binary_value = df_binary['binary_SpeciesEncoding'].iloc[1]
length_first_binary_value = len(first_binary_value)
print(length_first_binary_value)

def visualize_binary_values_heatmap(dataframe):
    for index, row in dataframe.iterrows():
        binary_values = {
            'SepalLengthCm': row['binary_SepalLengthCm'],
            'SepalWidthCm': row['binary_SepalWidthCm'],
            'PetalLengthCm': row['binary_PetalLengthCm'],
            'PetalWidthCm': row['binary_PetalWidthCm'],
            'SpeciesEncoding': row['binary_SpeciesEncoding']
        }

        for feature, binary_value in binary_values.items():
            # Binary stringden bit değerlerini çıkaralım
            bits = [int(bit) for bit in binary_value]
            # Isı haritasının boyutunu ayarlayalım
            grid_size = int(np.sqrt(len(bits)))
            reshaped_bits = np.array(bits[:grid_size * grid_size]).reshape(grid_size, grid_size)

            # Çok renkli ısı haritasını çizelim, annot'u kapatıp yeni bir cmap kullanalım
            plt.figure(figsize=(5, 5))
            sns.heatmap(reshaped_bits, cmap='Spectral')
            plt.title(f"{feature} - Row {index}")
            plt.show()

#visualize_binary_values_heatmap(df_binary)
"""
#RSA algoritmasını kullarak bir private key üretir.
#Key size rsa uzunluğunu belirtir.
#public_exponent açık anahtar oluşturulurken kullanılan üstel değeri belirtir.
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()#özel anahtarı kullanarak genel anahtar üretir.


pem_private_key = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption()
)
#Özel anahtarı PEM formatında dışa aktarır.PEM bir anahtar formatıdır.


pem_public_key = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
#Genel anahtarı PEM formatında dışa aktarır.


def number_to_bytes(number):
    return struct.pack('>d', number)  # Double (float64) formatında baytlara çeviriyoruz



def encrypt_message(public_key, number):
    byte_data = number_to_bytes(number)
    encrypted = public_key.encrypt(
        byte_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted

#bu fonksiyon verilen bir sayıyı genel anahtar ile şifreler.
#ilk olarak number sayısı bytlara dönüştürülür. Ardından bayt verisi genel anahtarı kullanarak şifrelenir.
#padding.MGF1(algorithm=hashes.SHA256()) mesaj şifrleme fonksiyonunun MGF1 ile SHA-256 algoritmasını kullanmasını ister.

def decrypt_message(private_key, encrypted_data):
    # Şifrelenmiş veriyi özel anahtar ile çöz
    decrypted_data = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    # Çözülen baytları double (float64) formatında sayıya dönüştür
    original_number = struct.unpack('>d', decrypted_data)[0]
    return original_number

df_sifreli = pd.DataFrame()
df_sifreli['encrypted_SepalLengthCm'] = df['SepalLengthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_SepalWidthCm'] = df['SepalWidthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_PetalLengthCm'] = df['PetalLengthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_PetalWidthCm'] = df['PetalWidthCm'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli['encrypted_Species_Encoding'] = df['Species_Encoding'].apply(lambda x: encrypt_message(public_key, x))
df_sifreli.head()

first_encrypted_value = df_sifreli['encrypted_SpeciesEncoding'].iloc[1]
length_first_encrypted_value = len(first_encrypted_value)
print(length_first_encrypted_value)

def convert_to_binary(encrypted_data):
    return ''.join(f'{byte:08b}' for byte in encrypted_data)


df_binary = pd.DataFrame()
df_binary['binary_SepalLengthCm'] = df['encrypted_SepalLengthCm'].apply(convert_to_binary)
df_binary['binary_SepalWidthCm'] = df['encrypted_SepalWidthCm'].apply(convert_to_binary)
df_binary['binary_PetalLengthCm'] = df['encrypted_PetalLengthCm'].apply(convert_to_binary)
df_binary['binary_PetalWidthCm'] = df['encrypted_PetalWidthCm'].apply(convert_to_binary)
df_binary['binary_Species_Encoding'] = df['encrypted_Species_Encoding'].apply(convert_to_binary)
df_binary.head()


first_binary_value = df_binary['binary_Species_Encoding'].iloc[1]
length_of_first_value = len(first_binary_value)

print(length_of_first_value)

