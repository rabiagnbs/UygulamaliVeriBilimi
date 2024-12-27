import numpy as np
import pandas as pd
import os
import struct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("/Users/rabiagnbs/Desktop/Iris.csv")
df.head()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range,3)
    low_limit = round(quartile1 - 1.5 * interquantile_range,3)
    return low_limit, up_limit

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, outlier_thresholds(df, col))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if not outliers.empty:
        print(outliers)

grab_outliers(df, "SepalWidthCm")

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "SepalWidthCm")

le = LabelEncoder()

df['Species_Encoding'] = le.fit_transform(df['Species']) + 1
df.head()

X=df.iloc[:, 1:5]
X.head()

y=df.iloc[:, 6]
y.head()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)
scaler= StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlpcl= MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=10000)
mlpcl.fit(X_train, y_train.values.ravel())
predictions =mlpcl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

comparison_df = pd.DataFrame({
    'Gerçek Sonuçlar': y_test.values,
    'Tahmin Edilen Sonuçlar': predictions
})
MSE = np.square(np.subtract(y_test, predictions)).mean()
y.mean()
y.std()
np.sqrt(MSE)

private_key_a = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key_a = private_key_a.public_key()

private_key_b = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key_b = private_key_b.public_key()


shared_key_a = private_key_a.exchange(ec.ECDH(), public_key_b)
shared_key_b = private_key_b.exchange(ec.ECDH(), public_key_a)


kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=os.urandom(16),
    iterations=100000,
    backend=default_backend()
)
aes_key = kdf.derive(shared_key_a)


def encrypt_value(key, value):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_value = iv + encryptor.update(struct.pack('>d', value)) + encryptor.finalize()
    return encrypted_value


def decrypt_value(key, encrypted_value):
    iv = encrypted_value[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_value = decryptor.update(encrypted_value[16:]) + decryptor.finalize()
    return struct.unpack('>d', decrypted_value)[0]


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

def convert_to_binary(encrypted_data):
    return ''.join(f'{byte:08b}' for byte in encrypted_data)

df_binary=pd.DataFrame()
df_binary['binary_SepalLengthCm'] = df_sifreli['encrypted_SepalLengthCm'].apply(convert_to_binary)
df_binary['binary_SepalWidthCm'] = df_sifreli['encrypted_SepalWidthCm'].apply(convert_to_binary)
df_binary['binary_PetalLengthCm'] = df_sifreli['encrypted_PetalLengthCm'].apply(convert_to_binary)
df_binary['binary_PetalWidthCm'] = df_sifreli['encrypted_PetalWidthCm'].apply(convert_to_binary)
df_binary.head()

# Encrypted DataFrame'i kaydet
df_sifreli.to_excel("encrypted_data.xlsx", index=False)
print("Encrypted data has been saved to 'encrypted_data.xlsx'")

# Binary DataFrame'i kaydet
df_binary.to_excel("binary_data.xlsx", index=False)
print("Binary data has been saved to 'binary_data.xlsx'")


""""
def visualize_binary_values(dataframe):
    for index, row in dataframe.iterrows():
        binary_values = {
            'SepalLengthCm': row['binary_SepalLengthCm'],
            'SepalWidthCm': row['binary_SepalWidthCm'],
            'PetalLengthCm': row['binary_PetalLengthCm'],
            'PetalWidthCm': row['binary_PetalWidthCm'],
            'SpeciesEncoding': row['binary_SpeciesEncoding']
        }
        for feature, binary_value in binary_values.items():
            bits = [int(bit) for bit in binary_value]
            grid_size = int(np.sqrt(len(bits)))
            reshaped_bits = np.array(bits[:grid_size * grid_size]).reshape(grid_size, grid_size)

            plt.figure(figsize=(5, 5))
            plt.imshow(reshaped_bits, cmap='binary')
            plt.title(f'Pixel Grid of {feature} - Row {index}')
            plt.tight_layout()
            plt.show()

#visualize_binary_values(df_binary)


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


def visualize_single_column_heatmaps(dataframe):
    for feature in dataframe.columns:
        # Sütunun tüm satırları için bit değerlerini al
        binary_values = dataframe[feature].apply(lambda x: [int(bit) for bit in x])

        # Bit değerlerini bir matrise dönüştür
        bits_all_rows = np.array(binary_values.tolist())

        # Görselleştirme
        plt.figure(figsize=(
        12, min(0.5 * len(bits_all_rows), 10)))  # Boyutu ayarla, yüksek satır sayıları için optimize edilir
        sns.heatmap(bits_all_rows, cmap='Spectral', cbar=False,
                    yticklabels=[f"{i}" for i in range(len(bits_all_rows))])
        plt.title(f"{feature} Özelliği için Isı Haritası")
        plt.xlabel("Bit Sırası")
        plt.ylabel("Satır Numarası")
        plt.show()


# Her bir sütun için tek bir ısı haritası çizdirin
#visualize_single_column_heatmaps(df_binary)


import os


def save_heatmap_images(dataframe, save_path="/Users/rabiagnbs/Desktop/Code/VeriBilimi/PycharmProjects/pythonProject/heatmap_sutun"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for feature in dataframe.columns:
        binary_values = dataframe[feature].apply(lambda x: [int(bit) for bit in x])
        bits_all_rows = np.array(binary_values.tolist())

        plt.figure(figsize=(5, 5))
        sns.heatmap(bits_all_rows, cmap='Spectral', cbar=False)
        plt.title(f"{feature}")
        plt.savefig(f"{save_path}/{feature}.png")
        plt.close()


# Görselleri kaydetmek için fonksiyonu çağırın
save_heatmap_images(df_binary)


"""